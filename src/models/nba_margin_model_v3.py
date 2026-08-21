"""
Fase 10 (NBA) -- v3 del modelo de margen: agrega DOS regresores nuevos sobre
v2 (Elo + back-to-back binario), screeneados con un t-test antes de correr
el walk-forward completo -- mismo criterio de rigor que descarto rest_diff
continuo y confirmo b2b_diff en v2.

**Screening previo (OLS full-sample, controlando por elo_diff y b2b_diff ya
confirmados), ANTES de construir este archivo**:
- `3in4_diff` (home_3in4 - away_3in4, indicador de "3 partidos en 4 noches",
  fatiga MAS severa que un simple back-to-back): coef=-0.3737, t=-3.086,
  **p=0.002 -- significativo** incluso controlando por b2b_diff.
- `home_is_denver` (dummy: el LOCAL es Denver Nuggets): coef=+1.1087,
  t=3.046, **p=0.002 -- significativo**. Efecto de altitud real y
  documentado en la industria de NBA (Denver juega a ~1,600m, ventaja de
  local mayor a la generica que ya captura el `HOME_ADVANTAGE=100` fijo de
  Elo para todos los equipos). Elegido por teoria (Bill Benter / literatura
  de NBA), NO por escanear los 30 equipos buscando el coeficiente mas
  significativo -- eso seria data snooping, esto es una hipotesis
  puntual confirmada con datos.

Ambas variables SOBREVIVEN controlando una por la otra y por Elo -- no son
la misma señal reempaquetada.

Requiere que games_clean.csv ya tenga home_elo/away_elo
(`add_nba_elo_features.py`) Y home_rest/away_rest/home_3in4/away_3in4
(`add_nba_rest_features.py`, version extendida). `home_is_denver` se deriva
directo de `home_team`, no necesita ninguna columna nueva.

Este archivo NO se corre standalone -- lo importa `backtest_nba_v3.py`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = ["home_elo", "away_elo", "point_margin", "home_rest", "away_rest",
                  "home_3in4", "away_3in4", "home_team"]
FEATURE_COLS = ["elo_diff", "b2b_diff", "3in4_diff", "home_is_denver"]
DENVER_TEAM_NAME = "Denver Nuggets"


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre 'python -m src.processing.add_nba_elo_features' "
            f"y 'python -m src.processing.add_nba_rest_features' (version con 3in4) sobre games_clean.csv."
        )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Binario de back-to-back -- ver nba_margin_model_v2.py. OJO: "NaN == 0"
    # da False en pandas/numpy (no NaN) -- se fuerza la propagacion de NaN
    # explicitamente con np.where (mismo fix que en v2).
    home_b2b = np.where(df["home_rest"].isna(), np.nan, (df["home_rest"] == 0).astype(float))
    away_b2b = np.where(df["away_rest"].isna(), np.nan, (df["away_rest"] == 0).astype(float))
    df["b2b_diff"] = home_b2b - away_b2b

    # 3-en-4-noches -- ya viene como 0.0/1.0/NaN calculado en
    # add_nba_rest_features.py, no hace falta el mismo fix de NaN==0 aca
    # porque la columna ya es 0.0/1.0/NaN directamente (no se recalcula con
    # una comparacion contra 0 en este archivo).
    df["3in4_diff"] = df["home_3in4"] - df["away_3in4"]

    df["home_is_denver"] = (df["home_team"] == DENVER_TEAM_NAME).astype(float)

    return df


def fit_margin_model(train_df: pd.DataFrame):
    """OLS: point_margin ~ elo_diff + b2b_diff + 3in4_diff + home_is_denver.
    Devuelve el modelo ajustado y sigma (desvio estandar de los residuos
    del propio training set)."""
    _check_columns(train_df)
    train_df = _add_features(train_df)

    before = len(train_df)
    train_df = train_df.dropna(subset=FEATURE_COLS + ["point_margin"])
    dropped = before - len(train_df)
    if dropped:
        print(f"  [INFO] {dropped} partidos de training sin features de descanso calculables (primer "
              f"partido de una franquicia) -- excluidos del entrenamiento, no rellenados.")

    X = sm.add_constant(train_df[FEATURE_COLS])
    y = train_df["point_margin"]
    model = sm.OLS(y, X).fit()

    sigma = float(model.resid.std(ddof=1))
    return model, sigma


def predict_dataframe(model, sigma: float, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo a un DataFrame completo, vectorizado. Filas sin
    features de descanso calculables quedan con prediccion NaN (explicito),
    no se inventa un valor."""
    _check_columns(df)
    feats = _add_features(df)
    X = sm.add_constant(feats[FEATURE_COLS], has_constant="add")

    out = pd.DataFrame(index=df.index)
    out["mu_margin"] = model.predict(X).values
    out["sigma_margin"] = sigma
    out["model_prob_home"] = 1.0 - norm.cdf(0.0, loc=out["mu_margin"], scale=out["sigma_margin"])
    out["model_prob_away"] = 1.0 - out["model_prob_home"]

    if "spread_line" in df.columns:
        out["model_prob_home_covers"] = 1.0 - norm.cdf(
            df["spread_line"].values, loc=out["mu_margin"], scale=out["sigma_margin"]
        )
        out["model_prob_away_covers"] = 1.0 - out["model_prob_home_covers"]

    return out
