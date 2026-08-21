"""
Fase 10 (NBA) -- v4 del modelo de margen: agrega 4 regresores GSSD (Game
Scores Standard Deviation) sobre v3 (Elo + calendario). Ver
`add_nba_gssd_features.py` para la definicion completa y la adaptacion
walk-forward-safe del modelo GSSD del libro de Andrew Mack.

**Por que agregar esto sobre Elo**: Elo (ajustado por MOV) ya captura
"quien gana y por cuanto" de forma agregada, pero no separa OFENSA de
DEFENSA -- dos equipos con el mismo Elo pueden llegar ahi por caminos muy
distintos (un equipo de ritmo rapido y defensa floja vs. un equipo lento y
defensivo). GSSD aporta esa descomposicion directamente desde los puntos
anotados/recibidos reales, no desde el resultado agregado.

**Screening previo (OLS full-sample, controlando por elo_diff + b2b_diff +
3in4_diff + home_is_denver ya confirmados), ANTES de construir este
archivo** -- las 4 variables son significativas Y con el signo
teoricamente correcto (chequeo de sanidad, no solo p-valor):
- `home_off_l10` (ataque del local, ultimos 10 de local): coef=+0.1157,
  t=7.463, p<0.0001 -- mas ataque local = mas margen. Signo correcto.
- `home_def_l10` (puntos recibidos por el local, ultimos 10 de local):
  coef=-0.1231, t=-8.090, p<0.0001 -- mas puntos recibidos = menos
  margen (peor defensa local perjudica). Signo correcto.
- `away_off_l10` (ataque del visitante, ultimos 10 de visitante):
  coef=-0.0668, t=-4.259, p<0.0001 -- mas ataque visitante = menos
  margen para el local. Signo correcto.
- `away_def_l10` (puntos recibidos por el visitante, ultimos 10 de
  visitante): coef=+0.0446, t=2.747, p=0.006 -- mas puntos recibidos por
  el visitante (peor defensa visitante) = mas margen para el local. Signo
  correcto.
R^2 full-sample sube de 0.177 (v3) a 0.180 con estas 4 variables. El
numero de condicion (2.8e3) se marco como aviso de multicolinealidad
moderada (esperable, Elo correlaciona con estas variables via el
multiplicador MOV) -- no anula la significancia individual de cada
regresor, no se descarta por esto.

Requiere que games_clean.csv ya tenga home_elo/away_elo
(`add_nba_elo_features.py`), home_rest/away_rest/home_3in4/away_3in4
(`add_nba_rest_features.py`) Y home_off_l10/home_def_l10/away_off_l10/
away_def_l10 (`add_nba_gssd_features.py`).

Este archivo NO se corre standalone -- lo importa `backtest_nba_v4.py`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = [
    "home_elo", "away_elo", "point_margin", "home_rest", "away_rest",
    "home_3in4", "away_3in4", "home_team",
    "home_off_l10", "home_def_l10", "away_off_l10", "away_def_l10",
]
FEATURE_COLS = [
    "elo_diff", "b2b_diff", "3in4_diff", "home_is_denver",
    "home_off_l10", "home_def_l10", "away_off_l10", "away_def_l10",
]
DENVER_TEAM_NAME = "Denver Nuggets"


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre 'python -m src.processing.add_nba_elo_features', "
            f"'python -m src.processing.add_nba_rest_features' y "
            f"'python -m src.processing.add_nba_gssd_features' sobre games_clean.csv."
        )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # OJO: "NaN == 0" da False en pandas/numpy (no NaN) -- se fuerza la
    # propagacion de NaN explicitamente con np.where (mismo fix que en v2/v3).
    home_b2b = np.where(df["home_rest"].isna(), np.nan, (df["home_rest"] == 0).astype(float))
    away_b2b = np.where(df["away_rest"].isna(), np.nan, (df["away_rest"] == 0).astype(float))
    df["b2b_diff"] = home_b2b - away_b2b

    df["3in4_diff"] = df["home_3in4"] - df["away_3in4"]
    df["home_is_denver"] = (df["home_team"] == DENVER_TEAM_NAME).astype(float)

    # home_off_l10/home_def_l10/away_off_l10/away_def_l10 ya vienen
    # calculados (trailing, sin fuga) de add_nba_gssd_features.py -- se usan
    # tal cual, sin transformar, mismo criterio literal del modelo GSSD del
    # libro (PFH, PAH, PFA, PAA como regresores separados, no diferenciados).
    return df


def fit_margin_model(train_df: pd.DataFrame):
    """OLS: point_margin ~ elo_diff + b2b_diff + 3in4_diff + home_is_denver +
    home_off_l10 + home_def_l10 + away_off_l10 + away_def_l10. Devuelve el
    modelo ajustado y sigma (desvio estandar de los residuos del propio
    training set)."""
    _check_columns(train_df)
    train_df = _add_features(train_df)

    before = len(train_df)
    train_df = train_df.dropna(subset=FEATURE_COLS + ["point_margin"])
    dropped = before - len(train_df)
    if dropped:
        print(f"  [INFO] {dropped} partidos de training sin features calculables (primer partido de una "
              f"franquicia, o menos de 3 partidos previos de local/visitante para GSSD) -- excluidos del "
              f"entrenamiento, no rellenados.")

    X = sm.add_constant(train_df[FEATURE_COLS])
    y = train_df["point_margin"]
    model = sm.OLS(y, X).fit()

    sigma = float(model.resid.std(ddof=1))
    return model, sigma


def predict_dataframe(model, sigma: float, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo a un DataFrame completo, vectorizado. Filas sin
    features calculables quedan con prediccion NaN (explicito), no se
    inventa un valor."""
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
