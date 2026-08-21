"""
Fase 10 (NBA) -- v5 del modelo de margen: reemplaza los 4 regresores GSSD
de promedio movil simple (v4, `_l10`) por su version de promedio
exponencial (`_ewm`, span=15) -- ver `add_nba_gssd_features.py` para el
barrido completo de ventanas/spans que motivo este cambio (EWM le gano a
la ventana simple en TODOS los spans probados, span=15 es el optimo).

No es una variable nueva -- es la MISMA idea de v4 (ataque/defensa
reciente de local/visitante) con una mejor forma de pesar "reciente".
Arquitectura identica a v4 en todo lo demas (mismos elo_diff, b2b_diff,
3in4_diff, home_is_denver).

Requiere que games_clean.csv ya tenga home_elo/away_elo, home_rest/away_rest/
home_3in4/away_3in4, Y home_off_ewm/home_def_ewm/away_off_ewm/away_def_ewm
(`add_nba_gssd_features.py`, version con EWM).

Este archivo NO se corre standalone -- lo importa `backtest_nba_v5.py`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = [
    "home_elo", "away_elo", "point_margin", "home_rest", "away_rest",
    "home_3in4", "away_3in4", "home_team",
    "home_off_ewm", "home_def_ewm", "away_off_ewm", "away_def_ewm",
]
FEATURE_COLS = [
    "elo_diff", "b2b_diff", "3in4_diff", "home_is_denver",
    "home_off_ewm", "home_def_ewm", "away_off_ewm", "away_def_ewm",
]
DENVER_TEAM_NAME = "Denver Nuggets"


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre 'python -m src.processing.add_nba_elo_features', "
            f"'python -m src.processing.add_nba_rest_features' y "
            f"'python -m src.processing.add_nba_gssd_features' (version con EWM) sobre games_clean.csv."
        )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # OJO: "NaN == 0" da False en pandas/numpy (no NaN) -- se fuerza la
    # propagacion de NaN explicitamente con np.where (mismo fix que en v2-v4).
    home_b2b = np.where(df["home_rest"].isna(), np.nan, (df["home_rest"] == 0).astype(float))
    away_b2b = np.where(df["away_rest"].isna(), np.nan, (df["away_rest"] == 0).astype(float))
    df["b2b_diff"] = home_b2b - away_b2b

    df["3in4_diff"] = df["home_3in4"] - df["away_3in4"]
    df["home_is_denver"] = (df["home_team"] == DENVER_TEAM_NAME).astype(float)

    # home_off_ewm/home_def_ewm/away_off_ewm/away_def_ewm ya vienen
    # calculados (EWM span=15, sin fuga) de add_nba_gssd_features.py.
    return df


def fit_margin_model(train_df: pd.DataFrame):
    """OLS: point_margin ~ elo_diff + b2b_diff + 3in4_diff + home_is_denver +
    home_off_ewm + home_def_ewm + away_off_ewm + away_def_ewm."""
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
