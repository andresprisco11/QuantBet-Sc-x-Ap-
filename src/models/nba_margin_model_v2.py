"""
Fase 10 (NBA) -- v2 del modelo de margen: agrega una variable de descanso
como segundo regresor de OLS. Arquitectura identica a `nba_margin_model.py`
(Normal(mu,sigma) sobre point_margin), esto NO reemplaza esa metodologia,
solo le agrega una variable mas confirmada.

**Historial real de esta variable -- documentado porque cambio de forma
durante el desarrollo, no se asumio de entrada**:

1. Primer intento: `rest_diff` CONTINUO (home_rest - away_rest, dias exactos),
   mismo camino que funciono en NFL. Resultado real: coeficiente 0.0024,
   t=0.965, p=0.335 -- **NO significativo**. rest_diff continuo no aporta
   señal real en NBA.
2. Segundo intento (este archivo): en vez de dias continuos, un indicador
   BINARIO de back-to-back (0 dias de descanso = jugo ayer tambien).
   Hipotesis: el efecto de fatiga en NBA no es lineal en dias de descanso
   (1 vs 2 vs 3 dias no importa mucho) sino un efecto de UMBRAL especifico
   del back-to-back (0 dias vs 1+ dias). Resultado real, confirmado con
   OLS sobre los 35,546 partidos: `b2b_diff` (home_b2b - away_b2b) da
   coef=-1.968, t=-15.84, p<0.0001 -- **altamente significativo**.
   Chequeo de sanidad con promedios crudos (sin controlar por Elo):
     - Local en b2b, visitante no: margen promedio +0.57
     - Visitante en b2b, local no:  margen promedio +4.33
     - Ninguno en b2b:               margen promedio +2.46
   Consistente con la intuicion: un equipo en back-to-back rinde peor
   relativo a su nivel esperado, tanto de local como de visitante.

Esto es consistente con la literatura de analitica de NBA (el back-to-back
especificamente, no el descanso continuo, es la variable de fatiga mas
citada en la industria) y explica por que NFL y NBA difieren aca: NFL casi
no tiene variacion real en dias de descanso (bye weeks aparte), NBA tiene
una variable binaria de alta frecuencia (15.5% locales / 29.1% visitantes
en b2b) con un efecto de umbral real.

Requiere que games_clean.csv ya tenga home_elo/away_elo
(`add_nba_elo_features.py`) Y home_rest/away_rest
(`add_nba_rest_features.py` -- NO hace falta recalcular esa columna, el
binario se deriva aca mismo de home_rest/away_rest). Partidos sin
home_rest/away_rest calculable (primer partido de una franquicia en el
historico, ~15-24 casos de 35,546) se excluyen del ENTRENAMIENTO
explicitamente (dropna reportado, no silencioso).

Este archivo NO se corre standalone -- lo importa `backtest_nba_v2.py`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = ["home_elo", "away_elo", "point_margin", "home_rest", "away_rest"]


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre 'python -m src.processing.add_nba_elo_features' "
            f"y 'python -m src.processing.add_nba_rest_features' sobre games_clean.csv."
        )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    # Binario de back-to-back, NO dias continuos -- ver docstring del modulo:
    # rest_diff continuo se probo primero y no fue significativo (p=0.335).
    # OJO: "NaN == 0" da False en pandas/numpy (no NaN) -- una comparacion
    # directa convertiria silenciosamente los ~15-24 partidos sin
    # home_rest/away_rest calculable en "no esta en back-to-back" (0.0) en
    # vez de excluirlos como documentado. Se fuerza la propagacion de NaN
    # explicitamente con np.where.
    df["home_b2b"] = np.where(df["home_rest"].isna(), np.nan, (df["home_rest"] == 0).astype(float))
    df["away_b2b"] = np.where(df["away_rest"].isna(), np.nan, (df["away_rest"] == 0).astype(float))
    df["b2b_diff"] = df["home_b2b"] - df["away_b2b"]  # NaN - x = NaN, propaga correctamente
    return df


def fit_margin_model(train_df: pd.DataFrame):
    """OLS: point_margin ~ elo_diff + b2b_diff. Devuelve el modelo ajustado
    y sigma (desvio estandar de los residuos del propio training set)."""
    _check_columns(train_df)
    train_df = _add_features(train_df)

    before = len(train_df)
    train_df = train_df.dropna(subset=["elo_diff", "b2b_diff", "point_margin"])
    dropped = before - len(train_df)
    if dropped:
        print(f"  [INFO] {dropped} partidos de training sin b2b_diff calculable (primer partido "
              f"de una franquicia) -- excluidos del entrenamiento, no rellenados.")

    X = sm.add_constant(train_df[["elo_diff", "b2b_diff"]])  # orden fijo: const, elo_diff, b2b_diff
    y = train_df["point_margin"]
    model = sm.OLS(y, X).fit()

    sigma = float(model.resid.std(ddof=1))
    return model, sigma


def predict_dataframe(model, sigma: float, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo a un DataFrame completo, vectorizado. Filas sin
    b2b_diff calculable quedan con prediccion NaN (explicito), no se
    inventa un valor."""
    _check_columns(df)
    feats = _add_features(df)
    X = sm.add_constant(feats[["elo_diff", "b2b_diff"]], has_constant="add")

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
