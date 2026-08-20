"""
Fase 10, v2 -- extiende nfl_margin_model.py agregando el descanso entre
partidos (home_rest/away_rest, ya presentes en matches_clean.csv desde
clean_nfl_data.py, sin usar hasta ahora) como segundo regresor junto a
elo_diff.

Motivo: el modelo v1 corre con una sola variable (elo_diff) -- el propio
tune_staking_rules_nfl.py lo señalo explicitamente en su nota final ("el
modelo de un solo feature (elo_diff) necesita mas señal antes de que NFL
sea viable economicamente"). El descanso entre partidos es una variable
estandar en el analisis de NFL (equipos con menos dias de descanso rinden
peor en promedio -- bye weeks, Thursday Night Football con descanso corto,
etc., efecto bien documentado en la literatura publica de la NFL, no
inventado para este proyecto) -- candidato natural de bajo riesgo antes de
meterse con la integracion de xG (que todavia depende de que termine la
ingesta de TheStatsAPI).

Misma metodologia que v1 en todo lo demas (margen como Normal(mu, sigma),
mu de una regresion OLS, sigma de los residuos de ENTRENAMIENTO -- walk
forward, sin fuga) -- el UNICO cambio real es agregar
`rest_diff = home_rest - away_rest` como segunda variable de la regresion.

IMPORTANTE -- columnas home_rest/away_rest NO confirmadas contra el CSV
real en esta sesion (a diferencia de home_elo/away_elo, que si se
confirmaron con la corrida real de add_nfl_elo_features.py). Si
_check_columns() falla listando estas dos, el primer paso es abrir
matches_clean.csv y confirmar el nombre EXACTO de las columnas de descanso
que genero clean_nfl_data.py -- no asumir que el problema esta en otro
lado.

Requiere que matches_clean.csv tenga home_elo/away_elo (de
add_nfl_elo_features.py, ya confirmado: correlacion 0.930 Elo-vs-record
real) Y home_rest/away_rest.

Este archivo NO se corre standalone -- lo importa `backtest_nfl_v2.py`
(script hermano, mismo patron que backtest_nfl.py/nfl_margin_model.py v1).
"""
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = ["home_elo", "away_elo", "point_margin", "home_rest", "away_rest"]


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- si son home_rest/away_rest, confirmar el nombre REAL "
            f"de esas columnas en matches_clean.csv (revisar clean_nfl_data.py) antes de asumir "
            f"que el problema esta en otro lado. Si son home_elo/away_elo, corre "
            f"'python -m src.processing.add_nfl_elo_features' primero."
        )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    return df


def fit_margin_model(train_df: pd.DataFrame):
    """OLS: point_margin ~ elo_diff + rest_diff. Mismo patron que v1
    (fit_margin_model de nfl_margin_model.py), con un segundo regresor."""
    _check_columns(train_df)
    train_df = _add_features(train_df)

    X = sm.add_constant(train_df[["elo_diff", "rest_diff"]])  # columnas: const, elo_diff, rest_diff (orden fijo)
    y = train_df["point_margin"]
    model = sm.OLS(y, X).fit()

    sigma = float(model.resid.std(ddof=1))
    return model, sigma


def predict_margin_distribution(model, sigma: float, home_elo: float, away_elo: float,
                                 home_rest: float, away_rest: float) -> dict:
    """mu/sigma de la Normal para UN partido -- misma separacion que v1
    para poder reusar mu/sigma con cualquier spread_line sin recalcular."""
    elo_diff = home_elo - away_elo
    rest_diff = home_rest - away_rest
    X = pd.DataFrame({"const": [1.0], "elo_diff": [elo_diff], "rest_diff": [rest_diff]})
    mu = float(model.predict(X).iloc[0])
    return {"mu": mu, "sigma": sigma}


def prob_home_win(mu: float, sigma: float) -> float:
    """P(margen > 0) -- identica a v1, no depende de cuantos regresores
    tenga mu."""
    return float(1.0 - norm.cdf(0.0, loc=mu, scale=sigma))


def prob_home_covers(mu: float, sigma: float, spread_line: float) -> float:
    """P(margen > spread_line) -- misma convencion de signo que v1,
    confirmada con datos reales en nfl_data_loader.py."""
    return float(1.0 - norm.cdf(spread_line, loc=mu, scale=sigma))


def predict_dataframe(model, sigma: float, df: pd.DataFrame) -> pd.DataFrame:
    """Version vectorizada -- mismo shape de salida que v1
    (predict_dataframe de nfl_margin_model.py) para que backtest_nfl_v2.py
    pueda reusar la misma logica de evaluacion/blending que backtest_nfl.py."""
    _check_columns(df)
    feat_df = _add_features(df)
    X = sm.add_constant(feat_df[["elo_diff", "rest_diff"]])

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