"""
Fase 10 -- primer modelo real de NFL: predice el MARGEN DE PUNTOS
(home_score - away_score) como una distribucion Normal(mu, sigma), NO una
clasificacion de moneyline como el resto del proyecto (futbol/tenis) --
exactamente la diferencia arquitectonica ya anticipada en
nfl_data_loader.py: el mercado de NFL gira en torno al SPREAD, asi que el
objeto de prediccion correcto es una distribucion continua de diferencial,
de la que se derivan DESPUES tanto la probabilidad de moneyline
(P(margen>0)) como la probabilidad de cubrir el spread
(P(margen>spread_line)) -- UNA sola distribucion alimenta las dos apuestas,
en vez de necesitar dos modelos separados.

Metodologia (estandar de la industria de power ratings -- Sagarin, Massey,
FiveThirtyEight -- no inventada para este proyecto, la parte nueva es
aplicarla a los datos propios de este proyecto vs. Elo de
add_nfl_elo_features.py):

1. **mu (margen esperado)** = regresion OLS de `point_margin` sobre la
   diferencia de Elo (`home_elo - away_elo`). El Elo YA tiene la ventaja de
   local incorporada en su propia expectativa (HOME_ADVANTAGE en
   add_nfl_elo_features.py), pero eso es una probabilidad de victoria, no
   una cantidad de puntos -- por eso el margen esperado se re-estima aca
   con su propia regresion en vez de derivarse directo del Elo.
2. **sigma (desvio del margen)** = desvio estandar de los RESIDUOS del
   training set (margen real - margen predicho) -- constante por ahora, no
   depende de elo_diff ni de la semana (simplificacion estandar de este
   tipo de modelo, walk-forward: sigma se estima SOLO con datos de
   entrenamiento, igual que mu, sin fuga).
3. De la Normal(mu, sigma) se derivan:
   - `prob_home_win` = P(margen > 0) = 1 - NormalCDF(0; mu, sigma)
   - `prob_home_covers(spread_line)` = P(margen > spread_line) --
     definicion ESTANDAR de "cubrir el spread": si el local es favorito
     (spread_line positivo, convencion de signo ya confirmada en
     nfl_data_loader.py con datos reales), tiene que ganar por MAS que
     spread_line para cubrir.

Walk-forward por temporada real (entrena con temporadas anteriores, evalua
la siguiente, ventana expansiva) -- se corre en `backtest_nfl.py` (script
hermano de este), mismo patron que `backtest_v4.py`/`backtest_v7.py` de
futbol.

Requiere que matches_clean.csv ya tenga `home_elo`/`away_elo`
(`add_nfl_elo_features.py`, ya confirmado por el usuario: correlacion 0.930
entre Elo final y record real 2025 -- ver roadmap).

Este archivo NO se corre standalone -- lo importa `backtest_nfl.py`.
"""
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = ["home_elo", "away_elo", "point_margin"]


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre "
            f"'python -m src.processing.add_nfl_elo_features' sobre matches_clean.csv."
        )


def fit_margin_model(train_df: pd.DataFrame):
    """OLS: point_margin ~ elo_diff. Devuelve el modelo ajustado y sigma
    (desvio estandar de los residuos del propio training set)."""
    _check_columns(train_df)
    train_df = train_df.copy()
    train_df["elo_diff"] = train_df["home_elo"] - train_df["away_elo"]

    X = sm.add_constant(train_df[["elo_diff"]])  # columnas: const, elo_diff (orden fijo)
    y = train_df["point_margin"]
    model = sm.OLS(y, X).fit()

    sigma = float(model.resid.std(ddof=1))
    return model, sigma


def predict_margin_distribution(model, sigma: float, home_elo: float, away_elo: float) -> dict:
    """mu/sigma de la Normal para UN partido -- separado de las
    probabilidades derivadas para poder reusar mu/sigma con cualquier
    spread_line sin recalcular el modelo."""
    elo_diff = home_elo - away_elo
    X = pd.DataFrame({"const": [1.0], "elo_diff": [elo_diff]})
    mu = float(model.predict(X).iloc[0])
    return {"mu": mu, "sigma": sigma}


def prob_home_win(mu: float, sigma: float) -> float:
    """P(margen > 0) -- probabilidad de moneyline del local."""
    return float(1.0 - norm.cdf(0.0, loc=mu, scale=sigma))


def prob_home_covers(mu: float, sigma: float, spread_line: float) -> float:
    """P(margen > spread_line) -- probabilidad de que el LOCAL cubra el
    spread. Convencion de signo ya confirmada con datos reales en
    nfl_data_loader.py: spread_line positivo = local favorito (tiene que
    ganar por MAS que esa cantidad para cubrir); negativo = visitante
    favorito."""
    return float(1.0 - norm.cdf(spread_line, loc=mu, scale=sigma))


def predict_dataframe(model, sigma: float, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo a un DataFrame completo, vectorizado (regresion
    lineal simple, no hace falta el loop fila-por-fila que si necesitaba
    poisson_model por la matriz de resultados)."""
    _check_columns(df)
    elo_diff = (df["home_elo"] - df["away_elo"]).rename("elo_diff")
    X = sm.add_constant(elo_diff.to_frame())

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