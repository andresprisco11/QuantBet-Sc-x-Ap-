"""
Fase 10 (NBA) -- primer modelo real de NBA: predice el MARGEN DE PUNTOS
(home_pts - away_pts) como una distribucion Normal(mu, sigma), mismo diseño
arquitectonico que `nfl_margin_model.py` (NO un clasificador de moneyline
reciclado de futbol/tenis) -- NBA tambien cotiza spread de puntos como
mercado principal (junto con moneyline y totales), asi que el mismo
argumento de NFL aplica: una distribucion continua de margen alimenta
moneyline Y spread con un solo modelo, en vez de necesitar dos.

Metodologia -- identica a NFL, mismos 3 pasos, misma industria estandar
(Sagarin/Massey/FiveThirtyEight), la parte nueva es aplicarla a los datos
propios de NBA (`games_clean.csv` + `add_nba_elo_features.py`) en vez de
NFL:

1. **mu (margen esperado)** = OLS de `point_margin` sobre `elo_diff`
   (home_elo - away_elo). Mismo motivo que NFL: el Elo ya tiene la ventaja
   de local en su expectativa de VICTORIA, no en puntos -- el margen
   esperado se re-estima aca con su propia regresion.
2. **sigma (desvio del margen)** = desvio estandar de los residuos del
   training set, constante, sin fuga (walk-forward).
3. Derivados: `prob_home_win` = P(margen>0), `prob_home_covers(spread_line)`
   = P(margen>spread_line) -- funcion generica ya incluida aunque
   `games_clean.csv` todavia NO tiene spread_line (eso llega cuando se
   pegue `theoddsapi_historical_loader.py`, no corrido todavia) -- se deja
   lista para no tener que rehacer este archivo despues.

Requiere que games_clean.csv ya tenga `home_elo`/`away_elo`
(`add_nba_elo_features.py`, ya confirmado por el usuario: correlacion 0.967
entre Elo final y record real 2025-26 -- ver roadmap).

Este archivo NO se corre standalone -- lo importa `backtest_nba.py`.
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
            f"'python -m src.processing.add_nba_elo_features' sobre games_clean.csv."
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
    """mu/sigma de la Normal para UN partido."""
    elo_diff = home_elo - away_elo
    X = pd.DataFrame({"const": [1.0], "elo_diff": [elo_diff]})
    mu = float(model.predict(X).iloc[0])
    return {"mu": mu, "sigma": sigma}


def prob_home_win(mu: float, sigma: float) -> float:
    """P(margen > 0) -- probabilidad de moneyline del local."""
    return float(1.0 - norm.cdf(0.0, loc=mu, scale=sigma))


def prob_home_covers(mu: float, sigma: float, spread_line: float) -> float:
    """P(margen > spread_line) -- probabilidad de que el LOCAL cubra el
    spread. Misma convencion de signo que NFL (positivo = local favorito,
    tiene que ganar por MAS que esa cantidad para cubrir) -- pendiente de
    reconfirmar con datos reales de spread de NBA cuando existan, no
    asumido sin mas."""
    return float(1.0 - norm.cdf(spread_line, loc=mu, scale=sigma))


def predict_dataframe(model, sigma: float, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo a un DataFrame completo, vectorizado."""
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
