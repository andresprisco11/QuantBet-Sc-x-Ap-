"""
Modelo de goles esperados via regresion de Poisson (enfoque Maher/Dixon-Coles):
cada equipo recibe un rating de ataque y uno de defensa, estimados sobre el
historico disponible. A partir de esos ratings se calculan las probabilidades
de resultado (1X2) y la matriz completa de marcador exacto para cada partido.

Simplificaciones conscientes de esta v1 (a revisar segun resultados de Fase 3):
- Rating de ataque/defensa fijo por equipo en las 5 temporadas (no distingue
  que el plantel de un equipo cambia con el tiempo).
- Ventaja de localia como coeficiente unico global (no varia por temporada,
  aunque en el EDA vimos que si varia en la realidad).
- Poisson simple (asume goles locales y visitantes independientes) -- sin el
  ajuste de correlacion Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

MAX_GOALS = 10  # rango de goles a considerar en la matriz de marcador exacto


def build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el dataset de partidos (una fila por partido) al formato "largo"
    que requiere la regresion de Poisson: dos filas por partido, una desde la
    perspectiva del equipo local y otra desde la del visitante.
    """
    home = df[["HomeTeam", "AwayTeam", "FTHG"]].rename(
        columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals"}
    )
    home["is_home"] = 1

    away = df[["AwayTeam", "HomeTeam", "FTAG"]].rename(
        columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals"}
    )
    away["is_home"] = 0

    return pd.concat([home, away], ignore_index=True)


def fit_poisson_model(long_df: pd.DataFrame):
    """Ajusta: goles ~ localia + ataque(equipo) + defensa(rival). Retorna el modelo entrenado."""
    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent)",
        data=long_df,
        family=sm.families.Poisson(),
    ).fit()
    return model


def predict_expected_goals(model, home_team: str, away_team: str) -> tuple[float, float]:
    """Calcula goles esperados (lambda) para local y visitante de un partido especifico."""
    home_row = pd.DataFrame({"team": [home_team], "opponent": [away_team], "is_home": [1]})
    away_row = pd.DataFrame({"team": [away_team], "opponent": [home_team], "is_home": [0]})

    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    return lambda_home, lambda_away


def scoreline_matrix(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matriz de probabilidad de cada marcador exacto, asumiendo independencia local/visitante."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_probs, away_probs)


def outcome_probs_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Colapsa la matriz de marcador exacto en probabilidades 1X2 (Local/Empate/Visitante)."""
    prob_home = np.tril(matrix, k=-1).sum()
    prob_draw = np.trace(matrix)
    prob_away = np.triu(matrix, k=1).sum()
    return prob_home, prob_draw, prob_away


def predict_match(model, home_team: str, away_team: str, max_goals: int = MAX_GOALS) -> dict:
    """Pipeline completo para un partido: goles esperados + probabilidades 1X2 + matriz de marcador."""
    lambda_home, lambda_away = predict_expected_goals(model, home_team, away_team)
    matrix = scoreline_matrix(lambda_home, lambda_away, max_goals)
    prob_home, prob_draw, prob_away = outcome_probs_from_matrix(matrix)

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "model_prob_home": prob_home,
        "model_prob_draw": prob_draw,
        "model_prob_away": prob_away,
        "scoreline_matrix": matrix,
    }


def predict_dataframe(model, df: pd.DataFrame, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Aplica predict_match a cada partido de un DataFrame y arma una tabla de resultados."""
    records = []
    for _, row in df.iterrows():
        result = predict_match(model, row["HomeTeam"], row["AwayTeam"], max_goals)
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)
