"""
Fase 2 v3: ajuste de correlacion Dixon & Coles (1997) para marcadores bajos.

MOTIVACION (limitacion documentada desde v1, nunca abordada hasta ahora):
el modelo de Poisson v1/v2 asume que los goles de local y visitante son
INDEPENDIENTES entre si (goles_local ~ Poisson(lambda), goles_visitante ~
Poisson(mu), sin correlacion). En la practica esto no es cierto exactamente
para marcadores bajos: los partidos 0-0 y 1-1 ocurren mas seguido de lo que
predice el Poisson independiente (los equipos "juegan con cuidado" cuando el
resultado esta muy ajustado), y los 1-0/0-1 ocurren un poco menos de lo
esperado. Dixon & Coles (1997) corrigen esto multiplicando la probabilidad
conjunta de los 4 marcadores bajos (0-0, 0-1, 1-0, 1-1) por una funcion
tau(x, y, lambda, mu, rho) con un parametro de dependencia rho estimado por
maxima verosimilitud sobre los datos reales -- el resto de la matriz de
marcadores (2+ goles) queda igual que en el Poisson independiente.

Esta es la pieza de arquitectura que faltaba, no un hiperparametro mas para
tunear (a diferencia del half-life, que ya confirmamos que no mueve la
aguja -- ver tune_half_life.py).

Se construye ENCIMA de v2 (recencia + PROMOTED_TEAM), no en reemplazo: usa
exactamente el mismo build_long_format_v2 / fit_poisson_model_v2 /
predict_expected_goals_v2 para lambda/mu, y solo cambia como se arma la
matriz de marcadores antes de integrar a probabilidades 1X2. Reutiliza
MAX_GOALS y outcome_probs_from_matrix de poisson_model.py sin modificarlos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as scipy_poisson

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.poisson_model import MAX_GOALS, scoreline_matrix, outcome_probs_from_matrix
from src.models.poisson_model_v2 import (
    build_long_format_v2, fit_poisson_model_v2, predict_expected_goals_v2,
    DEFAULT_HALF_LIFE_DAYS, PROMOTED_LABEL,
)

# Cota para rho durante la optimizacion. Con lambda/mu tipicos de futbol
# (0.5-3 goles esperados), valores de |rho| > ~0.3 pueden volver tau
# negativo para algunas combinaciones -- se acota el rango de busqueda a
# algo mas angosto que eso para mantenernos en la zona donde tau siempre
# da una probabilidad valida, sin necesidad de logica de casos especiales.
RHO_BOUND = 0.3


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Factor de correccion de Dixon & Coles para los 4 marcadores bajos.
    Para cualquier otro marcador (x>1 o y>1), tau = 1 -- no hay correccion,
    el Poisson independiente queda igual que antes.
    """
    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lam * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _expected_goals_array(model, df: pd.DataFrame, known_teams: set) -> tuple:
    """Precalcula lambda/mu para cada fila UNA sola vez -- se reutiliza en
    todas las evaluaciones de rho durante la optimizacion, en vez de volver
    a llamar model.predict() por cada candidato de rho (seria carisimo)."""
    lambdas, mus = [], []
    for _, row in df.iterrows():
        lam, mu = predict_expected_goals_v2(model, row["HomeTeam"], row["AwayTeam"], known_teams)
        lambdas.append(lam)
        mus.append(mu)
    return np.array(lambdas), np.array(mus)


def _negative_log_likelihood(rho: float, home_goals: np.ndarray, away_goals: np.ndarray,
                              lambdas: np.ndarray, mus: np.ndarray) -> float:
    """Log-verosimilitud negativa del modelo Poisson + tau de Dixon-Coles,
    sobre los marcadores REALES observados. minimize_scalar busca el rho
    que la minimiza (= maximiza la verosimilitud)."""
    total_log_lik = 0.0
    for x, y, lam, mu in zip(home_goals, away_goals, lambdas, mus):
        tau = dixon_coles_tau(int(x), int(y), lam, mu, rho)
        tau = max(tau, 1e-10)  # evita log(negativo) si rho se acerca al borde del rango valido
        p = tau * scipy_poisson.pmf(x, lam) * scipy_poisson.pmf(y, mu)
        p = max(p, 1e-10)
        total_log_lik += np.log(p)
    return -total_log_lik


def estimate_rho(model, train_df: pd.DataFrame, known_teams: set, bound: float = RHO_BOUND) -> float:
    """
    Estima rho por maxima verosimilitud SOLO sobre los datos de entrenamiento
    de este fold -- nunca sobre los datos de test, para no filtrar informacion
    del futuro hacia el pasado (mismo principio de walk-forward que el resto
    del proyecto).
    """
    lambdas, mus = _expected_goals_array(model, train_df, known_teams)
    home_goals = train_df["FTHG"].values
    away_goals = train_df["FTAG"].values

    result = minimize_scalar(
        _negative_log_likelihood,
        args=(home_goals, away_goals, lambdas, mus),
        bounds=(-bound, bound),
        method="bounded",
    )
    return float(result.x)


def scoreline_matrix_v3(lambda_home: float, lambda_away: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Igual que scoreline_matrix (v1/v2), pero aplica tau de Dixon-Coles a las
    4 celdas de marcador bajo antes de devolver la matriz. Se renormaliza al
    final porque multiplicar esas 4 celdas cambia levemente la masa total de
    probabilidad (tau no es exactamente 1 en promedio).
    """
    matrix = scoreline_matrix(lambda_home, lambda_away, max_goals)
    for x in (0, 1):
        for y in (0, 1):
            tau = dixon_coles_tau(x, y, lambda_home, lambda_away, rho)
            matrix[x, y] *= tau
    matrix = matrix / matrix.sum()
    return matrix


def predict_match_v3(model, home_team: str, away_team: str, known_teams: set, rho: float,
                      max_goals: int = MAX_GOALS) -> dict:
    """Pipeline completo v3 para un partido: goles esperados + tau de Dixon-Coles + probabilidades 1X2."""
    lambda_home, lambda_away = predict_expected_goals_v2(model, home_team, away_team, known_teams)
    matrix = scoreline_matrix_v3(lambda_home, lambda_away, rho, max_goals)
    prob_home, prob_draw, prob_away = outcome_probs_from_matrix(matrix)
    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "model_prob_home": prob_home,
        "model_prob_draw": prob_draw,
        "model_prob_away": prob_away,
    }


def predict_dataframe_v3(model, df: pd.DataFrame, known_teams: set, rho: float,
                          max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Igual que predict_dataframe_v2 -- nunca excluye partidos por equipo desconocido."""
    records = []
    for _, row in df.iterrows():
        result = predict_match_v3(model, row["HomeTeam"], row["AwayTeam"], known_teams, rho, max_goals)
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)