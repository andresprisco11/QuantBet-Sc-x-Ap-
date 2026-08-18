"""
Fase 2 v6: segunda variable de feature engineering, encima de v4 (NO de v5).

v4 agrego forma reciente por tiros al arco (team_recent_form) y fue el
primer resultado positivo real. v5 intento separar esa misma señal en
ataque/defensa y empeoro -- se documento como negativo y se decidio
explicitamente NO seguir encadenando sobre v5 (ver roadmap). Este script
retoma la base de v4 y agrega una señal DISTINTA: el diferencial neto de
corners recientes (team_recent_corner_diff), calculado en
add_team_form_features.py con el mismo criterio sin fuga que las demas
(shift(1) + rolling(5) + prior neutro 0.0).

POR QUE corners y no otra cosa: los corners capturan presion ofensiva
sostenida de forma parcialmente independiente de los tiros al arco (un
equipo puede generar muchos corners centrando desde banda sin rematar a
puerta, o al reves) -- es la siguiente señal mas barata de extraer del
mismo dataset de football-data.co.uk, antes de pasar a variables mas caras
(disciplina, tuning de ventana).

Se construye ENCIMA de v4, NO de v3 (Dixon-Coles, resultado negativo
documentado) ni de v5 (ataque/defensa separado, resultado negativo
documentado) -- misma logica que ya se aplico al construir v4 sobre v2 y
no sobre v3.

REQUIERE que el DataFrame de entrada ya tenga las columnas
home_recent_st_diff / away_recent_st_diff / home_recent_corner_diff /
away_recent_corner_diff -- correr add_team_form_features.py (version
actualizada, agrega tiros + corners) sobre matches_clean.csv si todavia
no existen.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.poisson_model import MAX_GOALS, scoreline_matrix, outcome_probs_from_matrix
from src.models.poisson_model_v2 import (
    identify_promoted_teams_by_season, _build_promoted_synthetic_rows,
    PROMOTED_LABEL, DEFAULT_HALF_LIFE_DAYS,
)

REQUIRED_FORM_COLS = [
    "home_recent_st_diff", "away_recent_st_diff",
    "home_recent_corner_diff", "away_recent_corner_diff",
]


def _check_form_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_FORM_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre primero "
            f"'python -m src.processing.add_team_form_features' (version actualizada, con corners) "
            f"sobre matches_clean.csv."
        )


def build_long_format_v6(df: pd.DataFrame, reference_date=None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """
    Igual que build_long_format_v4, pero cada fila lleva ademas
    'team_recent_corner_diff' -- el diferencial neto de corners recientes
    del equipo que protagoniza esa fila. Filas sinteticas PROMOTED_TEAM y
    el prior bootstrap reciben 0.0 en ambas variables de forma, mismo
    prior neutro que el resto del proyecto.
    """
    _check_form_columns(df)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    home_cols = ["HomeTeam", "AwayTeam", "FTHG", "Date", "season", "home_recent_st_diff", "home_recent_corner_diff"]
    home = df[home_cols].rename(columns={
        "HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals",
        "home_recent_st_diff": "team_recent_form", "home_recent_corner_diff": "team_recent_corner_diff",
    })
    home["is_home"] = 1

    away_cols = ["AwayTeam", "HomeTeam", "FTAG", "Date", "season", "away_recent_st_diff", "away_recent_corner_diff"]
    away = df[away_cols].rename(columns={
        "AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals",
        "away_recent_st_diff": "team_recent_form", "away_recent_corner_diff": "team_recent_corner_diff",
    })
    away["is_home"] = 0
    long_df = pd.concat([home, away], ignore_index=True)

    promoted_by_season = identify_promoted_teams_by_season(df)
    promoted_rows = _build_promoted_synthetic_rows(df, promoted_by_season)
    if not promoted_rows.empty:
        promoted_rows["team_recent_form"] = 0.0
        promoted_rows["team_recent_corner_diff"] = 0.0
        long_df = pd.concat([long_df, promoted_rows], ignore_index=True)

    has_team_level = (long_df["team"] == PROMOTED_LABEL).any()
    has_opponent_level = (long_df["opponent"] == PROMOTED_LABEL).any()
    if not (has_team_level and has_opponent_level):
        placeholder_opponent = df["HomeTeam"].iloc[0]
        avg_home_goals = df["FTHG"].mean()
        avg_away_goals = df["FTAG"].mean()
        ref_date = df["Date"].max()
        last_season = df.sort_values("Date")["season"].iloc[-1]
        bootstrap = pd.DataFrame({
            "team": [PROMOTED_LABEL, placeholder_opponent],
            "opponent": [placeholder_opponent, PROMOTED_LABEL],
            "goals": [avg_away_goals, avg_home_goals],
            "is_home": [0, 1],
            "Date": [ref_date, ref_date],
            "season": [last_season, last_season],
            "team_recent_form": [0.0, 0.0],
            "team_recent_corner_diff": [0.0, 0.0],
        })
        long_df = pd.concat([long_df, bootstrap], ignore_index=True)

    if reference_date is None:
        reference_date = long_df["Date"].max()
    else:
        reference_date = pd.to_datetime(reference_date)

    days_elapsed = (reference_date - long_df["Date"]).dt.days.clip(lower=0)
    decay_rate = np.log(2) / half_life_days
    long_df["weight"] = np.exp(-decay_rate * days_elapsed)

    return long_df


def fit_poisson_model_v6(long_df: pd.DataFrame):
    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent) + team_recent_form + team_recent_corner_diff",
        data=long_df,
        family=sm.families.Poisson(),
        freq_weights=long_df["weight"],
    ).fit()
    return model


def predict_expected_goals_v6(model, home_team: str, away_team: str, known_teams: set,
                               home_recent_form: float, away_recent_form: float,
                               home_recent_corner_diff: float, away_recent_corner_diff: float) -> tuple:
    home_label = home_team if home_team in known_teams else PROMOTED_LABEL
    away_label = away_team if away_team in known_teams else PROMOTED_LABEL

    home_row = pd.DataFrame({
        "team": [home_label], "opponent": [away_label], "is_home": [1],
        "team_recent_form": [home_recent_form], "team_recent_corner_diff": [home_recent_corner_diff],
    })
    away_row = pd.DataFrame({
        "team": [away_label], "opponent": [home_label], "is_home": [0],
        "team_recent_form": [away_recent_form], "team_recent_corner_diff": [away_recent_corner_diff],
    })
    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    return lambda_home, lambda_away


def predict_match_v6(model, home_team: str, away_team: str, known_teams: set,
                      home_recent_form: float, away_recent_form: float,
                      home_recent_corner_diff: float, away_recent_corner_diff: float,
                      max_goals: int = MAX_GOALS) -> dict:
    lambda_home, lambda_away = predict_expected_goals_v6(
        model, home_team, away_team, known_teams,
        home_recent_form, away_recent_form, home_recent_corner_diff, away_recent_corner_diff,
    )
    matrix = scoreline_matrix(lambda_home, lambda_away, max_goals)
    prob_home, prob_draw, prob_away = outcome_probs_from_matrix(matrix)
    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "model_prob_home": prob_home,
        "model_prob_draw": prob_draw,
        "model_prob_away": prob_away,
    }


def predict_dataframe_v6(model, df: pd.DataFrame, known_teams: set, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Usa home_recent_st_diff/away_recent_st_diff/home_recent_corner_diff/away_recent_corner_diff
    ya presentes en df (precomputadas globalmente, sin fuga de informacion)."""
    _check_form_columns(df)
    records = []
    for _, row in df.iterrows():
        result = predict_match_v6(
            model, row["HomeTeam"], row["AwayTeam"], known_teams,
            row["home_recent_st_diff"], row["away_recent_st_diff"],
            row["home_recent_corner_diff"], row["away_recent_corner_diff"], max_goals,
        )
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)