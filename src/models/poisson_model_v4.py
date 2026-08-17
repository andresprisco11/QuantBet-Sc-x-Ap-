"""
Fase 2 v4: primera variable de FEATURE ENGINEERING sobre el modelo base
(en vez de otro ajuste estructural del Poisson -- ver roadmap, "Implicacion
estrategica" de v3 para el porque de este cambio de categoria).

Agrega una sola covariable continua nueva: la forma reciente de cada equipo
medida por tiros al arco (home_recent_st_diff / away_recent_st_diff,
calculadas en add_team_form_features.py, ver ese archivo para el detalle de
como se computan sin fuga de informacion). El resto del modelo es identico
a v2: recencia (freq_weights) + rating PROMOTED_TEAM para ascendidos.

Se construye ENCIMA de v2, NO de v3: el ajuste de correlacion Dixon-Coles
de v3 quedo documentado como resultado negativo (diferencia real de
~0.00003 en Brier, ver roadmap) -- encadenarlo aca solo agregaria costo
computacional (una optimizacion de MLE extra por fold) sin ningun beneficio
medido. Si en el futuro se junta suficiente señal nueva como para que
valga la pena revisitarlo, poisson_model_v3.py sigue disponible tal cual.

REQUIERE que el DataFrame de entrada ya tenga las columnas
home_recent_st_diff / away_recent_st_diff -- correr primero
add_team_form_features.py sobre matches_clean.csv si todavia no existen.
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

REQUIRED_FORM_COLS = ["home_recent_st_diff", "away_recent_st_diff"]


def _check_form_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_FORM_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre primero "
            f"'python -m src.processing.add_team_form_features' sobre matches_clean.csv."
        )


def build_long_format_v4(df: pd.DataFrame, reference_date=None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """
    Igual que build_long_format_v2, pero cada fila lleva ademas
    'team_recent_form' -- la forma reciente (tiros al arco) del equipo que
    protagoniza esa fila, ya sea de local o visitante. Las filas sinteticas
    PROMOTED_TEAM y el prior bootstrap (equipo sin ningun historial en la
    ventana) reciben team_recent_form=0.0: no tenemos una forma reciente
    confiable para un equipo del que no sabemos nada, es el mismo prior
    neutro que ya usamos para su rating de goles.
    """
    _check_form_columns(df)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    home = df[["HomeTeam", "AwayTeam", "FTHG", "Date", "season", "home_recent_st_diff"]].rename(
        columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals", "home_recent_st_diff": "team_recent_form"}
    )
    home["is_home"] = 1
    away = df[["AwayTeam", "HomeTeam", "FTAG", "Date", "season", "away_recent_st_diff"]].rename(
        columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals", "away_recent_st_diff": "team_recent_form"}
    )
    away["is_home"] = 0
    long_df = pd.concat([home, away], ignore_index=True)

    promoted_by_season = identify_promoted_teams_by_season(df)
    promoted_rows = _build_promoted_synthetic_rows(df, promoted_by_season)
    if not promoted_rows.empty:
        promoted_rows["team_recent_form"] = 0.0
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


def fit_poisson_model_v4(long_df: pd.DataFrame):
    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent) + team_recent_form",
        data=long_df,
        family=sm.families.Poisson(),
        freq_weights=long_df["weight"],
    ).fit()
    return model


def predict_expected_goals_v4(model, home_team: str, away_team: str, known_teams: set,
                               home_recent_form: float, away_recent_form: float) -> tuple:
    home_label = home_team if home_team in known_teams else PROMOTED_LABEL
    away_label = away_team if away_team in known_teams else PROMOTED_LABEL

    home_row = pd.DataFrame({
        "team": [home_label], "opponent": [away_label], "is_home": [1], "team_recent_form": [home_recent_form],
    })
    away_row = pd.DataFrame({
        "team": [away_label], "opponent": [home_label], "is_home": [0], "team_recent_form": [away_recent_form],
    })
    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    return lambda_home, lambda_away


def predict_match_v4(model, home_team: str, away_team: str, known_teams: set,
                      home_recent_form: float, away_recent_form: float, max_goals: int = MAX_GOALS) -> dict:
    lambda_home, lambda_away = predict_expected_goals_v4(
        model, home_team, away_team, known_teams, home_recent_form, away_recent_form
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


def predict_dataframe_v4(model, df: pd.DataFrame, known_teams: set, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Usa las columnas home_recent_st_diff / away_recent_st_diff ya presentes en df
    (precomputadas globalmente, sin fuga de informacion) para cada partido a predecir."""
    _check_form_columns(df)
    records = []
    for _, row in df.iterrows():
        result = predict_match_v4(
            model, row["HomeTeam"], row["AwayTeam"], known_teams,
            row["home_recent_st_diff"], row["away_recent_st_diff"], max_goals,
        )
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)