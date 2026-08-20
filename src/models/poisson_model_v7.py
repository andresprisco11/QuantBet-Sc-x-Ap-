"""
Fase 2 v7: se construye ENCIMA de v4 (NO de v5 ni v6 -- los dos últimos
intentos de agregar señal nueva sobre v4 fallaron, mismo criterio ya
aplicado de "construir siempre sobre la mejor base confirmada, no sobre el
último intento"). Agrega una sola covariable nueva: el rating Elo del
equipo (`team_elo`, calculado en `add_team_elo_features.py`, walk-forward,
sin fuga) -- ver ese script para la motivación completa (mismo patrón que
funcionó en tenis esta sesión: Elo pondera la FUERZA DEL RIVAL vencido,
algo que ni `C(team)`/`C(opponent)` -- estáticos sobre la ventana de
entrenamiento -- ni `team_recent_form` -- forma reciente sin ajustar por
calidad del rival -- capturan hoy).

REQUIERE que el DataFrame de entrada ya tenga home_recent_st_diff /
away_recent_st_diff (de add_team_form_features.py, ya verificado, v4) Y
home_team_elo / away_team_elo (de add_team_elo_features.py, nuevo) --
correr ambos scripts sobre matches_clean.csv si todavía no existen.
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

REQUIRED_FORM_COLS = ["home_recent_st_diff", "away_recent_st_diff", "home_team_elo", "away_team_elo"]

# Elo inicial (1500) usado como prior neutro para filas sinteticas PROMOTED_TEAM/bootstrap
# -- mismo valor que add_team_elo_features.py, un equipo del que no sabemos nada arranca
# en el rating neutro del sistema, igual que su team_recent_form neutro es 0.0.
NEUTRAL_ELO = 1500.0


def _check_form_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_FORM_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre 'python -m src.processing.add_team_form_features' "
            f"y 'python -m src.processing.add_team_elo_features' sobre matches_clean.csv."
        )


def build_long_format_v7(df: pd.DataFrame, reference_date=None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """Igual que build_long_format_v4, pero cada fila lleva ademas 'team_elo' -- el
    rating Elo del equipo que protagoniza esa fila, ANTES de ese partido (calculado
    walk-forward en add_team_elo_features.py, sin fuga)."""
    _check_form_columns(df)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    home = df[["HomeTeam", "AwayTeam", "FTHG", "Date", "season", "home_recent_st_diff", "home_team_elo"]].rename(
        columns={
            "HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals",
            "home_recent_st_diff": "team_recent_form", "home_team_elo": "team_elo",
        }
    )
    home["is_home"] = 1
    away = df[["AwayTeam", "HomeTeam", "FTAG", "Date", "season", "away_recent_st_diff", "away_team_elo"]].rename(
        columns={
            "AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals",
            "away_recent_st_diff": "team_recent_form", "away_team_elo": "team_elo",
        }
    )
    away["is_home"] = 0
    long_df = pd.concat([home, away], ignore_index=True)

    promoted_by_season = identify_promoted_teams_by_season(df)
    promoted_rows = _build_promoted_synthetic_rows(df, promoted_by_season)
    if not promoted_rows.empty:
        promoted_rows["team_recent_form"] = 0.0
        promoted_rows["team_elo"] = NEUTRAL_ELO
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
            "team_elo": [NEUTRAL_ELO, NEUTRAL_ELO],
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


def fit_poisson_model_v7(long_df: pd.DataFrame):
    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent) + team_recent_form + team_elo",
        data=long_df,
        family=sm.families.Poisson(),
        freq_weights=long_df["weight"],
    ).fit()
    return model


def predict_expected_goals_v7(model, home_team: str, away_team: str, known_teams: set,
                               home_recent_form: float, away_recent_form: float,
                               home_elo: float, away_elo: float) -> tuple:
    home_label = home_team if home_team in known_teams else PROMOTED_LABEL
    away_label = away_team if away_team in known_teams else PROMOTED_LABEL

    home_row = pd.DataFrame({
        "team": [home_label], "opponent": [away_label], "is_home": [1],
        "team_recent_form": [home_recent_form], "team_elo": [home_elo],
    })
    away_row = pd.DataFrame({
        "team": [away_label], "opponent": [home_label], "is_home": [0],
        "team_recent_form": [away_recent_form], "team_elo": [away_elo],
    })
    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    return lambda_home, lambda_away


def predict_match_v7(model, home_team: str, away_team: str, known_teams: set,
                      home_recent_form: float, away_recent_form: float,
                      home_elo: float, away_elo: float, max_goals: int = MAX_GOALS) -> dict:
    lambda_home, lambda_away = predict_expected_goals_v7(
        model, home_team, away_team, known_teams, home_recent_form, away_recent_form, home_elo, away_elo,
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


def predict_dataframe_v7(model, df: pd.DataFrame, known_teams: set, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Usa home_recent_st_diff/away_recent_st_diff/home_team_elo/away_team_elo ya
    presentes en df (precomputadas globalmente, sin fuga de informacion)."""
    _check_form_columns(df)
    records = []
    for _, row in df.iterrows():
        result = predict_match_v7(
            model, row["HomeTeam"], row["AwayTeam"], known_teams,
            row["home_recent_st_diff"], row["away_recent_st_diff"],
            row["home_team_elo"], row["away_team_elo"], max_goals,
        )
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)