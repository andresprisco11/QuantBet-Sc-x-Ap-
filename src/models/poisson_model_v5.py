"""
Fase 2 v5: separa la variable de forma reciente de v4 (un solo diferencial
neto de tiros al arco) en DOS covariables independientes:

  - team_recent_attack: cuantos tiros al arco genero el equipo el mismo
    (ultimos 5 partidos, sin fuga de informacion) -- su propia forma
    OFENSIVA reciente.
  - opponent_recent_defense: cuantos tiros al arco concedio el RIVAL
    (ultimos 5 partidos) -- que tan solida o fragil viene su defensa.

POR QUE: el diferencial neto de v4 no distingue si un equipo rinde mal
porque su ataque esta flojo o porque su defensa regala tiros -- son cosas
distintas y un rival deberia reaccionar distinto a cada una. Separarlas le
da al GLM la libertad de aprender un coeficiente propio para cada efecto,
en vez de forzar un solo numero neto que promedia ambos.

El resto del modelo es identico a v2/v4: recencia (freq_weights) + rating
PROMOTED_TEAM para ascendidos. Sigue sin encadenar Dixon-Coles (v3, ver
poisson_model_v3.py) -- resultado negativo documentado, no vale la pena
el costo computacional extra.

REQUIERE que el DataFrame de entrada ya tenga las columnas
home_recent_attack / home_recent_defense / away_recent_attack /
away_recent_defense -- correr add_team_form_features.py (version
actualizada, agrega estas 4 columnas ademas de las 2 de v4) sobre
matches_clean.csv si todavia no existen.
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

REQUIRED_FORM_COLS = ["home_recent_attack", "home_recent_defense", "away_recent_attack", "away_recent_defense"]


def _check_form_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_FORM_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre primero "
            f"'python -m src.processing.add_team_form_features' (version actualizada) sobre matches_clean.csv."
        )


def build_long_format_v5(df: pd.DataFrame, reference_date=None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """
    Igual que build_long_format_v2, pero cada fila lleva ademas
    'team_recent_attack' (forma ofensiva propia) y 'opponent_recent_defense'
    (forma defensiva del rival en ESE partido). Las filas sinteticas
    PROMOTED_TEAM y el prior bootstrap reciben 0.0 en ambas -- mismo prior
    neutro que el resto del proyecto usa para equipos sin historial.
    """
    _check_form_columns(df)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    home = df[["HomeTeam", "AwayTeam", "FTHG", "Date", "season", "home_recent_attack", "away_recent_defense"]].rename(
        columns={
            "HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals",
            "home_recent_attack": "team_recent_attack", "away_recent_defense": "opponent_recent_defense",
        }
    )
    home["is_home"] = 1
    away = df[["AwayTeam", "HomeTeam", "FTAG", "Date", "season", "away_recent_attack", "home_recent_defense"]].rename(
        columns={
            "AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals",
            "away_recent_attack": "team_recent_attack", "home_recent_defense": "opponent_recent_defense",
        }
    )
    away["is_home"] = 0
    long_df = pd.concat([home, away], ignore_index=True)

    promoted_by_season = identify_promoted_teams_by_season(df)
    promoted_rows = _build_promoted_synthetic_rows(df, promoted_by_season)
    if not promoted_rows.empty:
        promoted_rows["team_recent_attack"] = 0.0
        promoted_rows["opponent_recent_defense"] = 0.0
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
            "team_recent_attack": [0.0, 0.0],
            "opponent_recent_defense": [0.0, 0.0],
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


def fit_poisson_model_v5(long_df: pd.DataFrame):
    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent) + team_recent_attack + opponent_recent_defense",
        data=long_df,
        family=sm.families.Poisson(),
        freq_weights=long_df["weight"],
    ).fit()
    return model


def predict_expected_goals_v5(model, home_team: str, away_team: str, known_teams: set,
                               home_attack: float, home_defense: float,
                               away_attack: float, away_defense: float) -> tuple:
    home_label = home_team if home_team in known_teams else PROMOTED_LABEL
    away_label = away_team if away_team in known_teams else PROMOTED_LABEL

    home_row = pd.DataFrame({
        "team": [home_label], "opponent": [away_label], "is_home": [1],
        "team_recent_attack": [home_attack], "opponent_recent_defense": [away_defense],
    })
    away_row = pd.DataFrame({
        "team": [away_label], "opponent": [home_label], "is_home": [0],
        "team_recent_attack": [away_attack], "opponent_recent_defense": [home_defense],
    })
    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    return lambda_home, lambda_away


def predict_match_v5(model, home_team: str, away_team: str, known_teams: set,
                      home_attack: float, home_defense: float, away_attack: float, away_defense: float,
                      max_goals: int = MAX_GOALS) -> dict:
    lambda_home, lambda_away = predict_expected_goals_v5(
        model, home_team, away_team, known_teams, home_attack, home_defense, away_attack, away_defense
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


def predict_dataframe_v5(model, df: pd.DataFrame, known_teams: set, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """Usa home_recent_attack/home_recent_defense/away_recent_attack/away_recent_defense
    ya presentes en df (precomputadas globalmente, sin fuga de informacion)."""
    _check_form_columns(df)
    records = []
    for _, row in df.iterrows():
        result = predict_match_v5(
            model, row["HomeTeam"], row["AwayTeam"], known_teams,
            row["home_recent_attack"], row["home_recent_defense"],
            row["away_recent_attack"], row["away_recent_defense"], max_goals,
        )
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)