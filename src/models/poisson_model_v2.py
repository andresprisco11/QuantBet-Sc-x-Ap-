"""
Fase 2 v2: mejoras sobre el modelo Poisson v1 (poisson_model.py), motivadas
directamente por el hallazgo de Fase 3 (el blend no le gana al mercado con
el modelo v1).

1. PONDERACION POR RECENCIA (time-decay), al estilo Dixon & Coles (1997):
   cada partido de entrenamiento pesa segun su antiguedad via decaimiento
   exponencial (peso = exp(-ln(2)/half_life * dias_transcurridos)), en vez
   de que un partido de hace 5 temporadas pese igual que uno de la semana
   pasada. Se implementa via freq_weights en el GLM de statsmodels.
   Nota tecnica: freq_weights en statsmodels esta pensado originalmente
   para conteos enteros de replicacion de observaciones, pero acepta pesos
   continuos sin problema para maxima verosimilitud ponderada -- es la
   forma estandar de implementar decaimiento temporal estilo Dixon-Coles
   sin escribir el optimizador de GLM a mano.

2. RATING PRIOR PARA EQUIPOS RECIEN ASCENDIDOS, en vez de excluirlos de la
   evaluacion (que es lo que hace v1, perdiendo ~38-108 partidos por
   temporada). Se construye una categoria sintetica "PROMOTED_TEAM"
   entrenada con el desempeno historico REAL de todos los equipos que
   debutaron en la ventana de entrenamiento (se detectan automaticamente:
   aparecen en la temporada N pero no en la N-1). Al predecir, cualquier
   equipo no visto en el entrenamiento se mapea a esta categoria en vez de
   descartar el partido.

Reutiliza scoreline_matrix / outcome_probs_from_matrix / MAX_GOALS de
poisson_model.py sin modificarlos -- esa parte de la matematica no cambia
entre v1 y v2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.poisson_model import MAX_GOALS, scoreline_matrix, outcome_probs_from_matrix

PROMOTED_LABEL = "PROMOTED_TEAM"
DEFAULT_HALF_LIFE_DAYS = 456  # ~1.5 temporadas: un partido de hace 456 dias pesa la mitad que uno de hoy


def identify_promoted_teams_by_season(df: pd.DataFrame) -> dict:
    """
    Para cada temporada del df recibido (excepto la primera), detecta los
    equipos que aparecen por primera vez -- son los recien ascendidos DENTRO
    de esta ventana de datos. La primera temporada del df se excluye a
    proposito: ahi TODOS los equipos son "nuevos" simplemente porque es
    donde arranca el historial disponible, no porque hayan ascendido.
    """
    seasons_ordered = sorted(df["season"].unique())
    seen_teams = set()
    promoted_by_season = {}
    for season in seasons_ordered:
        season_df = df[df["season"] == season]
        season_teams = set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"]))
        if seen_teams:
            promoted_by_season[season] = season_teams - seen_teams
        seen_teams |= season_teams
    return promoted_by_season


def _build_promoted_synthetic_rows(df: pd.DataFrame, promoted_by_season: dict) -> pd.DataFrame:
    """
    Construye filas sinteticas etiquetadas PROMOTED_TEAM a partir de los
    partidos REALES jugados en temporadas de debut. Se agregan ADEMAS de
    las filas normales del equipo (no las reemplazan), y cubren las 4
    combinaciones necesarias para que "PROMOTED_TEAM" tenga datos tanto en
    la columna 'team' como en 'opponent' de la formula del GLM:

      Caso A (promovido de LOCAL vs rival real X):
        S1: team=PROMOTED_TEAM, opponent=X   (perspectiva del promovido)
        S2: team=X, opponent=PROMOTED_TEAM   (perspectiva del rival real)
      Caso B (promovido de VISITANTE vs rival real X):
        S3: team=PROMOTED_TEAM, opponent=X   (perspectiva del promovido)
        S4: team=X, opponent=PROMOTED_TEAM   (perspectiva del rival real)
    """
    frames = []
    for season, teams in promoted_by_season.items():
        if not teams:
            continue
        season_df = df[df["season"] == season]

        home_promoted = season_df[season_df["HomeTeam"].isin(teams)]
        if not home_promoted.empty:
            s1 = home_promoted[["AwayTeam", "FTHG", "Date", "season"]].rename(
                columns={"AwayTeam": "opponent", "FTHG": "goals"}
            )
            s1["team"] = PROMOTED_LABEL
            s1["is_home"] = 1
            frames.append(s1)

            s2 = home_promoted[["AwayTeam", "FTAG", "Date", "season"]].rename(
                columns={"AwayTeam": "team", "FTAG": "goals"}
            )
            s2["opponent"] = PROMOTED_LABEL
            s2["is_home"] = 0
            frames.append(s2)

        away_promoted = season_df[season_df["AwayTeam"].isin(teams)]
        if not away_promoted.empty:
            s3 = away_promoted[["HomeTeam", "FTAG", "Date", "season"]].rename(
                columns={"HomeTeam": "opponent", "FTAG": "goals"}
            )
            s3["team"] = PROMOTED_LABEL
            s3["is_home"] = 0
            frames.append(s3)

            s4 = away_promoted[["HomeTeam", "FTHG", "Date", "season"]].rename(
                columns={"HomeTeam": "team", "FTHG": "goals"}
            )
            s4["opponent"] = PROMOTED_LABEL
            s4["is_home"] = 1
            frames.append(s4)

    if not frames:
        return pd.DataFrame(columns=["team", "opponent", "goals", "is_home", "Date", "season"])
    return pd.concat(frames, ignore_index=True)


def build_long_format_v2(df: pd.DataFrame, reference_date=None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> pd.DataFrame:
    """
    Version 2 del formato largo: agrega (a) filas sinteticas PROMOTED_TEAM
    construidas con historial real de debuts, y (b) una columna 'weight'
    con decaimiento exponencial por antiguedad, lista para usar como
    freq_weights en el ajuste del GLM.

    reference_date: fecha desde la cual se mide "antiguedad". Por defecto,
    la fecha mas reciente del propio set recibido -- asi el partido mas
    reciente de la ventana de entrenamiento siempre pesa 1.0.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    home = df[["HomeTeam", "AwayTeam", "FTHG", "Date", "season"]].rename(
        columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals"}
    )
    home["is_home"] = 1
    away = df[["AwayTeam", "HomeTeam", "FTAG", "Date", "season"]].rename(
        columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals"}
    )
    away["is_home"] = 0
    long_df = pd.concat([home, away], ignore_index=True)

    promoted_by_season = identify_promoted_teams_by_season(df)
    promoted_rows = _build_promoted_synthetic_rows(df, promoted_by_season)
    if not promoted_rows.empty:
        long_df = pd.concat([long_df, promoted_rows], ignore_index=True)

    # --- Red de seguridad: si la ventana de entrenamiento es demasiado
    # corta (tipicamente el primer fold del walk-forward, con una sola
    # temporada) no hay ningun debut historico previo para construir
    # PROMOTED_TEAM -- la categoria queda sin datos, y al predecir un
    # partido con un equipo desconocido, patsy revienta con PatsyError
    # porque nunca vio ese nivel en el entrenamiento (mismo tipo de bug
    # que el KeyError original con Brentford, pero en este codigo nuevo).
    # Solucion: sembrar un prior neutro -- un equipo recien ascendido
    # rinde, en promedio, como un equipo promedio de la liga en esa
    # ventana. Es una aproximacion burda solo para folds sin historial;
    # en cuanto exista un debut real, _build_promoted_synthetic_rows ya
    # aporta senal de verdad y este prior deja de hacer falta.
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


def fit_poisson_model_v2(long_df: pd.DataFrame):
    """Igual que fit_poisson_model (v1), pero pondera cada partido por recencia via freq_weights."""
    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent)",
        data=long_df,
        family=sm.families.Poisson(),
        freq_weights=long_df["weight"],
    ).fit()
    return model


def predict_expected_goals_v2(model, home_team: str, away_team: str, known_teams: set) -> tuple:
    """
    Igual que predict_expected_goals (v1), pero si home_team/away_team no
    estan en known_teams (equipo recien ascendido sin historial en esta
    ventana de entrenamiento), se sustituye por PROMOTED_TEAM en vez de
    fallar o tener que excluir el partido.
    """
    home_label = home_team if home_team in known_teams else PROMOTED_LABEL
    away_label = away_team if away_team in known_teams else PROMOTED_LABEL

    home_row = pd.DataFrame({"team": [home_label], "opponent": [away_label], "is_home": [1]})
    away_row = pd.DataFrame({"team": [away_label], "opponent": [home_label], "is_home": [0]})
    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    return lambda_home, lambda_away


def predict_match_v2(model, home_team: str, away_team: str, known_teams: set, max_goals: int = MAX_GOALS) -> dict:
    """Pipeline completo v2 para un partido: goles esperados + probabilidades 1X2."""
    lambda_home, lambda_away = predict_expected_goals_v2(model, home_team, away_team, known_teams)
    matrix = scoreline_matrix(lambda_home, lambda_away, max_goals)
    prob_home, prob_draw, prob_away = outcome_probs_from_matrix(matrix)
    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "model_prob_home": prob_home,
        "model_prob_draw": prob_draw,
        "model_prob_away": prob_away,
    }


def predict_dataframe_v2(model, df: pd.DataFrame, known_teams: set, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    """
    A diferencia de predict_dataframe (v1), esta version NUNCA excluye
    partidos por equipo desconocido -- todos se predicen, usando la
    categoria PROMOTED_TEAM cuando corresponde.
    """
    records = []
    for _, row in df.iterrows():
        result = predict_match_v2(model, row["HomeTeam"], row["AwayTeam"], known_teams, max_goals)
        records.append({
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "model_prob_home": result["model_prob_home"],
            "model_prob_draw": result["model_prob_draw"],
            "model_prob_away": result["model_prob_away"],
        })
    return pd.DataFrame(records)