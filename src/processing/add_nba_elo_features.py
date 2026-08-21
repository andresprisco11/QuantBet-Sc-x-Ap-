"""
Fase 10 (NBA) -- primer feature de fuerza de equipo para NBA, mismo linaje
de tecnica que ya funciono en tenis y NFL (Elo walk-forward con ajuste por
margen de victoria). Sin esto no hay nada que un modelo pueda usar todavia
-- `games_clean.csv` solo tiene resultados crudos, ningun predictor.

**Por que MOV-Elo (no Elo binario simple)**: NBA juega 82 partidos por
temporada regular (mucho mas que los 17 de NFL), pero el margen de victoria
sigue siendo una señal fuerte y estandar en analitica de NBA -- ganar por 30
vs. ganar por 2 no deberia mover el rating igual. Se usa la formula publicada
de FiveThirtyEight ("NBA Elo Ratings" / metodologia RAPTOR), ADAPTADA aca,
NO tuneada todavia contra los datos propios del proyecto -- mismo criterio
explicito que ya se aplico en `add_nfl_elo_features.py` y
`add_team_elo_features.py` (futbol): se usa el valor publicado como punto de
partida honesto, se valida despues contra el backtest economico real de
ESTE proyecto, no se asume que el valor de otro grupo es automaticamente el
optimo aca.

Constantes publicadas usadas (distintas de NFL a proposito -- no se reciclan
los valores de otro deporte sin revisar si aplican):
- INITIAL_ELO = 1500, K_FACTOR = 20, HOME_ADVANTAGE = 100 (puntos Elo, solo
  afecta la expectativa, el sistema sigue siendo zero-sum).
- MOV multiplier: ((|margen| + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_ganador)
  -- formula NBA especifica de FiveThirtyEight, DISTINTA de la formula
  logaritmica que usa NFL (`ln(|margen|+1) * (2.2/(0.001*elo_diff+2.2))`) --
  no es un error, son formulas publicadas distintas por deporte.
- Regresion a la media entre temporadas: 25% (0.25) hacia 1500 -- distinto
  del 1/3 que usa NFL, valor especifico publicado para NBA (temporada de 82
  partidos retiene mas señal entre años que la de 17 de NFL, de ahi la
  regresion mas chica).

**NBA nunca empata** (confirmado en `clean_nba_data.py`, chequeo de sanidad
ya validado con 0 empates reales en 35,546 partidos) -- a diferencia de NFL,
no hace falta un caso R=0.5, el resultado es siempre 1.0 o 0.0.

Walk-forward real y secuencial (orden cronologico por game_date), mismo
patron anti-fuga que el resto del proyecto: el rating ANTES de cada partido
se guarda en la fila, el diccionario de estado se actualiza DESPUES.

Requiere que data/processed/NBA/games_clean.csv ya exista (correr
clean_nba_data.py primero). Agrega home_elo/away_elo, idempotente (mismo
criterio drop-antes-de-recalcular que el resto del proyecto).

Uso: python -m src.processing.add_nba_elo_features
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"

INITIAL_ELO = 1500.0
K_FACTOR = 20.0                  # publicado por FiveThirtyEight para NBA, sin tunear todavia.
HOME_ADVANTAGE = 100.0            # idem.
SEASON_REGRESSION = 0.25          # idem: cuanto se regresiona cada rating a 1500 entre temporadas.

NEW_COLS = ["home_elo", "away_elo"]


def _mov_multiplier(point_margin_abs: float, elo_diff_winner: float) -> float:
    """Formula NBA de FiveThirtyEight (distinta de la de NFL -- ver docstring)."""
    return ((point_margin_abs + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)


def _regress_to_mean(elo: dict) -> dict:
    return {team: INITIAL_ELO + (rating - INITIAL_ELO) * (1.0 - SEASON_REGRESSION)
            for team, rating in elo.items()}


def _last_season_records(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Record real (W-L) de temporada por equipo -- chequeo de sanidad OBJETIVO
    (correlacion Elo vs. % de victorias real), mismo criterio que
    add_nfl_elo_features.py: no depende de que nadie reconozca equipos de NBA
    a ojo, se verifica con los datos mismos."""
    season_df = df[df["season"] == season]
    records = defaultdict(lambda: {"W": 0, "L": 0})
    for row in season_df.itertuples(index=False):
        if row.FTR == "H":
            records[row.home_team]["W"] += 1
            records[row.away_team]["L"] += 1
        else:
            records[row.away_team]["W"] += 1
            records[row.home_team]["L"] += 1

    rows = []
    for team, rec in records.items():
        games = rec["W"] + rec["L"]
        win_rate = rec["W"] / games if games else float("nan")
        rows.append({"team": team, "W": rec["W"], "L": rec["L"], "win_rate": win_rate})
    return pd.DataFrame(rows).set_index("team")


def _add_nba_elo_features(df: pd.DataFrame):
    df = df.sort_values("game_date").reset_index(drop=True)
    elo = defaultdict(lambda: INITIAL_ELO)

    home_elo_before, away_elo_before = [], []
    current_season = None

    for row in df.itertuples(index=False):
        if current_season is not None and row.season != current_season:
            elo = defaultdict(lambda: INITIAL_ELO, _regress_to_mean(dict(elo)))
        current_season = row.season

        home_team, away_team = row.home_team, row.away_team
        rating_home = elo[home_team]
        rating_away = elo[away_team]
        home_elo_before.append(rating_home)
        away_elo_before.append(rating_away)

        home_adj = rating_home + HOME_ADVANTAGE
        elo_diff_home_perspective = home_adj - rating_away
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff_home_perspective / 400.0))

        margin = row.point_margin  # home_pts - away_pts
        if row.FTR == "H":
            result_home = 1.0
            elo_diff_winner = elo_diff_home_perspective
        else:
            result_home = 0.0
            elo_diff_winner = -elo_diff_home_perspective

        mult = _mov_multiplier(abs(margin), elo_diff_winner)
        elo[home_team] = rating_home + K_FACTOR * mult * (result_home - expected_home)
        elo[away_team] = rating_away + K_FACTOR * mult * ((1.0 - result_home) - (1.0 - expected_home))

    df["home_elo"] = home_elo_before
    df["away_elo"] = away_elo_before
    return df, dict(elo)


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre 'python -m src.processing.clean_nba_data' primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])

    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    df, final_elo = _add_nba_elo_features(df)

    print(f"Partidos procesados: {len(df)}")
    elo_series = pd.Series(final_elo)
    print(f"Chequeo de sanidad basico -- Elo final (tras la ultima temporada procesada, "
          f"{df['season'].iloc[-1]}): media={elo_series.mean():.1f} (esperado ~1500), "
          f"desvio={elo_series.std():.1f}")

    last_season = df["season"].iloc[-1]
    records_df = _last_season_records(df, last_season)
    comparison = records_df.join(elo_series.rename("elo_final"), how="inner")
    corr = comparison["win_rate"].corr(comparison["elo_final"])
    print(f"\nChequeo de sanidad OBJETIVO -- correlacion Elo final vs. % de victorias real "
          f"de temporada {last_season}: {corr:.3f} "
          f"(esperado fuertemente positivo, >0.6-0.7, si el sistema esta capturando fuerza real; "
          f"cerca de 0 o negativo seria señal de un bug real, no una opinion sobre equipos).")

    top10 = comparison.sort_values("elo_final", ascending=False).head(10)
    print(f"\nTop 10 por Elo final, con su record REAL {last_season} (W-L) al lado:")
    for team, row in top10.iterrows():
        print(f"  {team}: elo={row['elo_final']:.0f}  record={int(row['W'])}-{int(row['L'])}  "
              f"win_rate={row['win_rate']:.1%}")

    df.to_csv(DATA_PATH, index=False)
    print(f"\nGuardado (columnas home_elo/away_elo agregadas) -> {DATA_PATH}")


if __name__ == "__main__":
    run()
