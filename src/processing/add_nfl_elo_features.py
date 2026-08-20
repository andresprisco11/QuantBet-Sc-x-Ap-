"""
Elo de NFL con ajuste por margen de victoria (MOV) y regresion a la media
entre temporadas -- mismo linaje de tecnica que ya funciono en tenis (Elo
walk-forward por rival vencido) y que se probo sin exito en futbol v7 (por
redundancia con C(team)/C(opponent), ver Fase 9) -- pero adaptado con dos
ingredientes que NFL si necesita y que ni tenis ni futbol tenian:

1. **Ajuste por margen de victoria (MOV)**: NFL juega muchos menos partidos
   por temporada (17) que futbol (34-38) o tenis (decenas por jugador top),
   asi que cada resultado individual pesa mucho mas -- un Elo binario
   (gano/perdio) desperdicia la informacion de POR CUANTO gano, que en NFL
   es una senal fuerte y estandar en la industria de analitica deportiva.
   Formula publicada de FiveThirtyEight ("NFL Elo Ratings", Nate Silver/
   Neil Paine, 2016), adaptada aqui, NO tuneada todavia contra los datos
   propios del proyecto:
     mov_multiplier = ln(|margen| + 1) * (2.2 / (0.001 * elo_diff_ganador + 2.2))
   donde elo_diff_ganador es la diferencia de rating (con ventaja de local
   ya sumada) entre el equipo que gano y el que perdio, ANTES del partido.
   Caso de empate (margen=0): ln(0+1)=0 -> el multiplicador da CERO, un
   empate no mueve el rating bajo la formula estricta -- decision de diseno
   explicita (empates son 0.21% del historico, ver clean_nfl_data.py), no
   un bug.

2. **Regresion a la media entre temporadas**: a diferencia de futbol/tenis
   (mismo plantel compitiendo semana a semana sin una "temporada nueva" que
   reinicie nada), en NFL el roster cambia sustancialmente cada anio
   (agencia libre, draft, retiros, lesiones de largo plazo) -- llevar el
   rating de diciembre intacto a septiembre siguiente sobre-pesa
   informacion vieja. Mismo criterio publicado de FiveThirtyEight: al
   inicio de cada temporada nueva, cada rating se regresiona 1/3 del camino
   de vuelta a 1500 (NO se reinicia del todo -- un equipo bueno el anio
   pasado sigue arrancando algo mejor que el promedio, solo que no igual de
   bien).

**Home field advantage**: constante fija (48 puntos Elo, cifra redonda
citada publicamente en el mismo trabajo de FiveThirtyEight) -- SIN TUNEAR
todavia, mismo criterio ya aplicado en futbol/tenis: se valida con el
backtest economico real de ESTE proyecto antes de tocarla, no se asume que
el valor publicado por otro grupo con otra metodologia es automaticamente
el optimo aca.

Walk-forward real y secuencial (orden cronologico por gameday/week), mismo
patron anti-fuga que add_team_elo_features.py (futbol): el rating ANTES de
cada partido se guarda en la fila, el diccionario de estado se actualiza
DESPUES.

Requiere que data/processed/NFL/matches_clean.csv ya exista (correr
clean_nfl_data.py primero). Agrega home_elo/away_elo, idempotente (mismo
criterio drop-antes-de-recalcular que el resto del proyecto).

Uso: python -m src.processing.add_nfl_elo_features
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "matches_clean.csv"

INITIAL_ELO = 1500.0
K_FACTOR = 20.0                 # publicado por FiveThirtyEight para NFL, sin tunear todavia.
HOME_ADVANTAGE = 48.0            # idem, cifra redonda publicada, sin tunear todavia.
SEASON_REGRESSION = 1.0 / 3.0    # idem: cuanto se regresiona cada rating a 1500 entre temporadas.

NEW_COLS = ["home_elo", "away_elo"]


def _mov_multiplier(point_margin_abs: float, elo_diff_winner: float) -> float:
    """Formula de FiveThirtyEight (NFL Elo), adaptada. Empate (margin=0) da
    multiplicador 0 -- decision de diseno explicita, ver docstring."""
    if point_margin_abs == 0:
        return 0.0
    return np.log(point_margin_abs + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))


def _regress_to_mean(elo: dict) -> dict:
    return {team: INITIAL_ELO + (rating - INITIAL_ELO) * (1.0 - SEASON_REGRESSION)
            for team, rating in elo.items()}


def _last_season_records(df: pd.DataFrame, season) -> pd.DataFrame:
    """Record real (W-L-T) de temporada regular por equipo -- se usa para un
    chequeo de sanidad OBJETIVO (correlacion Elo vs. % de victorias real),
    en vez de pedirle a una persona que reconozca equipos de NFL a ojo. No
    depende de que el usuario o el CTO sepan de memoria quien fue bueno esa
    temporada -- se verifica con los datos mismos."""
    season_df = df[(df["season"] == season) & (df["game_type"] == "REG")]
    records = defaultdict(lambda: {"W": 0, "L": 0, "T": 0})
    for row in season_df.itertuples(index=False):
        if row.FTR == "H":
            records[row.home_team]["W"] += 1
            records[row.away_team]["L"] += 1
        elif row.FTR == "A":
            records[row.away_team]["W"] += 1
            records[row.home_team]["L"] += 1
        else:
            records[row.home_team]["T"] += 1
            records[row.away_team]["T"] += 1

    rows = []
    for team, rec in records.items():
        games = rec["W"] + rec["L"] + rec["T"]
        win_rate = (rec["W"] + 0.5 * rec["T"]) / games if games else float("nan")
        rows.append({"team": team, "W": rec["W"], "L": rec["L"], "T": rec["T"], "win_rate": win_rate})
    return pd.DataFrame(rows).set_index("team")


def _add_nfl_elo_features(df: pd.DataFrame):
    df = df.sort_values(["gameday", "week"]).reset_index(drop=True)
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

        margin = row.point_margin  # home_score - away_score
        if row.FTR == "H":
            result_home = 1.0
            elo_diff_winner = elo_diff_home_perspective
        elif row.FTR == "A":
            result_home = 0.0
            elo_diff_winner = -elo_diff_home_perspective
        else:  # empate
            result_home = 0.5
            elo_diff_winner = 0.0

        mult = _mov_multiplier(abs(margin), elo_diff_winner)
        elo[home_team] = rating_home + K_FACTOR * mult * (result_home - expected_home)
        elo[away_team] = rating_away + K_FACTOR * mult * ((1.0 - result_home) - (1.0 - expected_home))

    df["home_elo"] = home_elo_before
    df["away_elo"] = away_elo_before
    return df, dict(elo)


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre 'python -m src.processing.clean_nfl_data' primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["gameday"] = pd.to_datetime(df["gameday"])

    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    df, final_elo = _add_nfl_elo_features(df)

    print(f"Partidos procesados: {len(df)}")
    elo_series = pd.Series(final_elo)
    print(f"Chequeo de sanidad basico -- Elo final (tras la ultima temporada procesada, {df['season'].max()}): "
          f"media={elo_series.mean():.1f} (esperado ~1500), desvio={elo_series.std():.1f}")

    # Chequeo de sanidad OBJETIVO: correlacion entre Elo final y record REAL de la
    # ultima temporada -- no depende de que nadie reconozca equipos de NFL a ojo,
    # se verifica con los datos mismos (mismo criterio "nunca asumir, confirmar con
    # datos reales" que el resto del proyecto).
    last_season = int(df["season"].max())
    records_df = _last_season_records(df, last_season)
    comparison = records_df.join(elo_series.rename("elo_final"), how="inner")
    corr = comparison["win_rate"].corr(comparison["elo_final"])
    print(f"\nChequeo de sanidad OBJETIVO -- correlacion Elo final vs. % de victorias real "
          f"de temporada regular {last_season}: {corr:.3f} "
          f"(esperado fuertemente positivo, >0.6-0.7, si el sistema esta capturando fuerza real; "
          f"cerca de 0 o negativo seria señal de un bug real, no una opinion sobre equipos).")

    top10 = comparison.sort_values("elo_final", ascending=False).head(10)
    print(f"\nTop 10 por Elo final, con su record REAL {last_season} (W-L-T) al lado -- la señal de "
          f"alarma es un equipo arriba en Elo con mal record, no si el nombre 'suena' fuerte:")
    for team, row in top10.iterrows():
        print(f"  {team}: elo={row['elo_final']:.0f}  record={int(row['W'])}-{int(row['L'])}-{int(row['T'])}  "
              f"win_rate={row['win_rate']:.1%}")

    df.to_csv(DATA_PATH, index=False)
    print(f"\nGuardado (columnas home_elo/away_elo agregadas) -> {DATA_PATH}")


if __name__ == "__main__":
    run()