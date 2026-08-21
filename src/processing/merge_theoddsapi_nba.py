"""
Fase 10 (NBA) -- pega las cuotas historicas reales de The Odds API
(`theoddsapi_historical_loader.py`) contra `games_clean.csv`, mismo principio
de separacion que `merge_thestatsapi_xg.py` (futbol): ingesta y calculo de
probabilidad NUNCA en el mismo script que la descarga cruda.

**Llave de merge -- confirmada con datos reales, NO asumida**: a diferencia
del merge de xG de futbol (que necesito un NAME_MAPS a mano por diferencias
de nombre entre fuentes), aca los nombres de equipo YA coinciden exacto entre
las dos fuentes (confirmado: los 30 equipos actuales de `NBA_historical_odds_
raw.csv` tienen match exacto de string contra `TEAM_NAME` en `games_clean.csv`,
0 nombres sin match) -- no hace falta ningun diccionario de mapeo.

La parte que SI habia que confirmar con datos reales (y se confirmo, no se
asumio): el campo correcto para la llave de fecha es `snapshot_date`, NO la
fecha UTC de `commence_time`. Ejemplo real verificado: el partido Portland
Trail Blazers vs. Sacramento Kings con `game_date=2026-04-12` en games_clean
tiene `commence_time=2026-04-13T00:40:00Z` (un dia UTC adelante, normal para
un partido nocturno en horario Pacifico) pero `snapshot_date=2026-04-12`
-- exactamente igual a game_date. `snapshot_date` ya viene en la convencion
de "noche de partido" de EEUU porque asi lo definio `theoddsapi_historical_
loader.py` al armar el timestamp del snapshot -- usarlo directo evita
cualquier conversion de zona horaria.

**Cuidado real encontrado con snapshots repetidos de partidos ya jugados**:
un mismo partido a veces sigue apareciendo en snapshots de dias POSTERIORES
al partido real (el evento queda "colgado" en la respuesta de la API unos
dias antes de que se archive del todo -- mismo fenomeno ya documentado para
el hueco de las Finals 2020 en el roadmap). Esto NO rompe el merge: como se
hace join exacto por `snapshot_date == game_date`, esos snapshots repetidos
de dias posteriores simplemente no matchean ningun partido real de esa fecha
y se ignoran solos -- no hace falta filtrarlos a mano.

**Calculo de probabilidad no-vig -- consenso multi-libro, no un solo libro
sharp** (Pinnacle no esta disponible, confirmado en `theoddsapi_loader.py`):
por cada partido, se calcula la probabilidad implicita no-vig de CADA casa
de apuestas (formula de odds americanas + normalizacion a 2 vias, mismas
formulas ya usadas en `clean_nfl_data.py`), y despues se promedia entre
todas las casas disponibles ese partido -- el benchmark de mercado es el
CONSENSO de las casas retail confirmadas (DraftKings/FanDuel/BetMGM/etc.),
no la cuota de una sola casa.

Consolida data/processed/NBA/games_clean.csv +
data/raw/THEODDSAPI/NBA_historical_odds_raw.csv ->
data/processed/NBA/games_clean_with_odds.csv

Uso: python -m src.processing.merge_theoddsapi_nba
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

GAMES_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"
ODDS_PATH = RAW_DATA_DIR / "THEODDSAPI" / "NBA_historical_odds_raw.csv"
OUT_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean_with_odds.csv"

# Unica inconsistencia de nombre CONFIRMADA con datos reales (2026-08-21) entre
# las dos fuentes: nba_api usa "LA Clippers" en algunas temporadas y "Los Angeles
# Clippers" en otras para el MISMO equipo (882 vs. 1510 apariciones en
# games_clean.csv), mientras que The Odds API SIEMPRE usa "Los Angeles Clippers"
# (17,576 apariciones, 0 veces "LA Clippers", confirmado revisando el archivo
# completo de cuotas). Sin este alias, todo partido de Clippers etiquetado "LA
# Clippers" en games_clean fallaba el merge -- 482 de los 784 partidos sin cuota
# de la primera corrida eran exactamente esto. Resto de los 30 equipos: match
# exacto de string confirmado, sin necesidad de ningun otro alias.
TEAM_NAME_ALIASES = {"LA Clippers": "Los Angeles Clippers"}


def _american_to_prob_vec(odds: pd.Series) -> pd.Series:
    """Formula estandar de odds americanas, VECTORIZADA (np.where) -- misma
    formula que clean_nfl_data.py (NO 1/odds, esa es la de odds decimales
    que usa el resto del proyecto). Vectorizado porque el archivo de cuotas
    de NBA tiene ~262k filas -- un loop fila-por-fila con .apply()/groupby()
    tardaba minutos, esto tarda segundos (confirmado con un timeout real en
    la primera version de este script antes de vectorizarlo)."""
    import numpy as np
    return pd.Series(
        np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0)),
        index=odds.index,
    )


def _consensus_market_prob(odds: pd.DataFrame) -> pd.DataFrame:
    """Por cada (snapshot_date, home_team, away_team), promedia la
    probabilidad no-vig del local entre TODAS las casas disponibles ese
    partido -- consenso multi-libro, no un solo libro sharp (Pinnacle no
    esta disponible en este proveedor, confirmado). Version vectorizada:
    separa filas de local/visitante y las junta lado a lado con un merge
    en vez de iterar grupo por grupo (ver _american_to_prob_vec)."""
    h2h = odds[odds["market"] == "h2h"].copy()

    key_cols = ["snapshot_date", "event_id", "home_team", "away_team", "bookmaker"]
    home_side = h2h[h2h["outcome_name"] == h2h["home_team"]][key_cols + ["outcome_price"]]
    home_side = home_side.rename(columns={"outcome_price": "home_price"})
    away_side = h2h[h2h["outcome_name"] == h2h["away_team"]][key_cols + ["outcome_price"]]
    away_side = away_side.rename(columns={"outcome_price": "away_price"})

    per_book = home_side.merge(away_side, on=key_cols, how="inner")

    prob_home_raw = _american_to_prob_vec(per_book["home_price"])
    prob_away_raw = _american_to_prob_vec(per_book["away_price"])
    total = prob_home_raw + prob_away_raw
    per_book["book_prob_home_novig"] = prob_home_raw / total  # normalizacion no-vig, vectorizada

    consensus = (
        per_book.groupby(["snapshot_date", "home_team", "away_team"])
        .agg(market_prob_home=("book_prob_home_novig", "mean"),
             n_bookmakers=("bookmaker", "nunique"))
        .reset_index()
    )
    return consensus


def run() -> None:
    if not GAMES_PATH.exists():
        print(f"[SKIP] No existe {GAMES_PATH} -- corre clean_nba_data.py primero.")
        return
    if not ODDS_PATH.exists():
        print(f"[SKIP] No existe {ODDS_PATH} -- corre theoddsapi_historical_loader.py primero.")
        return

    games = pd.read_csv(GAMES_PATH)
    odds_raw = pd.read_csv(ODDS_PATH)

    consensus = _consensus_market_prob(odds_raw)
    print(f"Partidos con consenso de mercado calculado: {len(consensus)} "
          f"(promedio de {consensus['n_bookmakers'].mean():.1f} casas por partido)")

    # Alias de nombre para el merge -- ver TEAM_NAME_ALIASES arriba. Se aplica solo
    # a columnas temporales de join, NO se sobreescribe home_team/away_team del
    # dataset final (esas quedan con el nombre real que uso nba_api ese partido).
    games["_home_join"] = games["home_team"].replace(TEAM_NAME_ALIASES)
    games["_away_join"] = games["away_team"].replace(TEAM_NAME_ALIASES)

    merged = games.merge(
        consensus,
        left_on=["game_date", "_home_join", "_away_join"],
        right_on=["snapshot_date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_odds"),
    ).drop(columns=["snapshot_date", "_home_join", "_away_join", "home_team_odds", "away_team_odds"])

    n_with_odds = int(merged["market_prob_home"].notna().sum())
    print(f"\nTotal de partidos en games_clean.csv: {len(merged)}")
    print(f"Partidos con cuota de mercado pegada: {n_with_odds} ({n_with_odds/len(merged):.1%})")

    # Chequeo de cobertura sobre el rango donde SI deberia haber cuota (2020-10-01
    # en adelante, limite real del proveedor -- confirmado en theoddsapi_historical_
    # loader.py) -- fuera de ese rango el 0% de cobertura es esperado, no un error.
    in_range = merged[merged["game_date"] >= "2020-10-01"]
    n_in_range_with_odds = int(in_range["market_prob_home"].notna().sum())
    print(f"\nDentro del rango con cuotas disponibles (game_date>=2020-10-01): "
          f"{len(in_range)} partidos, {n_in_range_with_odds} con cuota pegada "
          f"({n_in_range_with_odds/len(in_range):.1%}).")
    if n_in_range_with_odds / len(in_range) < 0.90:
        print(f"[AVISO] cobertura por debajo del 90% dentro del rango esperado -- revisar "
              f"si faltan dias de descarga (correr theoddsapi_historical_loader.py de nuevo) "
              f"o si hay playoffs/partidos especiales sin cuotas de temporada regular.")

    # Chequeo de sanidad OBJETIVO -- el mercado deberia estar bien calibrado
    # (Brier bajo) casi por definicion, esto es solo para detectar un bug real de
    # mapeo (ej. probabilidad de local invertida con la de visitante).
    reliable = merged["market_prob_home"].notna()
    actual = merged.loc[reliable, "FTR"].map({"H": 1.0, "A": 0.0})
    market_brier = float(((merged.loc[reliable, "market_prob_home"] - actual) ** 2).mean())
    print(f"\nChequeo de sanidad -- Brier score del mercado (consenso no-vig) sobre los "
          f"{int(reliable.sum())} partidos con cuota: {market_brier:.6f} "
          f"(deberia ser BAJO, tipicamente <0.23 en NBA -- un numero alto o cercano a 0.5 "
          f"seria señal de un bug real de mapeo home/away, no una opinion sobre el mercado).")

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nGuardado -> {OUT_PATH}")


if __name__ == "__main__":
    run()
