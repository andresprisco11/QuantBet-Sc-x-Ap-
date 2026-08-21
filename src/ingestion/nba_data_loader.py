"""
Fase 10 (NBA) -- primer script de ingesta de RESULTADOS de partidos de NBA
(distinto de cuotas -- eso ya lo cubre The Odds API / theoddsapi_historical_loader.py).
Sin esto, el loader de cuotas históricas no sirve para nada: hay cuotas de mercado
pero nada contra qué compararlas para backtesting.

FUENTE ELEGIDA (decisión del CTO, 2026-08-21): `nba_api`
(https://github.com/swar/nba_api), wrapper no oficial pero ampliamente usado de
stats.nba.com. Investigado antes de escribir código (mismo criterio que
nfl_data_loader.py/tennis_data_loader.py: nunca la primera fuente que aparece sin
mirar alternativas) -- es el estándar de facto de la comunidad de analítica de NBA
en Python, sin costo, sin API key.

**RIESGO OPERATIVO REAL, CONFIRMADO POR INVESTIGACIÓN (no una suposición) --
2026-08-21**: stats.nba.com bloquea requests que vienen de rangos de IP de
datacenters/hosting en la nube (confirmado un caso reportado y sin resolver para
Heroku, GitHub issue #320 del propio repo). Esto significa: **este script casi
seguro NO va a funcionar corrido desde un entorno de nube (incluido este mismo
entorno de Claude)** -- tiene que correrse desde tu máquina local (IP residencial),
que de todas formas es como está diseñado todo el proyecto (yo escribo el código,
vos lo corrés localmente). No es un bug si falla acá; sería raro que funcionara acá.

**ESQUEMA SIN CONFIRMAR TODAVÍA CON DATOS REALES** -- a diferencia del resto de los
loaders de este proyecto, este NO se pudo validar con una llamada real antes de
escribir el código (el entorno de desarrollo no tiene acceso a stats.nba.com por la
razón de arriba). `probe()` existe exactamente para eso: confirmar con una corrida
real tuya, antes de construir nada más sobre esto, el mismo criterio de disciplina
que ya se aplicó a Sportmonks/tennis-data.co.uk/nflverse -- la diferencia es que acá
la confirmación tiene que pasar por vos, no por mí.

**Cosas a confirmar con probe() (no asumidas)**:
1. Cobertura histórica real (LeagueGameFinder en teoría cubre desde 1946-47, pero
   "en teoría" no es "confirmado" -- probe() pide explícitamente una temporada vieja
   para chequear esto con un dato real).
2. Formato de las filas: LeagueGameFinder devuelve **una fila por EQUIPO por
   partido** (dos filas por partido -- ej. "LAL vs. BOS" y "BOS @ LAL" son las dos
   caras del mismo juego), NO una fila por partido como el resto del proyecto
   (football-data.co.uk, nflverse). El próximo script (clean_nba_data.py, no este)
   va a tener que pivotear esas dos filas en una sola fila home/away -- mismo tipo
   de decisión de diseño que ya se documentó para NFL con las cuotas americanas.
3. Si hay throttling real al pedir muchas temporadas seguidas -- por eso
   download_all() mete un delay fijo entre requests (práctica estándar reportada
   por la comunidad de nba_api, no inventada) y guardado incremental por temporada
   (mismo patrón resumible que thestatsapi_xg_loader.py/nfl_data_loader.py).

Requiere: pip install nba_api

Uso:
    python -m src.ingestion.nba_data_loader --probe
    python -m src.ingestion.nba_data_loader --download-all --start-season 1996-97 --end-season 2025-26
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_DATA_DIR

REQUEST_DELAY_SECONDS = 0.8  # practica estandar de la comunidad nba_api para no gatillar throttling


def _season_str_to_int(season: str) -> int:
    """'2023-24' -> 2023, para poder ordenar/filtrar temporadas."""
    return int(season.split("-")[0])


def _generate_seasons(start_season: str, end_season: str) -> list:
    start_year = _season_str_to_int(start_season)
    end_year = _season_str_to_int(end_season)
    seasons = []
    for year in range(start_year, end_year + 1):
        seasons.append(f"{year}-{str(year + 1)[-2:]}")
    return seasons


def _fetch_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError as e:
        raise ImportError("Falta 'nba_api'. Instalar con: pip install nba_api") from e

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        player_or_team_abbreviation="T",  # nivel equipo, no jugador
        league_id_nullable="00",  # NBA (00) -- distingue de WNBA/G-League en la misma API
    )
    return finder.get_data_frames()[0]


def probe() -> None:
    """Confirma con datos reales -- NO asumir -- antes de construir nada mas. Correr
    esto DESDE TU MAQUINA LOCAL, no en la nube (ver docstring del archivo)."""
    print("=== Paso 1: temporada reciente y completa (2023-24) ===")
    df_recent = _fetch_season("2023-24")
    print(f"Shape: {df_recent.shape}")
    print(f"Columnas ({len(df_recent.columns)}): {list(df_recent.columns)}")
    print(f"\nFilas por partido -- confirmar si son 2 filas/partido (una por equipo):")
    print(f"  GAME_ID unicos: {df_recent['GAME_ID'].nunique()}, filas totales: {len(df_recent)}, "
          f"ratio: {len(df_recent) / df_recent['GAME_ID'].nunique():.2f}")
    print(f"\nMuestra real (un mismo GAME_ID, las dos filas si existen):")
    sample_game_id = df_recent["GAME_ID"].iloc[0]
    print(df_recent[df_recent["GAME_ID"] == sample_game_id].to_string(index=False))

    print("\n=== Paso 2: chequeo de cobertura historica real (temporada vieja: 1996-97) ===")
    time.sleep(REQUEST_DELAY_SECONDS)
    try:
        df_old = _fetch_season("1996-97")
        print(f"1996-97: {len(df_old)} filas, {df_old['GAME_ID'].nunique()} partidos -- "
              f"{'SI hay datos reales' if len(df_old) > 0 else 'VACIO -- sin cobertura confirmada'}")
    except Exception as e:
        print(f"[ERROR] No se pudo confirmar 1996-97: {e}")

    print("\n[SIGUIENTE PASO MANUAL] Si esto se corto o dio timeout, es casi seguro el "
          "bloqueo de IP de nube documentado en el docstring -- confirmar corriendo esto "
          "desde tu maquina local, no desde este entorno.")


def download_all(start_season: str = "1996-97", end_season: str = "2025-26") -> None:
    out_dir = RAW_DATA_DIR / "NBA"
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = _generate_seasons(start_season, end_season)
    for season in seasons:
        out_path = out_dir / f"nba_{season.replace('-', '_')}.csv"
        if out_path.exists():
            print(f"{season}: ya existe -> {out_path}, se salta.")
            continue

        try:
            df = _fetch_season(season)
        except Exception as e:
            print(f"[ERROR] {season}: {e} -- se salta, correr de nuevo despues para reintentar.")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        df.to_csv(out_path, index=False)
        n_games = df["GAME_ID"].nunique()
        print(f"{season}: {len(df)} filas, {n_games} partidos -> {out_path}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDescarga terminada. Revisar {out_dir} -- recordar que son 2 filas por "
          f"partido (una por equipo), clean_nba_data.py (proximo script, no este) las "
          f"pivotea a 1 fila por partido igual que el resto del proyecto.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--start-season", default="1996-97")
    parser.add_argument("--end-season", default="2025-26")
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.download_all:
        download_all(args.start_season, args.end_season)
    else:
        parser.print_help()
