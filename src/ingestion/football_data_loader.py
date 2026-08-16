"""
Módulo de ingesta: descarga históricos de partidos + cuotas desde
football-data.co.uk y los persiste crudos en data/raw/.

Diseño modular: agregar una nueva liga/deporte = agregar una entrada
en config.settings.LEAGUES, no reescribir esta lógica.
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, RAW_DATA_DIR, SEASONS

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"


def download_season(league_key: str, season: str) -> pd.DataFrame:
    """
    Descarga el CSV de una temporada/liga específica y lo devuelve como DataFrame.
    No transforma nada — ingesta cruda, la limpieza va en un módulo aparte.
    """
    league_code = LEAGUES[league_key]["code"]
    url = BASE_URL.format(season=season, league_code=league_code)

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    # football-data.co.uk usa encoding latin-1 en varios CSVs históricos
    from io import StringIO
    df = pd.read_csv(StringIO(response.content.decode("latin-1")))
    df["season"] = season
    df["league_key"] = league_key

    return df


def save_raw(df: pd.DataFrame, league_key: str, season: str) -> Path:
    """Guarda el CSV crudo en data/raw/<liga>/<temporada>.csv"""
    league_dir = RAW_DATA_DIR / league_key
    league_dir.mkdir(parents=True, exist_ok=True)

    filepath = league_dir / f"{season}.csv"
    df.to_csv(filepath, index=False)
    return filepath


def ingest_league(league_key: str) -> None:
    """Descarga y guarda todas las temporadas configuradas para una liga."""
    for season in SEASONS:
        try:
            df = download_season(league_key, season)
            filepath = save_raw(df, league_key, season)
            print(f"[OK] {league_key} {season}: {len(df)} partidos -> {filepath}")
        except requests.exceptions.HTTPError as e:
            print(f"[SKIP] {league_key} {season}: no disponible ({e})")


if __name__ == "__main__":
    for league_key in LEAGUES:
        ingest_league(league_key)
        