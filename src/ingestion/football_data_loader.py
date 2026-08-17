"""
Modulo de ingesta: descarga historicos de partidos + cuotas desde
football-data.co.uk y los persiste crudos en data/raw/.

Incluye reintentos con backoff porque la conexion a este sitio puede
ser inestable -- no reintenta de inmediato, espera un poco mas cada vez.

Incluye tambien validacion de integridad de liga: football-data.co.uk
puede devolver 200 OK con contenido incorrecto (ej. cuando el archivo
de una temporada nueva aun no existe, a veces sirve datos de otra
division en su lugar). Verificamos que la columna 'Div' del CSV
coincida con el codigo de liga esperado antes de aceptar los datos.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, RAW_DATA_DIR, SEASONS

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"


class LeagueMismatchError(Exception):
    """Se lanza cuando el CSV descargado no corresponde a la liga esperada."""
    pass


def download_season(league_key: str, season: str, max_retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    league_code = LEAGUES[league_key]["code"]
    url = BASE_URL.format(season=season, league_code=league_code)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            from io import StringIO

            # Estos CSVs a veces traen un BOM (marca de orden de bytes) al
            # inicio del archivo. Si decodificamos siempre como latin-1, ese
            # BOM se corrompe y queda pegado al nombre de la primera columna
            # (ej. "Div" se convierte en "ï»¿Div"), lo cual rompe cualquier
            # chequeo posterior que busque la columna "Div" exacta.
            # utf-8-sig decodifica y quita el BOM automaticamente si existe;
            # si el archivo no es UTF-8 valido (nombres con caracteres raros),
            # caemos de vuelta a latin-1.
            raw_bytes = response.content
            try:
                text = raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")

            df = pd.read_csv(StringIO(text))
            # Defensa adicional: normalizar nombres de columna por si queda
            # basura de encoding pegada (espacios, BOM residual, etc.).
            df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]

            # --- Validacion de integridad: la Div del CSV debe coincidir ---
            # con el codigo de liga pedido. Si no coincide, el sitio nos
            # sirvio datos incorrectos (ej. temporada aun no publicada).
            if "Div" in df.columns:
                actual_divs = set(df["Div"].dropna().astype(str).str.strip().unique())
                if actual_divs and league_code not in actual_divs:
                    raise LeagueMismatchError(
                        f"{league_key} {season}: se esperaba Div='{league_code}' "
                        f"pero el CSV trae {actual_divs}. Probablemente la temporada "
                        f"aun no esta publicada en football-data.co.uk. Descartando."
                    )
            else:
                print(f"  [AVISO] {league_key} {season}: no se encontro columna 'Div' "
                      f"para validar (columnas: {list(df.columns)[:5]}...). Se acepta sin validar.")

            df["season"] = season
            df["league_key"] = league_key
            return df
        except LeagueMismatchError:
            raise
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait = attempt * 3
                print(f"  [REINTENTO {attempt}/{max_retries}] {season}: {type(e).__name__} -> esperando {wait}s...")
                time.sleep(wait)

    raise last_error


def save_raw(df: pd.DataFrame, league_key: str, season: str) -> Path:
    league_dir = RAW_DATA_DIR / league_key
    league_dir.mkdir(parents=True, exist_ok=True)
    filepath = league_dir / f"{season}.csv"
    df.to_csv(filepath, index=False)
    return filepath


def ingest_league(league_key: str) -> None:
    for season in SEASONS:
        try:
            df = download_season(league_key, season)
            filepath = save_raw(df, league_key, season)
            print(f"[OK] {league_key} {season}: {len(df)} partidos -> {filepath}")
        except LeagueMismatchError as e:
            print(f"[SKIP] {e}")
            _remove_stale_file(league_key, season)
        except requests.exceptions.RequestException as e:
            print(f"[SKIP] {league_key} {season}: no disponible tras varios intentos ({type(e).__name__})")


def _remove_stale_file(league_key: str, season: str) -> None:
    """Si ya existe un CSV crudo de una temporada que ahora falla la
    validacion de liga, lo borramos -- puede ser un archivo contaminado
    de una corrida anterior (antes de que existiera esta validacion)."""
    filepath = RAW_DATA_DIR / league_key / f"{season}.csv"
    if filepath.exists():
        filepath.unlink()
        print(f"  [LIMPIEZA] Se elimino {filepath} (datos invalidos de una corrida anterior).")


if __name__ == "__main__":
    for league_key in LEAGUES:
        ingest_league(league_key)