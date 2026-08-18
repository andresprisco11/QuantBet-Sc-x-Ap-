"""
Modulo de ingesta para MLS (Fase 8b) -- deliberadamente SEPARADO de
football_data_loader.py, no una extension del mismo dentro de LEAGUES.

Por que un modulo aparte, no "una liga mas" en config/settings.py:
football-data.co.uk sirve las 4 ligas europeas actuales como un CSV POR
TEMPORADA (mmz4281/{season}/{code}.csv, mismo esquema de columnas en las
4 -- ver LEAGUES en config/settings.py). MLS vive en 'new/USA.csv': UN
SOLO archivo con TODAS las temporadas juntas (no hay que iterar sobre
SEASONS), y con un esquema de columnas distinto, confirmado en una sesion
anterior de este proyecto: Country, League, Season, Date, Time, Home,
Away, HG, AG, Res, PSCH, PSCD, PSCA, MaxCH/D/A, AvgCH/D/A, BFECH/D/A,
B365CH/D/A -- SIN columnas de tiros/corners (HS/AS/HST/AST/HC/AC no
existen) y SIN cuota de APERTURA de Pinnacle (solo PSCH/PSCD/PSCA de
CIERRE, no hay PSH/PSD/PSA) -- a diferencia de las 4 ligas europeas.
Tambien los nombres de gol/resultado son otros: HG/AG/Res en vez de
FTHG/FTAG/FTR.

IMPORTANTE -- este modulo NO asume ese esquema a ciegas. En esta sesion
intente re-verificarlo en vivo (WebFetch) antes de escribir este archivo
y el sitio devolvio HTTP 429 (rate limit) dos veces seguidas -- no pude
re-confirmar el header exacto ahora mismo. En vez de arriesgarme a
hardcodear columnas de memoria (exactamente el tipo de error que este
proyecto ya cazo varias veces con EPL), este loader valida en tiempo de
ejecucion: imprime TODAS las columnas reales que trae el archivo, y los
valores unicos de 'Country'/'League' encontrados, ANTES de filtrar o
guardar nada. Si el sitio cambio el formato desde la ultima verificacion,
esto lo va a mostrar de inmediato en la salida de consola -- pegame ese
output apenas lo corras, antes de que sigamos con el siguiente paso.

Este modulo hace SOLO la descarga + guardado crudo (equivalente a
ingest_league() de football_data_loader.py). NO intenta adaptar
clean_data.py todavia -- eso es un paso separado A PROPOSITO, porque
depende de las columnas reales que confirmemos con la salida de esta
corrida (ver roadmap, Fase 8b). En particular: sin cuota de apertura, el
CLV tal como esta definido hoy (cierre - apertura) NO se puede calcular
para MLS de la misma forma -- esa decision se toma explicitamente en el
siguiente paso, no aca.
"""
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MLS_SOURCE, RAW_DATA_DIR


class MLSDataUnavailableError(Exception):
    """Se lanza cuando el archivo de MLS no se puede descargar o no tiene
    el contenido esperado (mismo tratamiento defensivo que
    SeasonUnavailableError en football_data_loader.py)."""
    pass


def download_mls(timeout: int = 30) -> pd.DataFrame:
    url = MLS_SOURCE["url"]
    response = requests.get(url, timeout=timeout)

    if response.status_code != 200:
        raise MLSDataUnavailableError(
            f"MLS: HTTP {response.status_code} en vez de 200 al pedir {url}."
        )

    raw_bytes = response.content
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    try:
        df = pd.read_csv(StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise MLSDataUnavailableError(
            f"MLS: la respuesta (HTTP 200) no es un CSV valido ({type(e).__name__}: {e})."
        )

    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]

    print(f"[DIAGNOSTICO] MLS: {len(df)} filas descargadas de {url}")
    print(f"[DIAGNOSTICO] Columnas encontradas ({len(df.columns)}): {list(df.columns)}")

    country_col = MLS_SOURCE.get("country_column", "Country")
    if country_col in df.columns:
        print(f"[DIAGNOSTICO] Valores unicos en '{country_col}': "
              f"{sorted(df[country_col].dropna().astype(str).unique().tolist())}")
        # Chequeo adicional, no solo Country: EEUU tiene mas de una division
        # de futbol (MLS es primera division, pero USL Championship/USL
        # League One son divisiones inferiores) y football-data.co.uk podria
        # estar sirviendo mas de una bajo el mismo 'Country'. Si la columna
        # 'League' trae mas de un valor, no se puede asumir a ciegas que
        # todo el archivo es MLS -- se imprime para decidir con datos reales,
        # no queda filtrado automaticamente todavia.
        if "League" in df.columns:
            league_values = sorted(df["League"].dropna().astype(str).unique().tolist())
            print(f"[DIAGNOSTICO] Valores unicos en 'League' (sin filtrar todavia): {league_values}")
            if len(league_values) > 1:
                print(f"[AVISO] Hay {len(league_values)} valores distintos en 'League' -- si alguno "
                      f"NO es MLS (ej. una division inferior de EEUU), hay que filtrar tambien por "
                      f"'League' antes de guardar, o el dataset de 'MLS' va a traer partidos que no "
                      f"son MLS. Revisar antes del siguiente paso.")
    else:
        print(f"[AVISO] No se encontro la columna '{country_col}' esperada para filtrar "
              f"MLS de otras ligas/paises menores que este archivo pueda traer junto. "
              f"Columnas disponibles arriba -- revisar MLS_SOURCE['country_column'] en "
              f"config/settings.py antes de confiar en el filtro. Se devuelve el archivo "
              f"COMPLETO sin filtrar para que lo puedas inspeccionar.")
        return df

    filter_value = MLS_SOURCE["country_filter_value"]
    df_mls = df[df[country_col].astype(str).str.strip() == filter_value].copy()

    if df_mls.empty:
        raise MLSDataUnavailableError(
            f"MLS: se filtro '{country_col}' == '{filter_value}' pero quedaron 0 filas -- "
            f"el valor esperado probablemente cambio. Ver los valores unicos impresos "
            f"arriba y actualizar MLS_SOURCE['country_filter_value'] en config/settings.py."
        )

    print(f"[DIAGNOSTICO] Tras filtrar {country_col}=='{filter_value}': {len(df_mls)} filas.")
    df_mls["league_key"] = "MLS"
    return df_mls


def save_raw_mls(df: pd.DataFrame) -> Path:
    """A diferencia de save_raw() en football_data_loader.py, ac no hay un
    archivo por temporada -- el sitio ya entrega todo junto en un CSV, asi
    que se guarda tal cual en un unico archivo crudo."""
    league_dir = RAW_DATA_DIR / "MLS"
    league_dir.mkdir(parents=True, exist_ok=True)
    filepath = league_dir / "all_seasons.csv"
    df.to_csv(filepath, index=False)
    return filepath


def ingest_mls() -> None:
    try:
        df = download_mls()
    except MLSDataUnavailableError as e:
        print(f"[SKIP] {e}")
        return

    filepath = save_raw_mls(df)
    print(f"[OK] MLS: {len(df)} partidos -> {filepath}")

    if "Season" in df.columns:
        seasons = sorted(df["Season"].dropna().astype(str).unique().tolist())
        print(f"[DIAGNOSTICO] Temporadas presentes en el archivo: {seasons}")


if __name__ == "__main__":
    ingest_mls()