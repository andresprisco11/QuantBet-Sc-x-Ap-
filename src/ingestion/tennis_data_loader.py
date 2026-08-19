"""
Escalamiento a Tenis (mandato original del proyecto: "de futbol a NFL,
Tenis y NBA"). Segundo deporte del proyecto -- decisión del CTO
(2026-08-19): Tenis antes que NFL, porque reutiliza casi toda la
arquitectura de probabilidad/blend ya construida (mismo proveedor de
cuotas, Pinnacle, y resultado binario sin empate) en vez de requerir un
modelo estadistico nuevo como NFL (spread de puntos, no moneyline).

DISCIPLINA DEL PROYECTO, sin excepcion aca tampoco: "nunca adivinar un
esquema/endpoint/columna -- confirmar con datos reales antes de construir
nada". `probe()` (ya corrido y confirmado, 2026-08-19) trajo un archivo
real (ATP 2025: 2,644 partidos, 38 columnas) y confirmo el esquema contra
notes.txt antes de construir `download_all()`.

TERMINOS DE USO -- diligencia hecha, confianza MODERADA (no tan explicita
como football-data.co.uk, pero razonable, ver detalle):
- tennis-data.co.uk es del mismo grupo que football-data.co.uk
  (confirmado, referencia a "Football Data" como sitio hermano).
- El disclaimer especifico (http://www.tennis-data.co.uk/disclaimer.php,
  revisado por el usuario en su navegador 2026-08-19) resulto ser SOLO
  sobre responsabilidad de perdidas de apuestas -- no dice nada sobre
  permiso o restriccion de uso de los datos en si (a diferencia del
  disclaimer de football-data.co.uk, que si autoriza explicitamente
  "prediccion de partidos de liga"). notes.txt dice que Tennis-Data
  "mantiene copyright completo sobre los archivos", pero tampoco prohibe
  el uso para investigacion/prediccion.
- Señal a favor: existe un paquete publicado en CRAN (repositorio de R
  con revision, no un sitio cualquiera) hecho especificamente para
  descargar datos de este sitio, y proyectos de ML publicos que usan este
  mismo dataset para investigacion -- evidencia de que la comunidad lo
  trata como un dataset abierto para este uso, distinto del caso
  Understat/FBref (bloqueo/prohibicion explicita).
- Decision del CTO (2026-08-19): proceder, con el mismo perfil bajo que
  football-data.co.uk (un archivo por año/tour, sin scraping agresivo,
  sin republicar los datos). Si el proyecto alguna vez escala mas alla de
  uso propio de un solo usuario, revisar esto de nuevo con mas rigor.

Fuente de datos: XLSX (no CSV -- se intento adivinar una URL de CSV en la
version anterior de este script y NO existio, HTTP 300). Confirmado via
alldata.php + probe(): ATP en
http://www.tennis-data.co.uk/{year}/{year}.xlsx, WTA en
http://www.tennis-data.co.uk/{year}w/{year}.xlsx. Cobertura historica
segun el sitio: ATP desde 2000, WTA desde 2007 (no se descarga toda esa
profundidad automaticamente por default -- ver --start-year).

Requiere 'openpyxl' instalado para que pandas pueda leer .xlsx
(pip install openpyxl --si hace falta).

Guarda un CSV crudo por año/tour en data/raw/TENNIS_ATP/ y
data/raw/TENNIS_WTA/ -- mismo patron de "un archivo por temporada" que
football_data_loader.py, para poder re-procesar sin volver a pegarle al
sitio. NOTA: la ruta de RAW_DATA_DIR se infiere como
PROCESSED_DATA_DIR.parent / "raw" (consistente con la estructura
data/raw/, data/processed/ ya documentada en el roadmap) porque
config.settings no tiene confirmada una constante RAW_DATA_DIR explicita
-- si el nombre real de la carpeta es otro, ajustar RAW_DATA_DIR abajo.

Uso:
    python -m src.ingestion.tennis_data_loader --probe --year 2025 --tour ATP
    python -m src.ingestion.tennis_data_loader --download-all --start-year 2015 --end-year 2026
    python -m src.ingestion.tennis_data_loader --download-all --tours ATP --start-year 2000 --end-year 2026
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

BASE_URL = "http://www.tennis-data.co.uk"

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"  # ver nota en el docstring -- inferido, no confirmado por settings.py

MAX_RETRIES = 3
BACKOFF_SECONDS = 2


def _atp_xlsx_url(year: int) -> str:
    return f"{BASE_URL}/{year}/{year}.xlsx"


def _wta_xlsx_url(year: int) -> str:
    return f"{BASE_URL}/{year}w/{year}.xlsx"


def _atp_csv_url_guess(year: int) -> str:
    # NO CONFIRMADO -- probe() ya lo intento para 2025 y NO funciono (HTTP 300).
    # Se deja solo para referencia/diagnostico, no se usa en download_all().
    return f"{BASE_URL}/{year}/{year}.csv"


def _fetch_season_xlsx(url: str) -> pd.DataFrame:
    """Descarga un XLSX con reintentos y backoff -- mismo patron de resiliencia
    que football_data_loader.py, adaptado a pandas.read_excel en vez de read_csv."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return pd.read_excel(url)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)
    raise last_error


def probe(year: int, tour: str = "ATP") -> None:
    """
    Descarga UN archivo (un año, un tour) y muestra el esquema real:
    columnas, dtypes, shape, primeras filas. No asume que notes.txt
    describe el archivo real con exactitud -- lo confirma.
    """
    xlsx_url = _atp_xlsx_url(year) if tour.upper() == "ATP" else _wta_xlsx_url(year)
    csv_url = _atp_csv_url_guess(year)

    print(f"=== Probe tennis-data.co.uk -- tour={tour}, year={year} ===\n")

    print(f"[1/2] Intentando XLSX (URL confirmada via alldata.php): {xlsx_url}")
    df = None
    try:
        df = pd.read_excel(xlsx_url)
        print(f"  OK -- {len(df)} filas, {len(df.columns)} columnas.")
    except Exception as e:
        print(f"  [FALLO] {type(e).__name__}: {e}")

    if df is not None:
        print(f"\nColumnas reales encontradas ({len(df.columns)}):")
        print(f"  {list(df.columns)}")
        print(f"\nDtypes:")
        print(df.dtypes.to_string())
        print(f"\nPrimeras 3 filas:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.head(3).to_string())

        print("\n--- Chequeos puntuales (no asumir, confirmar) ---")
        for col in ["PSW", "PSL", "B365W", "B365L", "MaxW", "MaxL", "AvgW", "AvgL",
                    "Winner", "Loser", "Surface", "Round", "Best of", "WRank", "LRank"]:
            present = col in df.columns
            print(f"  {'[OK]' if present else '[FALTA]'} columna '{col}'")

        pinnacle_cols = [c for c in df.columns if "PS" in c or "Pinnacle" in c]
        print(f"\n  Columnas que podrian ser de Pinnacle (buscando 'PS'/'Pinnacle'): {pinnacle_cols}")

    print(f"\n[2/2] Intentando CSV (URL NO confirmada, solo un guess por patron de sitio hermano): {csv_url}")
    try:
        resp = requests.get(csv_url, timeout=15)
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("text"):
            print(f"  OK -- HTTP 200, Content-Type={resp.headers.get('Content-Type')}. "
                  f"El CSV SI existe en esa URL.")
        else:
            print(f"  [NO] HTTP {resp.status_code} -- usar XLSX (ya confirmado arriba).")
    except Exception as e:
        print(f"  [FALLO] {type(e).__name__}: {e} -- usar XLSX (ya confirmado arriba).")


def download_all(start_year: int, end_year: int, tours: list) -> None:
    """
    Descarga un XLSX por año/tour en el rango [start_year, end_year], lo
    guarda como CSV crudo en data/raw/TENNIS_<TOUR>/<year>.csv. Trata un
    año sin archivo publicado (formato futuro, o WTA antes de 2007) como
    'no disponible', no como un crash -- mismo criterio que el fix de
    HTTP 300 en football_data_loader.py.
    """
    for tour in tours:
        tour = tour.upper()
        out_dir = RAW_DATA_DIR / f"TENNIS_{tour}"
        out_dir.mkdir(parents=True, exist_ok=True)
        url_fn = _atp_xlsx_url if tour == "ATP" else _wta_xlsx_url

        print(f"\n=== Descargando {tour}, {start_year}-{end_year} ===")
        n_ok, n_skip = 0, 0
        for year in range(start_year, end_year + 1):
            url = url_fn(year)
            try:
                df = _fetch_season_xlsx(url)
            except Exception as e:
                print(f"  [SKIP] {year}: no disponible ({type(e).__name__}) -- {url}")
                n_skip += 1
                continue

            df["tour"] = tour
            df["source_year"] = year
            out_path = out_dir / f"{year}.csv"
            df.to_csv(out_path, index=False)
            print(f"  [OK] {year}: {len(df)} partidos -> {out_path}")
            n_ok += 1

        print(f"  Resumen {tour}: {n_ok} años descargados, {n_skip} no disponibles.")

    print("\nGuardado en data/raw/TENNIS_ATP/ y data/raw/TENNIS_WTA/ -- un CSV por año, sin "
          "procesar todavia. Siguiente paso (no hecho aca): un clean_data.py equivalente para "
          "tenis que consolide todos los años, calcule probabilidades implicitas sin margen a "
          "partir de PSW/PSL (Pinnacle), y normalice al formato que el resto del pipeline espera.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de tennis-data.co.uk -- probe de esquema y descarga de historia cruda.")
    parser.add_argument("--probe", action="store_true", help="Descarga y muestra el esquema real de un archivo.")
    parser.add_argument("--year", type=int, default=2025, help="Año a probar con --probe (default: 2025).")
    parser.add_argument("--tour", type=str, default="ATP", choices=["ATP", "WTA"], help="Tour a probar con --probe (default: ATP).")
    parser.add_argument("--download-all", action="store_true", help="Descarga historia cruda para el rango de años dado.")
    parser.add_argument("--start-year", type=int, default=2015, help="Primer año a descargar con --download-all (default: 2015).")
    parser.add_argument("--end-year", type=int, default=2026, help="Ultimo año a descargar con --download-all (default: 2026).")
    parser.add_argument("--tours", type=str, default="ATP,WTA", help="Tours a descargar, separados por coma (default: ATP,WTA).")
    args = parser.parse_args()

    if args.probe:
        probe(args.year, args.tour)
    elif args.download_all:
        download_all(args.start_year, args.end_year, args.tours.split(","))
    else:
        parser.print_help()