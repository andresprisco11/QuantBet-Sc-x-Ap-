"""
Fase 11 (futbol) -- ingesta de cuotas EN VIVO/futuras para reemplazar el
pegado a mano de capturas de pantalla en matchday_experiment.py (ver
roadmap, "Decision estrategica -- feed en vivo", 2026-08-22).

**Por que este archivo, no reusar theoddsapi_historical_loader.py tal
cual**: ese script pega contra /v4/historical/sports/{sport}/odds (10
creditos x region x market por snapshot) -- pensado para reconstruir el
PASADO dia por dia. Para partidos que todavia no se jugaron hace falta el
endpoint de cuotas ACTUALES, mas barato (1 credito x region x market por
llamada) y que ademas devuelve TODOS los partidos programados de la liga
en una sola llamada, sin necesidad de fecha exacta.

**Pregunta real sin confirmar, y por que no se asume la respuesta**:
theoddsapi_loader.py (el probe existente) SOLO confirmo/descarto Pinnacle
para basketball_nba -- la duda de si The Odds API trae Pinnacle para
FUTBOL (soccer) especificamente sigue abierta. Es una pregunta distinta:
Pinnacle es mucho mas fuerte en soccer que en NBA, region='eu' podria
traerlo aca aunque no lo haya traido para NBA. No se asume nada -- por
eso este archivo separa probe() (barato, confirma sport_key real +
lista real de bookmakers) de fetch_upcoming_odds() (el fetch de verdad),
mismo principio que el resto del proyecto.

**sport_key de las 4 ligas -- SIN CONFIRMAR todavia**: son los valores
estandar documentados por el vendor (The Odds API, /v4/sports), pero
ESTE proyecto nunca los probo con una llamada real para soccer. Correr
--probe antes de asumir que existen o estan activos.

Requiere: THEODDSAPI_KEY como variable de entorno (la misma key que ya
usa theoddsapi_historical_loader.py para NBA -- no hace falta una nueva).

Uso:
    python -m src.ingestion.theoddsapi_live_odds_loader --probe EPL
    python -m src.ingestion.theoddsapi_live_odds_loader --probe-all
    python -m src.ingestion.theoddsapi_live_odds_loader --check-credits
    python -m src.ingestion.theoddsapi_live_odds_loader --fetch EPL
    python -m src.ingestion.theoddsapi_live_odds_loader --fetch-all
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_DATA_DIR, LEAGUES

BASE_URL = "https://api.the-odds-api.com/v4"

# SIN CONFIRMAR -- valores estandar documentados por The Odds API para las
# 4 ligas que ya sigue el proyecto (ver config/settings.py LEAGUES). Correr
# --probe-all antes de confiar en que estos 4 identificadores son reales y
# estan activos (mismo criterio que SPORT_KEYS en
# theoddsapi_historical_loader.py -- 'NBA' fue el unico confirmado ahi).
SPORT_KEYS = {
    "EPL": "soccer_epl",
    "LALIGA": "soccer_spain_la_liga",
    "SERIEA": "soccer_italy_serie_a",
    "BUNDESLIGA": "soccer_germany_bundesliga",
}

# Pinnacle suele reportarse bajo la region 'eu' en The Odds API. Se piden
# varias regiones a la vez en el probe (barato, sigue siendo 1 llamada) para
# no tener que adivinar cual trae Pinnacle para soccer -- se ve en la
# respuesta real cual bookmaker aparece bajo que region.
PROBE_REGIONS = "eu,uk,us"
# CONFIRMADO 2026-08-22 (ver roadmap): Pinnacle SI aparece en las 4 ligas
# pidiendo 'eu,uk,us' juntas -- esa llamada combinada no dice bajo cual
# region especifica cayo Pinnacle, asi que se mantienen las 3 tambien en el
# fetch real (no solo 'eu') para no arriesgar perderlo por asumir de mas.
# Costo extra trivial (3 creditos x liga en vez de 1, con ~8800 restantes
# de 20,000/mes) frente al riesgo de silenciosamente dejar de traer Pinnacle.
FETCH_REGIONS = "eu,uk,us"
MARKETS = "h2h"  # 1X2 -- lo unico que necesita matchday_experiment.py / economic_backtest.py hoy

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 3.0


def _api_key() -> str:
    key = os.environ.get("THEODDSAPI_KEY")
    if not key:
        raise EnvironmentError(
            "Falta la variable de entorno THEODDSAPI_KEY -- setearla antes de correr esto. "
            "Es la misma key que ya usa theoddsapi_historical_loader.py para NBA, NUNCA hardcodear "
            "la key en este archivo ni commitearla."
        )
    return key


def _get_with_retries(url: str, params: dict) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = BACKOFF_BASE_SECONDS * attempt
                print(f"  [429] rate limit -- esperando {wait:.1f}s (intento {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            wait = BACKOFF_BASE_SECONDS * attempt
            print(f"  [ERROR] {e} -- reintentando en {wait:.1f}s (intento {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
    raise RuntimeError(f"Se agotaron los reintentos contra {url}") from last_exc


def check_credits() -> None:
    """Llamada gratis (no consume creditos) -- mismo patron que
    theoddsapi_historical_loader.py, para chequear presupuesto sin gastar."""
    resp = _get_with_retries(f"{BASE_URL}/sports", {"apiKey": _api_key()})
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")
    print(f"Creditos usados hasta ahora: {used}")
    print(f"Creditos restantes: {remaining}")


def probe(league_key: str) -> None:
    """Confirma con una llamada real si el sport_key de esta liga existe, si
    hay partidos programados, y sobre todo -- la pregunta abierta de este
    archivo -- que bookmakers reales trae la API para soccer, y si Pinnacle
    esta entre ellos. Imprime todo tal cual viene, sin interpretar."""
    sport_key = SPORT_KEYS.get(league_key)
    if sport_key is None:
        print(f"[ERROR] '{league_key}' no esta en SPORT_KEYS: {list(SPORT_KEYS)}")
        return

    print(f"=== Probe The Odds API -- {league_key} (sport_key='{sport_key}') ===")
    resp = _get_with_retries(
        f"{BASE_URL}/sports/{sport_key}/odds",
        {
            "apiKey": _api_key(),
            "regions": PROBE_REGIONS,
            "markets": MARKETS,
            "oddsFormat": "decimal",
        },
    )
    data = resp.json()
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")

    if not data:
        print(f"[AVISO] Respuesta vacia -- puede ser que '{sport_key}' no exista, este inactivo, "
              f"o que no haya partidos programados en este momento para {league_key}. Revisar "
              f"tambien https://the-odds-api.com/sports-odds-data/soccer-odds.html para el "
              f"sport_key real vigente si esto persiste.")
        print(f"Creditos usados: {used} | restantes: {remaining}")
        return

    print(f"Partidos devueltos: {len(data)}")
    bookmakers_por_region = {}
    for event in data:
        for bm in event.get("bookmakers", []):
            bookmakers_por_region.setdefault(bm.get("key"), 0)
            bookmakers_por_region[bm.get("key")] += 1

    print("\n=== Bookmakers reales encontrados (y en cuantos partidos aparecen) ===")
    for bm, n in sorted(bookmakers_por_region.items(), key=lambda x: -x[1]):
        print(f"  - {bm}: {n}/{len(data)} partidos")

    has_pinnacle = "pinnacle" in bookmakers_por_region
    print(f"\n[RESULTADO REAL] Pinnacle {'SI' if has_pinnacle else 'NO'} esta disponible para {league_key} "
          f"en las regiones '{PROBE_REGIONS}'.")
    if not has_pinnacle:
        print("[AVISO] Si Pinnacle no aparece aca, el blend Benter Boost sigue sin poder ejecutarse "
              "contra el libro con el que fue calibrado -- habria que decidir entre (a) usar el "
              "consenso multi-libro como proxy (mismo enfoque que theoddsapi_historical_loader.py "
              "adopto para NBA cuando Pinnacle no aparecio), o (b) buscar otra fuente para Pinnacle "
              "especificamente en soccer.")

    print(f"\nPrimer partido de muestra (para confirmar nombres de campo reales):")
    print(data[0])

    print(f"\nCreditos usados hasta ahora: {used} | restantes: {remaining}")


def probe_all() -> None:
    """Corre probe() sobre las 4 ligas del proyecto en una sola pasada --
    barato (4 llamadas de current odds, no historicas), confirma de una vez
    sport_key + cobertura de Pinnacle para las 4 antes de construir nada
    encima a ciegas."""
    for league_key in LEAGUES:
        probe(league_key)
        print("\n" + "=" * 70 + "\n")


def fetch_upcoming_odds(league_key: str, regions: str = FETCH_REGIONS) -> pd.DataFrame:
    """Trae las cuotas ACTUALES de todos los partidos programados de esta
    liga (proximos dias, lo que la API tenga cargado) y las normaliza a un
    DataFrame long-format: una fila por (partido, bookmaker, resultado).

    NO reemplaza a matchday_experiment.py todavia -- ese cableado (leer este
    DataFrame en vez de la lista MATCHES hardcodeada) es el siguiente paso,
    una vez confirmado con --probe que esto trae datos reales y utiles."""
    sport_key = SPORT_KEYS.get(league_key)
    if sport_key is None:
        raise ValueError(f"'{league_key}' no esta en SPORT_KEYS: {list(SPORT_KEYS)}")

    resp = _get_with_retries(
        f"{BASE_URL}/sports/{sport_key}/odds",
        {
            "apiKey": _api_key(),
            "regions": regions,
            "markets": MARKETS,
            "oddsFormat": "decimal",
        },
    )
    data = resp.json()
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")

    rows = []
    for event in data:
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "league": league_key,
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "bookmaker": bm.get("key"),
                        "bookmaker_last_update": bm.get("last_update"),
                        "market": market.get("key"),
                        "outcome_name": outcome.get("name"),
                        "outcome_price_decimal": outcome.get("price"),
                    })

    df = pd.DataFrame(rows)
    print(f"[{league_key}] {len(data)} partidos, {len(df)} filas de cuotas "
          f"(usados={used}, restantes={remaining})")
    return df


def fetch_all_and_save() -> None:
    """Descarga las 4 ligas y guarda un CSV por corrida, con fecha en el
    nombre -- mismo criterio de trazabilidad que el resto de data/raw/
    (nunca sobreescribir la corrida anterior en el mismo archivo)."""
    import datetime
    today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = RAW_DATA_DIR / "THEODDSAPI"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for league_key in LEAGUES:
        try:
            df = fetch_upcoming_odds(league_key)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"[ERROR] {league_key}: {e}")

    if not all_dfs:
        print("[AVISO] No se descargo nada -- ver errores arriba.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    out_path = out_dir / f"football_live_odds_{today_str}.csv"
    full_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path} ({len(full_df)} filas totales)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=list(SPORT_KEYS), default=None)
    parser.add_argument("--probe-all", action="store_true")
    parser.add_argument("--check-credits", action="store_true")
    parser.add_argument("--fetch", choices=list(SPORT_KEYS), default=None)
    parser.add_argument("--fetch-all", action="store_true")
    args = parser.parse_args()

    if args.probe:
        probe(args.probe)
    elif args.probe_all:
        probe_all()
    elif args.check_credits:
        check_credits()
    elif args.fetch:
        df = fetch_upcoming_odds(args.fetch)
        print(df.head(20).to_string())
    elif args.fetch_all:
        fetch_all_and_save()
    else:
        parser.print_help()
