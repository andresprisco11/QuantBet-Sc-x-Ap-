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

# --- Ligas de EXPANSION para la via de arbitraje sharp-vs-blandas ---------
# Motivo (2026-08-22): la deteccion de valor contra casas blandas NO usa
# nuestro modelo, solo necesita que Pinnacle cotice el partido. Por lo tanto
# NO esta limitada a las 4 ligas con historico limpio -- puede correr sobre
# cualquier competencia que la API sirva.
#
# Esto ataca el cuello de botella real: el CLV necesita 100+ apuestas para
# ser informativo, y con 4 ligas se juntan ~16 por fin de semana (6 semanas).
# Ampliando la superficie, el veredicto llega en semanas en vez de meses.
#
# Ademas hay una razon de fondo, no solo de volumen: las casas blandas son
# MAS flojas cuanto menos liquida la competencia. El edge deberia ser mayor
# aca que en la Premier, no menor.
#
# SIN CONFIRMAR: estos sport_key son los documentados por el vendor pero
# este proyecto no los probo. Correr --list-soccer (llamada GRATIS) para ver
# cuales existen y estan activos de verdad antes de usarlos.
EXPANSION_KEYS = {
    "LIGUE1": "soccer_france_ligue_one",
    "EREDIVISIE": "soccer_netherlands_eredivisie",
    "PRIMEIRA": "soccer_portugal_primeira_liga",
    "CHAMPIONSHIP": "soccer_efl_champ",
    "BELGIUM": "soccer_belgium_first_div",
    "TURKEY": "soccer_turkey_super_league",
    "GREECE": "soccer_greece_super_league",
    "BRAZIL": "soccer_brazil_campeonato",
    "ARGENTINA": "soccer_argentina_primera_division",
    "MLS": "soccer_usa_mls",
    "LALIGA2": "soccer_spain_segunda_division",
    "BUNDESLIGA2": "soccer_germany_bundesliga2",
    "SERIEB": "soccer_italy_serie_b",
    "UCL": "soccer_uefa_champs_league",
    "MEXICO": "soccer_mexico_ligamx",
}

ALL_KEYS = {**SPORT_KEYS, **EXPANSION_KEYS}

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
# h2h = 1X2. 'totals' (mas/menos goles) se agrega como mercado opcional:
# multiplica la superficie de deteccion por partido sin costar una llamada
# extra por liga -- The Odds API cobra por region x mercado, asi que pedir
# h2h+totals juntos cuesta 2x, no 2 llamadas. Vale la pena: el cuello de
# botella del proyecto hoy es juntar muestra para el CLV.
MARKETS = "h2h"
MARKETS_CON_TOTALES = "h2h,totals"

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


def discover_active_soccer(excluir_femenino: bool = False) -> dict:
    """Devuelve {NOMBRE: sport_key} de TODAS las competencias de futbol
    ACTIVAS ahora mismo, preguntandoselo a la API en vez de mantener una
    lista a mano.

    Por que asi y no con EXPANSION_KEYS fija (decidido 2026-08-22): una
    lista hardcodeada envejece sola -- las competencias entran y salen de
    temporada, y una key que deja de estar activa se convierte en una
    llamada desperdiciada o en un error silencioso. Ademas evita el riesgo
    contrario: inventar un sport_key que no existe.

    La llamada a /v4/sports es GRATIS (no consume creditos), asi que
    descubrir en cada corrida no cuesta nada.

    CONFIRMADO 2026-08-22 con llamada real: la API sirve 67 competencias de
    futbol, 45 activas. NO incluye la Primera A de Colombia -- no hay ningun
    sport_key colombiano en la respuesta real (entre 'soccer_chile_
    campeonato' y 'soccer_conmebol_*' no hay nada). No se puede cubrir lo
    que la fuente no sirve."""
    resp = _get_with_retries(f"{BASE_URL}/sports", {"apiKey": _api_key()})
    activos = {}
    for s in resp.json():
        key = str(s.get("key", ""))
        if not key.startswith("soccer_") or not s.get("active"):
            continue
        if excluir_femenino and ("women" in key or "womens" in key):
            continue
        nombre = key.replace("soccer_", "").upper()
        activos[nombre] = key
    return activos


def list_soccer() -> None:
    """Llamada GRATIS (/v4/sports no consume creditos) que lista TODAS las
    competencias de futbol que la API sirve realmente, marcando cuales estan
    activas. Es el paso obligatorio antes de confiar en EXPANSION_KEYS --
    misma disciplina de siempre: no asumir un identificador, confirmarlo."""
    resp = _get_with_retries(f"{BASE_URL}/sports", {"apiKey": _api_key(), "all": "true"})
    data = resp.json()
    soccer = [s for s in data if str(s.get("key", "")).startswith("soccer_")]
    activos = [s for s in soccer if s.get("active")]

    print(f"Competencias de futbol que sirve la API: {len(soccer)} "
          f"({len(activos)} activas ahora mismo)\n")
    print(f"{'sport_key':<42}{'activa':<9}titulo")
    print("-" * 92)
    for s in sorted(soccer, key=lambda x: (not x.get("active"), x["key"])):
        print(f"{s['key']:<42}{'SI' if s.get('active') else 'no':<9}{s.get('title','')}")

    reales = {s["key"] for s in soccer}
    activas = {s["key"] for s in activos}
    print(f"\n--- Chequeo de las que este proyecto tiene configuradas ---")
    for nombre, key in ALL_KEYS.items():
        if key in activas:
            estado = "OK, activa"
        elif key in reales:
            estado = "existe pero INACTIVA (fuera de temporada)"
        else:
            estado = "*** NO EXISTE -- corregir el sport_key ***"
        print(f"  {nombre:<15} {key:<42} {estado}")


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
    sport_key = ALL_KEYS.get(league_key)
    if sport_key is None:
        print(f"[ERROR] '{league_key}' no esta en ALL_KEYS: {list(ALL_KEYS)}")
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


def fetch_upcoming_odds(league_key: str, regions: str = FETCH_REGIONS,
                        markets: str = MARKETS) -> pd.DataFrame:
    """Trae las cuotas ACTUALES de todos los partidos programados de esta
    liga (proximos dias, lo que la API tenga cargado) y las normaliza a un
    DataFrame long-format: una fila por (partido, bookmaker, resultado).

    NO reemplaza a matchday_experiment.py todavia -- ese cableado (leer este
    DataFrame en vez de la lista MATCHES hardcodeada) es el siguiente paso,
    una vez confirmado con --probe que esto trae datos reales y utiles."""
    # Acepta un nombre configurado O un sport_key directo -- necesario para
    # que funcione con discover_active_soccer(), que devuelve competencias
    # que no estan en ALL_KEYS.
    sport_key = ALL_KEYS.get(league_key, league_key if league_key.startswith("soccer_") else None)
    if sport_key is None:
        raise ValueError(f"'{league_key}' no esta en ALL_KEYS ni parece un sport_key: {list(ALL_KEYS)}")

    resp = _get_with_retries(
        f"{BASE_URL}/sports/{sport_key}/odds",
        {
            "apiKey": _api_key(),
            "regions": regions,
            "markets": markets,
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
                        # 'point' es la LINEA del mercado de totales (2.5, 3.0...).
                        # Es None para h2h. Sin esta columna no se pueden comparar
                        # totales entre casas: un Over 2.5 y un Over 3.0 son
                        # apuestas DISTINTAS y compararlas daria un edge falso.
                        "outcome_point": outcome.get("point"),
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
    parser.add_argument("--probe", choices=list(ALL_KEYS), default=None)
    parser.add_argument("--probe-all", action="store_true")
    parser.add_argument("--check-credits", action="store_true")
    parser.add_argument("--fetch", choices=list(ALL_KEYS), default=None)
    parser.add_argument("--list-soccer", action="store_true",
                        help="Listar TODAS las competencias de futbol reales de la API (GRATIS).")
    parser.add_argument("--fetch-all", action="store_true")
    args = parser.parse_args()

    if args.list_soccer:
        list_soccer()
    elif args.probe:
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
