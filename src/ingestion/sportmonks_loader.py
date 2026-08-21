"""
Decision de datos: Sportmonks (roadmap, 2026-08-18) -- GO acotado a
Serie A/Bundesliga. Este modulo es el andamiaje de ingesta, preparado
ANTES de activar el trial de 14 dias para no perder tiempo del trial en
plomeria basica -- mismo principio de separacion de modulos que
mls_loader.py (no forzar Sportmonks dentro de football_data_loader.py).

IMPORTANTE, leer antes de usar: los nombres exactos de endpoints, parametros
y campos de respuesta de la API de Sportmonks (v3, api.sportmonks.com) NO
estan confirmados contra una respuesta real en este entorno -- se escribieron
siguiendo la convencion publica documentada de Sportmonks Football API v3
(fixtures con includes de statistics/xg, filtrado por league_id/season_id),
pero pueden no ser exactos. Siguiendo la misma disciplina que el resto del
proyecto (nunca asumir un nombre de columna/endpoint sin confirmarlo,
ver 'Incidentes de integridad de codigo' en el roadmap): la PRIMERA accion
al tener la API key real del trial es correr `probe()`, que hace una sola
llamada de bajo costo y imprime la estructura real de la respuesta (claves
disponibles, no el payload entero) para confirmar o corregir los nombres de
campo antes de construir el resto del pipeline sobre supuestos.

No corre nada todavia -- no hay API key configurada. Este archivo se deja
listo para que, apenas el usuario tenga la cuenta del trial, sea cuestion
de pegar el token y correr `probe()`.

Uso previsto una vez confirmada la API:
    export SPORTMONKS_API_TOKEN="..."
    python -m src.ingestion.sportmonks_loader --probe
    python -m src.ingestion.sportmonks_loader --league SERIEA --season 2024
"""
import os
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

BASE_URL = "https://api.sportmonks.com/v3/football"
# 2026-08-19: /types devolvio 404 bajo BASE_URL (football) -- en Sportmonks
# v3 los tipos son un recurso compartido entre deportes, probablemente bajo
# /v3/core en vez de /v3/football. Se prueba aca, sin cambiar BASE_URL (que
# sigue siendo correcto para /leagues y /fixtures, ya confirmados).
CORE_BASE_URL = "https://api.sportmonks.com/v3/core"

# CONFIRMADO 2026-08-19 via list_all_leagues() contra la API real (un unico
# match cada uno, short_code 'ITA SA' / 'GER BI' confirma pais) -- no son
# un supuesto, son el resultado real de /leagues.
SPORTMONKS_LEAGUE_IDS = {
    "SERIEA": 384,       # 'Serie A', country_id=251 (Italia), short_code='ITA SA'
    "BUNDESLIGA": 82,    # 'Bundesliga', country_id=11 (Alemania), short_code='GER BI'
}

MAX_RETRIES = 3
BACKOFF_SECONDS = 2


def _get_api_token() -> str:
    token = os.environ.get("SPORTMONKS_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Falta la variable de entorno SPORTMONKS_API_TOKEN. "
            "Conseguila activando el trial de 14 dias en sportmonks.com y "
            "seteala antes de correr este script (no se hardcodea la key en el repo)."
        )
    return token


def _request(endpoint: str, params: dict = None, base_url: str = None) -> dict:
    """GET generico contra la API de Sportmonks, con reintentos y backoff --
    mismo patron que football_data_loader.py para la ingesta europea, para
    no reinventar la disciplina de manejo de errores del proyecto.

    base_url: override opcional (default BASE_URL, /v3/football) -- usado
    para endpoints confirmados que viven bajo otra base, ej. /v3/core."""
    params = dict(params or {})
    params["api_token"] = _get_api_token()
    url = f"{base_url or BASE_URL}{endpoint}"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            last_error = e
            print(f"  [reintento {attempt}/{MAX_RETRIES}] error de red en {endpoint}: {e}")
            time.sleep(BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code == 429:
            # Rate limit -- Sportmonks devuelve el tiempo de espera en headers
            # en algunos planes; si no esta disponible, backoff fijo.
            wait = int(resp.headers.get("Retry-After", BACKOFF_SECONDS * attempt))
            print(f"  [rate limit] {endpoint} -- esperando {wait}s antes de reintentar")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"  [ERROR] {endpoint} devolvio HTTP {resp.status_code}: {resp.text[:300]}")
            last_error = RuntimeError(f"HTTP {resp.status_code}")
            time.sleep(BACKOFF_SECONDS * attempt)
            continue

        return resp.json()

    raise RuntimeError(f"No se pudo completar la llamada a {endpoint} tras {MAX_RETRIES} intentos: {last_error}")


def probe() -> None:
    """Primera llamada real a hacer con la API key del trial. NO asume nada
    sobre la estructura de la respuesta -- solo la imprime, para confirmar
    (o corregir) los supuestos de este archivo antes de construir el resto
    del pipeline. Barato: un solo endpoint, sin paginar todo."""
    print("=== Sportmonks -- probe de conectividad y estructura de respuesta ===")
    try:
        data = _request("/leagues", params={"per_page": 5})
    except Exception as e:
        print(f"[ERROR] La llamada de prueba fallo: {e}")
        print("Revisar: token valido, trial activo, plan incluye el endpoint /leagues.")
        return

    print("Llamada exitosa. Claves de nivel superior en la respuesta:", list(data.keys()))
    items = data.get("data", [])
    if items:
        print(f"\nPrimer item de 'data' (para confirmar nombres de campo reales):")
        print(items[0])
    else:
        print("\n[AVISO] La respuesta no trajo 'data' o vino vacia -- revisar la respuesta cruda:")
        print(data)

    print("\nSiguiente paso manual: buscar en la lista completa de /leagues los IDs de "
          "'Serie A' (Italia) y 'Bundesliga' (Alemania), y completar SPORTMONKS_LEAGUE_IDS "
          "arriba en este archivo con los IDs reales antes de construir fetch_fixtures_with_xg().")


def list_all_leagues(keyword: str = None) -> None:
    """Paso 2 del probe: encontrar los IDs reales de Serie A/Bundesliga sin
    adivinar un endpoint de busqueda (no confirmado). Recorre /leagues
    paginando con 'page'/'per_page' -- parametros REST estandar, no
    confirmados contra la documentacion real de Sportmonks, pero de bajo
    riesgo: si 'page' no fuera el nombre correcto, la respuesta de la
    pagina 2 en adelante seria identica a la de la pagina 1, visible
    comparando el primer item de cada pagina si hiciera falta debuggear.

    La condicion de corte NO depende de ninguna clave de 'pagination' (no
    confirmada) -- se corta cuando la pagina trae menos items que
    'per_page' (señal de ultima pagina que no requiere asumir nombres de
    campo), con un tope de seguridad de 20 paginas.

    keyword: si se pasa, filtra por nombre (case-insensitive, substring) --
    ej. 'Serie A' o 'Bundesliga'. Si es None, imprime TODAS las ligas
    (puede ser una lista larga)."""
    page = 1
    per_page = 50
    all_leagues = []
    while True:
        data = _request("/leagues", params={"per_page": per_page, "page": page})
        if page == 1:
            print("Estructura real de 'pagination' (primera pagina, de referencia):", data.get("pagination"))
        items = data.get("data", [])
        for item in items:
            name = item.get("name", "")
            if keyword is None or keyword.lower() in name.lower():
                all_leagues.append({
                    "id": item.get("id"), "name": name,
                    "country_id": item.get("country_id"),
                    "short_code": item.get("short_code"),
                })
        if len(items) < per_page or page >= 20:
            break
        page += 1

    label = f" que contienen '{keyword}'" if keyword else ""
    print(f"\n{len(all_leagues)} liga(s) encontrada(s){label}:")
    for lg in all_leagues:
        print(f"  id={lg['id']:<6} country_id={str(lg['country_id']):<8} "
              f"short_code={str(lg['short_code']):<8} name={lg['name']}")
    print("\nSiguiente paso manual: identificar cual id corresponde a Italia (Serie A) y "
          "cual a Alemania (Bundesliga) -- usar 'short_code' como pista (ej. 'UK PL' en el "
          "ejemplo de Premier League de probe() sugiere que el codigo de pais esta ahi "
          "embebido) y completar SPORTMONKS_LEAGUE_IDS arriba en este archivo con los IDs reales.")


def probe_fixture(include: str = "statistics") -> None:
    """Paso 3 del probe, ahora que los IDs de liga estan confirmados: antes
    de escribir fetch_fixtures_with_xg() hace falta ver la estructura REAL
    de un fixture, sobre todo si el include usado trae de verdad datos de
    xG y bajo que nombre de campo -- no se adivina, se mide.

    'include' es el parametro estandar de Sportmonks v3 para pedir datos
    relacionados (ej. 'statistics', 'participants', 'scores') -- el nombre
    exacto del include que trae xG especificamente NO esta confirmado
    todavia. Se corre con el default 'statistics' primero; si no aparece
    nada parecido a xG en la respuesta, se vuelve a correr pasando otro
    valor de include (ej. --probe-fixture "statistics;participants") para
    iterar sin tener que adivinar en el codigo."""
    print(f"=== Sportmonks -- probe de un fixture, include='{include}' ===")
    try:
        data = _request("/fixtures", params={"per_page": 1, "include": include})
    except Exception as e:
        print(f"[ERROR] La llamada fallo: {e}")
        return

    items = data.get("data", [])
    if not items:
        print("[AVISO] La respuesta no trajo fixtures en 'data'. Respuesta cruda:")
        print(data)
        return

    print("Primer fixture completo (revisar si trae una clave de estadisticas por equipo "
          "y si alguno de esos campos es xG / expected goals):")
    print(items[0])
    print("\nSi no aparece nada de xG con este include: correr de nuevo con otro valor, ej.:")
    print('  python -m src.ingestion.sportmonks_loader --probe-fixture "statistics;participants"')
    print("No adivinar el nombre del campo de xG en el codigo hasta verlo aca.")


def probe_types(keyword: str = "expected") -> None:
    """El fixture de prueba (MLS, un partido al azar porque /fixtures sin
    filtro trajo el primero disponible) mostro 43 tipos de estadistica
    nombrados, ninguno de xG -- pero eso no prueba que el plan no incluya
    xG en general: puede ser que ese partido puntual (o esa liga) no lo
    tenga trackeado, aunque el tipo exista. Este endpoint consulta el
    CATALOGO COMPLETO de tipos de estadistica que Sportmonks define (no
    los de un partido en particular) y filtra por nombre -- responde de
    una vez si 'Expected Goals'/xG existe como concepto disponible.

    Si este endpoint no existe bajo la base /v3/football (podria estar
    bajo /v3/core en la API real, no confirmado), el error va a ser
    explicito y se ajusta BASE_URL para este llamado especifico, sin
    tocar el resto del modulo."""
    print(f"=== Sportmonks -- catalogo completo de tipos, buscando '{keyword}' ===")
    print(f"(usando base {CORE_BASE_URL} -- /v3/football/types devolvio 404 el 2026-08-19)")
    page = 1
    per_page = 50
    matches = []
    total_seen = 0
    while True:
        try:
            data = _request("/types", params={"per_page": per_page, "page": page}, base_url=CORE_BASE_URL)
        except Exception as e:
            print(f"[ERROR] /types fallo tambien bajo {CORE_BASE_URL}: {e}")
            print("El endpoint de tipos no esta confirmado bajo ninguna de las 2 bases probadas.")
            return
        items = data.get("data", [])
        total_seen += len(items)
        for item in items:
            name = str(item.get("name", ""))
            code = str(item.get("code", ""))
            if keyword.lower() in name.lower() or keyword.lower() in code.lower():
                matches.append(item)
        if len(items) < per_page or page >= 40:
            break
        page += 1

    print(f"Tipos totales revisados: {total_seen}. Matches para '{keyword}': {len(matches)}")
    for m in matches:
        print(f"  {m}")
    if not matches:
        print(f"\n[RESULTADO] Ningun tipo contiene '{keyword}' en {total_seen} tipos revisados.")


XG_TYPE_ID = 5304  # 'Expected Goals (xG)', confirmado 2026-08-19 via probe_types() contra /v3/core/types

def probe_league_fixture(league_key: str) -> None:
    """Ultimo chequeo antes de construir fetch_fixtures_with_xg() de verdad:
    que xG EXISTA en el catalogo (confirmado, type_id=5304) no prueba que
    este POBLADO para Serie A/Bundesliga en este plan -- el fixture de
    prueba anterior era de MLS y no lo tenia. Trae un fixture real de la
    liga pedida y revisa si type_id=5304 aparece con un valor real.

    El filtro por liga usa el parametro 'filters' con la sintaxis
    'fixtureLeagues:{id}' -- convencion documentada de Sportmonks v3, NO
    confirmada todavia en este entorno. Por eso esta funcion verifica
    explicitamente que el fixture devuelto sea de verdad de la liga
    pedida (comparando league_id) antes de sacar ninguna conclusion sobre
    xG -- si el filtro no funciono, lo dice, en vez de asumir que el
    partido que llego es el correcto."""
    league_id = SPORTMONKS_LEAGUE_IDS.get(league_key)
    if league_id is None:
        print(f"[ERROR] SPORTMONKS_LEAGUE_IDS['{league_key}'] no esta completado.")
        return

    print(f"=== Sportmonks -- probe de un fixture real de {league_key} (league_id={league_id}) ===")
    try:
        data = _request("/fixtures", params={
            "per_page": 1, "include": "statistics.type",
            "filters": f"fixtureLeagues:{league_id}",
        })
    except Exception as e:
        print(f"[ERROR] La llamada fallo: {e}")
        return

    items = data.get("data", [])
    if not items:
        print("[AVISO] La respuesta no trajo fixtures. Filtro 'filters=fixtureLeagues:ID' "
              "podria no ser el nombre/sintaxis correcta -- revisar respuesta cruda:")
        print(data)
        return

    fixture = items[0]
    got_league_id = fixture.get("league_id")
    print(f"Fixture devuelto: '{fixture.get('name')}' (league_id={got_league_id})")
    if got_league_id != league_id:
        print(f"[AVISO] El filtro NO funciono como se esperaba -- se pidio league_id={league_id} "
              f"y volvio league_id={got_league_id}. La sintaxis 'filters=fixtureLeagues:ID' no es "
              f"la correcta, hay que probar otra antes de confiar en este resultado.")
        return

    stats = fixture.get("statistics", [])
    xg_entries = [s for s in stats if s.get("type_id") == XG_TYPE_ID]
    if xg_entries:
        print(f"\n[CONFIRMADO] xG SI esta poblado para {league_key} en este plan. Entradas encontradas:")
        for e in xg_entries:
            print(f"  {e}")
    else:
        tipos_presentes = sorted(set(s.get("type_id") for s in stats))
        print(f"\n[RESULTADO] Este fixture de {league_key} NO trae type_id={XG_TYPE_ID} (xG) entre sus "
              f"{len(stats)} estadisticas. Type_ids presentes: {tipos_presentes}")
        print("Esto puede ser este partido puntual (revisar con otro --season/fixture mas reciente) "
              "o que el plan del trial no incluya xG poblado para esta liga, aunque exista en el "
              "catalogo general -- no concluir todavia con un solo partido.")


def check_xg_coverage(league_key: str, sample_size: int = 20) -> None:
    """Un solo partido sin xG no prueba nada (podia ser ese partido puntual).
    Esta funcion mide cobertura real sobre una muestra de partidos de la
    liga -- cuantos de los ultimos N fixtures devueltos por la API traen
    type_id=5304 poblado. Responde la pregunta de fondo: el plan del trial,
    ¿cubre xG para esta liga o no?"""
    league_id = SPORTMONKS_LEAGUE_IDS.get(league_key)
    if league_id is None:
        print(f"[ERROR] SPORTMONKS_LEAGUE_IDS['{league_key}'] no esta completado.")
        return

    print(f"=== Sportmonks -- cobertura de xG sobre {sample_size} fixtures de {league_key} (league_id={league_id}) ===")
    try:
        data = _request("/fixtures", params={
            "per_page": sample_size, "include": "statistics.type",
            "filters": f"fixtureLeagues:{league_id}",
        })
    except Exception as e:
        print(f"[ERROR] La llamada fallo: {e}")
        return

    items = data.get("data", [])
    if not items:
        print("[AVISO] La respuesta no trajo fixtures.")
        return

    con_xg, sin_xg = [], []
    for fx in items:
        stats = fx.get("statistics", [])
        tiene_xg = any(s.get("type_id") == XG_TYPE_ID for s in stats)
        label = f"{fx.get('name')} ({fx.get('starting_at')}, season_id={fx.get('season_id')})"
        (con_xg if tiene_xg else sin_xg).append(label)

    print(f"\nCon xG poblado: {len(con_xg)}/{len(items)}")
    for lbl in con_xg:
        print(f"  [SI] {lbl}")
    print(f"\nSin xG poblado: {len(sin_xg)}/{len(items)}")
    for lbl in sin_xg[:10]:
        print(f"  [NO] {lbl}")
    if len(sin_xg) > 10:
        print(f"  ... y {len(sin_xg) - 10} mas")

    pct = len(con_xg) / len(items) * 100
    print(f"\n[RESULTADO] Cobertura de xG en esta muestra de {league_key}: {pct:.1f}%")


def fetch_fixtures_with_xg(league_key: str, season_year: int) -> list:
    """Placeholder -- NO CONFIRMADO. Una vez que probe() confirme la estructura
    real, esta funcion se completa para traer fixtures con estadisticas de xG
    (include=statistics o el include real que exponga xG por equipo/partido),
    paginando con 'per_page'/'page' segun la respuesta real de la API.

    Diseño previsto (a ajustar tras probe()): devolver una lista de dicts con
    al menos Date, HomeTeam, AwayTeam, xG_home, xG_away, FTHG, FTAG, FTR --
    mismas columnas base que matches_clean.csv, para que el resto del
    pipeline (add_team_form_features.py, backtest_v4.py) pueda tratar xG
    como una fuente adicional sin reescribir la arquitectura, mismo principio
    de compatibilidad que ya se aplico al integrar La Liga/Serie A/Bundesliga
    en Fase 8."""
    league_id = SPORTMONKS_LEAGUE_IDS.get(league_key)
    if league_id is None:
        raise RuntimeError(
            f"SPORTMONKS_LEAGUE_IDS['{league_key}'] no esta completado todavia. "
            f"Correr probe() primero y completar el ID real."
        )
    raise NotImplementedError(
        "fetch_fixtures_with_xg() todavia no esta implementado -- placeholder a completar "
        "una vez confirmada la estructura real de respuesta via probe(). No adivinar el "
        "esquema de xG sin verlo primero, misma disciplina que el resto del proyecto."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de Sportmonks (Serie A/Bundesliga, xG)")
    parser.add_argument("--probe", action="store_true", help="Confirmar conectividad y estructura de respuesta")
    parser.add_argument("--find-leagues", type=str, default=None, metavar="KEYWORD",
                         help="Buscar ligas por nombre (ej. 'Serie A', 'Bundesliga') para encontrar sus IDs reales")
    parser.add_argument("--probe-fixture", type=str, nargs="?", const="statistics", default=None, metavar="INCLUDE",
                         help="Ver la estructura real de un fixture (default include='statistics')")
    parser.add_argument("--probe-types", type=str, nargs="?", const="expected", default=None, metavar="KEYWORD",
                         help="Buscar en el catalogo completo de tipos de estadistica (default 'expected')")
    parser.add_argument("--probe-league-fixture", choices=["SERIEA", "BUNDESLIGA"], default=None,
                         help="Ver si xG esta poblado en un fixture real de esta liga")
    parser.add_argument("--check-xg-coverage", choices=["SERIEA", "BUNDESLIGA"], default=None,
                         help="Medir cobertura de xG sobre una muestra de fixtures de esta liga")
    parser.add_argument("--league", choices=["SERIEA", "BUNDESLIGA"], help="Liga a descargar")
    parser.add_argument("--season", type=int, help="Año de temporada")
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.find_leagues is not None:
        list_all_leagues(args.find_leagues)
    elif args.probe_fixture is not None:
        probe_fixture(args.probe_fixture)
    elif args.probe_types is not None:
        probe_types(args.probe_types)
    elif args.probe_league_fixture is not None:
        probe_league_fixture(args.probe_league_fixture)
    elif args.check_xg_coverage is not None:
        check_xg_coverage(args.check_xg_coverage)
    elif args.league and args.season:
        fetch_fixtures_with_xg(args.league, args.season)
    else:
        print("Usar --probe primero (con SPORTMONKS_API_TOKEN seteado) para confirmar la API "
              "antes de intentar descargar datos reales.")