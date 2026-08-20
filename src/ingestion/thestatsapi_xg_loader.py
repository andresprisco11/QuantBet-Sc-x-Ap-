"""
Fase 8 (retomada 2026-08-20, noche) -- ingesta real de xG via TheStatsAPI,
para desbloquear la hipotesis (c) de Serie A/Bundesliga (PAUSADA desde
2026-08-19 por falta de cobertura historica de Sportmonks -- ver roadmap).

Cada endpoint usado aca fue confirmado contra una respuesta REAL del API
Tester del propio usuario antes de escribir una linea de este script --
mismo criterio de disciplina de todo el proyecto (nunca asumir un path o
un esquema).

Confirmado con datos reales (2026-08-20):
- Auth: `Authorization: Bearer <key>` (headers en _headers(), formato
  confirmado por el propio "Code Snippets" del API Tester).
- Listado de partidos: GET /football/matches?competition_id=X&season_id=Y
  -- trae id, competition_id, season_id, utc_date, home_team{id,name},
  away_team{id,name}, score{home,away}, entre otros. Paginado (meta.page/
  total_pages).
- Stats por partido, incluye xG: GET /football/matches/{match_id}/stats --
  trae overview.expected_goals.all.{home,away} (xG estandar) y
  np_expected_goals.all.{home,away} (xG sin penales, no ofrecida por
  Sportmonks) mas shots/posesion/big_chances -- se guardan todos, costo
  marginal cero en el mismo request.
- `competition_id` de las 4 ligas del proyecto, confirmado via ID Finder /
  "List Data Coverage Per Competition" (2026-08-20): ver COMPETITION_IDS.
- `season_id` de las 4 temporadas con xG real confirmado (2022/23 a
  2025/26), confirmado via "Per-Season Data Coverage For One Competition"
  para las 4 ligas (2026-08-20) -- ver SEASON_IDS. Antes de 2022/23, xG
  esta confirmado en 0% en las 4 ligas (temporadas completas, con
  fixtures/odds/lineups/stats en 100% y xG especificamente en 0) -- NO se
  piden esas temporadas, seria gastar requests del plan para nada.

Alcance de esta corrida: TODAS las 4 ligas (no solo Serie A/Bundesliga) --
mas barato tenerlas todas ya que el dato esta confirmado en las 4, aunque
la pregunta que motiva esto (hipotesis (c)) es especificamente sobre
Serie A/Bundesliga. Tener EPL/La Liga con xG tambien sirve como grupo de
control: si xG tampoco mejora el modelo ahi (donde v4 ya funciona bien
sin xG), refuerza que la mejora en Serie A/Bundesliga (si aparece) es
real y no un artefacto de agregar cualquier feature nueva.

Rate limit del plan Starter: 120 req/min. Este script usa un delay
conservador entre CADA request (listado y stats por igual) para no
acercarse al limite -- ver REQUEST_DELAY_SECONDS.

Volumen esperado: ~304-380 partidos/temporada x 4 temporadas x 4 ligas =
~5,600 partidos = ~5,600 requests de stats + ~60-80 requests de listado
paginado = ~5,700 requests totales. Con REQUEST_DELAY_SECONDS=1.0 (bajado
de 0.6 tras un 429 real, ver mas abajo), la corrida completa (--all) tarda
aproximadamente 95-100 minutos -- mas que la estimacion original, es
normal, no es que el script este colgado. Una sola liga (ej. Bundesliga,
~1,230 partidos) tarda ~20-22 minutos, pero en la practica el rate limit
real puede estirarlo bastante mas (confirmado, ver corrida del
2026-08-20). Guarda progreso incrementalmente POR TEMPORADA (no al final
de toda la liga) y si se corta a mitad de camino, correr el mismo comando
de nuevo retoma sin repetir los partidos ya guardados -- no hace falta
borrar nada ni empezar de cero.

Requiere: THESTATSAPI_KEY como variable de entorno (nunca hardcodear la
key en el codigo ni commitearla).

Uso:
    python -m src.ingestion.thestatsapi_xg_loader --league EPL
    python -m src.ingestion.thestatsapi_xg_loader --all
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

# Mismo criterio que tennis_data_loader.py: config.settings no tiene una
# constante RAW_DATA_DIR confirmada -- se infiere. Si el nombre real de la
# carpeta en el proyecto es otro, ajustar aca.
RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"

BASE_URL = "https://api.thestatsapi.com/api"
REQUEST_DELAY_SECONDS = 1.0  # ~60 req/min -- bajado de 0.6s (100/min) tras un 429
                              # real (2026-08-20): el limite documentado de 120/min
                              # no dio margen suficiente en la practica (puede
                              # compartirse con uso del dashboard/API Tester en
                              # paralelo, o el limite real ser mas estricto que el
                              # documentado -- no se investiga la causa exacta, se
                              # baja el ritmo y se maneja el 429 explicitamente).
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 5.0  # subido de 2.0 -- un corte breve de red real (DNS,
                             # timeout) confirmado en la corrida anterior necesita
                             # mas margen que un backoff pensado para reintentos
                             # genericos chicos
RATE_LIMIT_BACKOFF_SECONDS = 15.0  # base para 429 -- mucho mas alto que el backoff
                                     # generico, porque la ventana de rate limit
                                     # necesita mas tiempo real para liberarse que
                                     # un error transitorio de red

# Confirmado con datos reales (ID Finder / "List Data Coverage Per
# Competition", 2026-08-20) -- NO son un supuesto.
COMPETITION_IDS = {
    "EPL": "comp_3039",
    "LALIGA": "comp_8814",
    "SERIEA": "comp_5840",
    "BUNDESLIGA": "comp_4643",
}

# Confirmado con datos reales ("Per-Season Data Coverage For One
# Competition", respuesta completa pegada por el usuario, 2026-08-20) --
# son los season_id REALES de las 4 temporadas con xG confirmado en cada
# liga, no un supuesto ni una construccion a partir del nombre.
SEASON_IDS = {
    "EPL": {
        "22/23": "sn_654318",
        "23/24": "sn_606923",
        "24/25": "sn_3057848",
        "25/26": "sn_6125938",
    },
    "LALIGA": {
        "22/23": "sn_709252",
        "23/24": "sn_606099",
        "24/25": "sn_5761468",
        "25/26": "sn_7246390",
    },
    "SERIEA": {
        "22/23": "sn_417582",
        "23/24": "sn_114559",
        "24/25": "sn_4591550",
        "25/26": "sn_3061436",
    },
    "BUNDESLIGA": {
        "22/23": "sn_869875",
        "23/24": "sn_882459",
        "24/25": "sn_6185114",
        "25/26": "sn_5789634",
    },
}

# Volumen esperado por temporada (para el chequeo de sanidad al final de
# cada liga) -- EPL/LaLiga/SerieA juegan con 20 equipos (380 partidos),
# Bundesliga con 18 (306-308). 25/26 esta en curso (temporada parcial),
# no se compara contra el total esperado.
EXPECTED_MATCHES_PER_SEASON = {
    "EPL": 380,
    "LALIGA": 380,
    "SERIEA": 380,
    "BUNDESLIGA": 308,
}


class MatchStatsNotFound(Exception):
    """404 especifico de /football/matches/{id}/stats -- confirmado con una
    corrida real (2026-08-20): NO es un error transitorio, es un partido sin
    stats cargadas en la API. Reintentarlo pierde tiempo para nada -- se
    propaga de una y el llamador lo guarda con NaN en vez de crashear toda
    la corrida por un solo partido."""
    pass


def _headers() -> dict:
    key = os.environ.get("THESTATSAPI_KEY")
    if not key:
        raise EnvironmentError(
            "Falta la variable de entorno THESTATSAPI_KEY -- confirmar que sigue "
            "configurada en tu maquina antes de correr esto."
        )
    return {"Authorization": f"Bearer {key}"}


ADAPTIVE_DELAY_CAP_SECONDS = 6.0  # techo -- si esto no alcanza, el problema no es de ritmo

# Estado global del ritmo REAL tolerado -- confirmado en la practica
# (2026-08-20) que el limite documentado de 120/min no se sostiene ni cerca
# (con 1s de delay, practicamente TODOS los requests pegaban 429). En vez de
# adivinar otro numero fijo, el delay base sube solo cuando pega un 429 y
# se relaja de a poco cuando no -- converge al ritmo real sin intervencion.
_adaptive_delay = REQUEST_DELAY_SECONDS


def _get_with_retries(url: str, params: dict = None) -> dict:
    """GET con reintentos + backoff, mismo patron de resiliencia que
    football_data_loader.py/tennis_data_loader.py/nfl_data_loader.py --
    con manejo EXPLICITO de 429 (rate limit), distinto de un error
    transitorio de red: espera lo que pida el header Retry-After si esta
    presente, o un backoff mucho mas largo que el generico si no. Ademas
    ajusta el ritmo base (_adaptive_delay) segun el trafico real."""
    global _adaptive_delay
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            if resp.status_code == 404:
                raise MatchStatsNotFound(url)
            if resp.status_code == 429:
                _adaptive_delay = min(_adaptive_delay * 1.4, ADAPTIVE_DELAY_CAP_SECONDS)
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SECONDS * attempt
                print(f"    [429 rate limit, intento {attempt}/{MAX_RETRIES}] esperando {wait:.0f}s "
                      f"(ritmo base subido a {_adaptive_delay:.1f}s/request)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            # Exito -- relajar de a poco el ritmo base, nunca por debajo del
            # piso configurado (REQUEST_DELAY_SECONDS).
            _adaptive_delay = max(REQUEST_DELAY_SECONDS, _adaptive_delay * 0.97)
            return resp.json()
        except MatchStatsNotFound:
            raise  # 404 real -- no es transitorio, no se reintenta, se propaga de una
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_SECONDS * attempt
                print(f"    [reintento {attempt}/{MAX_RETRIES}] {exc} -- esperando {wait:.0f}s")
                time.sleep(wait)
        finally:
            time.sleep(_adaptive_delay)
    raise RuntimeError(f"Fallaron los {MAX_RETRIES} reintentos contra {url}: {last_error}")


def _fetch_finished_matches(league_key: str, season_label: str, season_id: str) -> list:
    """Trae TODOS los partidos jugados (score.home no nulo) de una
    competition_id/season_id, paginando. Verifica explicitamente que el
    season_id devuelto coincide con el pedido -- mismo criterio que
    sportmonks_loader.py ("comparando league_id pedido vs. devuelto, no
    asumido"), nunca confiar en que el filtro del servidor funciono sin
    chequearlo."""
    competition_id = COMPETITION_IDS[league_key]
    matches = []
    page = 1
    while True:
        data = _get_with_retries(
            f"{BASE_URL}/football/matches",
            params={"competition_id": competition_id, "season_id": season_id, "per_page": 100, "page": page},
        )
        page_matches = data.get("data", [])
        for m in page_matches:
            if m.get("season_id") != season_id:
                # El filtro del servidor no hizo lo que se le pidio -- no se
                # descarta en silencio, se avisa fuerte para investigar.
                print(f"    [AVISO] partido {m.get('id')} con season_id={m.get('season_id')} "
                      f"distinto al pedido ({season_id}) -- se descarta de este lote.")
                continue
            if m.get("score", {}).get("home") is not None:
                matches.append(m)
        meta = data.get("meta", {})
        if page >= meta.get("total_pages", 1):
            break
        page += 1
    return matches


_EMPTY_STATS = {
    "home_xg": None, "away_xg": None, "home_npxg": None, "away_npxg": None,
    "home_big_chances": None, "away_big_chances": None,
    "home_total_shots": None, "away_total_shots": None,
    "home_possession": None, "away_possession": None,
}


def _fetch_match_stats(match_id: str) -> dict:
    """Trae /football/matches/{id}/stats y extrae los campos relevantes.
    Devuelve un dict con NaN en los campos que falten -- un partido con
    stats incompletas (o un 404 real, confirmado que existe en la practica)
    no debe crashear la corrida completa."""
    try:
        data = _get_with_retries(f"{BASE_URL}/football/matches/{match_id}/stats")
    except MatchStatsNotFound:
        print(f"    [SIN STATS] {match_id} -- 404 real, la API no tiene stats para este partido "
              f"(se guarda con NaN, no se reintenta mas)")
        return dict(_EMPTY_STATS)
    d = data.get("data", {})
    overview = d.get("overview", {})

    def _side(block: dict, side: str):
        # OJO: no alcanza con `(block or {}).get("all", {})` -- el default de
        # .get() solo aplica si la clave "all" FALTA, no si existe con valor
        # None (varios partidos traen "all": null explicito en campos como
        # big_chances, confirmado con el error real del usuario). Se fuerza
        # el fallback a {} en los dos niveles, no solo en el externo.
        all_block = (block or {}).get("all") or {}
        return all_block.get(side)

    return {
        "home_xg": _side(overview.get("expected_goals"), "home"),
        "away_xg": _side(overview.get("expected_goals"), "away"),
        "home_npxg": _side(d.get("np_expected_goals"), "home"),
        "away_npxg": _side(d.get("np_expected_goals"), "away"),
        "home_big_chances": _side(overview.get("big_chances"), "home"),
        "away_big_chances": _side(overview.get("big_chances"), "away"),
        "home_total_shots": _side(overview.get("total_shots"), "home"),
        "away_total_shots": _side(overview.get("total_shots"), "away"),
        "home_possession": _side(overview.get("ball_possession"), "home"),
        "away_possession": _side(overview.get("ball_possession"), "away"),
    }


def download_league(league_key: str) -> None:
    if league_key not in COMPETITION_IDS:
        raise ValueError(f"Liga desconocida: {league_key}. Opciones: {list(COMPETITION_IDS.keys())}")

    out_dir = RAW_DATA_DIR / "THESTATSAPI"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{league_key}_xg_raw.csv"

    # Retomar sin repetir -- confirmado necesario en la practica (2026-08-20,
    # un 404 real crasheo una corrida a mitad de la temporada 3 de 4, y antes
    # de este cambio NO habia nada guardado de las 2.5 temporadas ya hechas).
    done_ids = set()
    file_has_header = out_path.exists() and out_path.stat().st_size > 0
    if file_has_header:
        existing = pd.read_csv(out_path)
        done_ids = set(existing["match_id"].astype(str))
        print(f"  Ya hay {len(done_ids)} partidos guardados de una corrida anterior en {out_path.name} "
              f"-- se saltan, no se vuelven a pedir.")

    print(f"\n=== {league_key} (competition_id={COMPETITION_IDS[league_key]}) ===")
    for season_label, season_id in SEASON_IDS[league_key].items():
        print(f"  Temporada {season_label} (season_id={season_id})...")
        all_matches = _fetch_finished_matches(league_key, season_label, season_id)
        matches = [m for m in all_matches if str(m["id"]) not in done_ids]
        already = len(all_matches) - len(matches)
        print(f"    {len(all_matches)} partidos jugados encontrados "
              f"({already} ya bajados antes, {len(matches)} nuevos a pedir)...")

        season_rows = []
        for i, m in enumerate(matches, start=1):
            match_id = m["id"]
            stats = _fetch_match_stats(match_id)
            season_rows.append({
                "league_key": league_key,
                "season_label": season_label,
                "match_id": match_id,
                "utc_date": m.get("utc_date"),
                "home_team_name": m.get("home_team", {}).get("name"),
                "away_team_name": m.get("away_team", {}).get("name"),
                "home_score": m.get("score", {}).get("home"),
                "away_score": m.get("score", {}).get("away"),
                **stats,
            })
            if i % 50 == 0 or i == len(matches):
                print(f"    ...{i}/{len(matches)} partidos nuevos procesados")

        # Guardado INCREMENTAL -- apenas termina la temporada, no al final de
        # toda la liga. Si algo falla en la temporada siguiente, esto ya
        # quedo en disco.
        if season_rows:
            season_df = pd.DataFrame(season_rows)
            season_df.to_csv(out_path, mode="a", header=not file_has_header, index=False)
            file_has_header = True
            done_ids.update(season_df["match_id"].astype(str))
            print(f"    Guardado -> {out_path} ({len(season_rows)} partidos nuevos)")

        # Chequeo de sanidad de volumen, temporada por temporada -- una
        # temporada "in_progress" (25/26) no se compara contra el esperado.
        expected = EXPECTED_MATCHES_PER_SEASON[league_key]
        if season_label != "25/26" and len(all_matches) < expected * 0.95:
            print(f"    [AVISO] solo {len(all_matches)}/{expected} partidos esperados en {season_label} "
                  f"-- por debajo del 95%, revisar antes de confiar en esta temporada.")

    final_df = pd.read_csv(out_path)
    xg_populated = final_df["home_xg"].notna().mean() if not final_df.empty else 0.0
    print(f"\n  Total {league_key} en disco: {len(final_df)} partidos. "
          f"Cobertura real de xG: {xg_populated:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=list(COMPETITION_IDS.keys()), help="Correr una sola liga")
    parser.add_argument("--all", action="store_true", help="Correr las 4 ligas (~55-65 min totales)")
    args = parser.parse_args()

    if args.all:
        for lk in COMPETITION_IDS:
            download_league(lk)
    elif args.league:
        download_league(args.league)
    else:
        parser.print_help()