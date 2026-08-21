"""
Fase 10 (NBA) -- explota el endpoint HISTORICO de The Odds API, no solo el
endpoint de cuotas en vivo que ya prueba theoddsapi_loader.py.

**Por que este script, no solo el probe anterior**: el probe (theoddsapi_loader.py)
confirmo que Pinnacle NO esta disponible -- eso cierra la puerta a CLV-vs-libro-sharp
"clasico" (apertura/cierre de UN libro conocido, como se hace con Pinnacle en futbol).
Pero el endpoint historico SI habilita algo real y distinto: un CONSENSO no-vig de
~40+ casas retail (DraftKings/FanDuel/BetMGM/etc, confirmado en el probe real del
2026-08-21) capturado dia a dia durante toda una temporada. Eso es:
  1. Un benchmark de "probabilidad de mercado" real para el blend Benter Boost de NBA
     (mismo principio que blending.py de futbol y el blend binario de NFL -- consenso
     de muchos libros en vez de un solo libro sharp).
  2. Una serie de tiempo de movimiento de linea (si se capturan varios snapshots por
     partido) -- un proxy real de "hacia donde se mueve el dinero", aunque no sea
     Pinnacle especificamente.
Ninguna de las dos cosas se calcula aca -- este script SOLO descarga y guarda los
snapshots crudos. El calculo de probabilidad implicita/no-vig va en un script de
limpieza aparte (mismo patron que clean_nfl_data.py separa la conversion de cuotas
americanas de nfl_data_loader.py) -- disciplina del proyecto: ingesta y calculo NUNCA
en el mismo script.

**Endpoints reales usados (confirmado contra la documentacion oficial v4, no
marketing, 2026-08-21)**:
- GET /v4/historical/sports/{sport}/odds -- snapshot de TODOS los partidos activos
  en un timestamp dado. Costo: 10 creditos x region x market POR LLAMADA. Datos
  disponibles desde 2020-06-06 (granularidad de 10 min hasta sept-2022, luego 5 min).
- GET /v4/sports -- lista de deportes, GRATIS (no consume creditos), se usa aca solo
  para chequear creditos restantes sin gastar nada.

**Estrategia de snapshot -- explicita, no perfecta**: un snapshot por dia de partidos
(no por partido individual) cerca del arranque de la jornada, capturado con el
endpoint "bulk" (/historical/sports/{sport}/odds) que trae TODOS los partidos de esa
noche en una sola llamada -- mucho mas barato que pedir partido por partido. Esto es
un PROXY de cuota de cierre (linea cerca del inicio de los partidos), no la cuota
exacta de cierre de cada partido individual (para eso haria falta un snapshot por
evento en su commence_time exacto -- mejora futura, mas cara en creditos, no
necesaria para la primera version del benchmark de mercado).

**Presupuesto de creditos -- guardado real, no un numero adivinado**: con markets=h2h
y regions=us (default), cada snapshot cuesta 10 creditos. Una temporada de NBA tiene
~170-180 noches de partidos -> ~1,700-1,800 creditos por temporada por mercado. El
plan actual tiene 20,000 creditos/mes (confirmado en el probe: 19,985 restantes tras
un solo request de prueba). Este script NUNCA gasta a ciegas: chequea
x-requests-remaining despues de CADA llamada y se detiene ANTES de la siguiente si
excederia --max-credits.

**sport_key -- SOLO 'basketball_nba' esta confirmado con una llamada real** (ver
theoddsapi_loader.py). 'americanfootball_nfl' y 'soccer_usa_mls' son los sport_key
estandar de The Odds API segun su documentacion publica, pero NO se confirmaron
todavia con una llamada real desde este proyecto -- no asumir que funcionan sin
correr --check-sports primero.

Requiere: THEODDSAPI_KEY como variable de entorno (nunca hardcodear la key).

Uso:
    python -m src.ingestion.theoddsapi_historical_loader --check-sports
    python -m src.ingestion.theoddsapi_historical_loader --check-credits
    python -m src.ingestion.theoddsapi_historical_loader --sport NBA --start-date 2024-10-22 --end-date 2025-04-13 --max-credits 2000
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_DATA_DIR

BASE_URL = "https://api.the-odds-api.com/v4"

# Solo NBA esta confirmado con una llamada real (probe de theoddsapi_loader.py,
# 2026-08-21). NFL y MLS son los sport_key documentados por el vendor, sin
# confirmar todavia -- correr --check-sports antes de usarlos en serio.
SPORT_KEYS = {
    "NBA": "basketball_nba",       # CONFIRMADO real
    "NFL": "americanfootball_nfl", # SIN CONFIRMAR
    "MLS": "soccer_usa_mls",       # SIN CONFIRMAR
}

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 3.0
REQUEST_DELAY_SECONDS = 0.5  # esta API no mostro problemas de rate limit en el probe real


def _api_key() -> str:
    import os
    key = os.environ.get("THEODDSAPI_KEY")
    if not key:
        raise EnvironmentError(
            "Falta la variable de entorno THEODDSAPI_KEY -- setearla antes de correr esto. "
            "NUNCA hardcodear la key en este archivo ni commitearla."
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


def check_sports() -> None:
    """Confirma con una llamada real (gratis) que sport_key existe y esta activo --
    paso obligatorio antes de usar NFL/MLS en este script, mismo criterio que el
    resto del proyecto: nunca asumir un identificador, confirmarlo."""
    resp = _get_with_retries(f"{BASE_URL}/sports", {"apiKey": _api_key(), "all": "true"})
    data = resp.json()
    real_keys = {s["key"]: s.get("active", False) for s in data}

    print(f"Deportes reales devueltos por la API: {len(real_keys)}")
    for name, key in SPORT_KEYS.items():
        if key in real_keys:
            print(f"  [OK] {name} -> '{key}' existe. active={real_keys[key]}")
        else:
            print(f"  [FALTA] {name} -> '{key}' NO aparece en la respuesta real -- revisar el sport_key.")


def check_credits() -> None:
    """Llamada gratis (no consume creditos) que igual devuelve los headers reales
    de uso -- forma barata de chequear el presupuesto antes de correr algo caro."""
    resp = _get_with_retries(f"{BASE_URL}/sports", {"apiKey": _api_key()})
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")
    print(f"Creditos usados hasta ahora: {used}")
    print(f"Creditos restantes: {remaining}")


def _daterange(start: datetime, end: datetime, step_days: int):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=step_days)


def download_historical_range(
    sport_name: str,
    start_date: str,
    end_date: str,
    regions: str = "us",
    markets: str = "h2h",
    snapshot_hour_utc: int = 23,
    max_credits: int = 2000,
) -> None:
    """Descarga un snapshot por dia (cerca de snapshot_hour_utc) entre start_date y
    end_date, guardando incrementalmente (mismo patron resumible que
    thestatsapi_xg_loader.py -- si se corta a mitad de camino, la proxima corrida
    retoma desde el ultimo dia ya guardado, no repite trabajo ni gasta creditos de mas)."""
    sport_key = SPORT_KEYS.get(sport_name)
    if sport_key is None:
        print(f"[ERROR] '{sport_name}' no esta en SPORT_KEYS: {list(SPORT_KEYS)}")
        return

    n_markets = len(markets.split(","))
    n_regions = len(regions.split(","))
    cost_per_snapshot = 10 * n_markets * n_regions

    out_dir = RAW_DATA_DIR / "THEODDSAPI"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sport_name}_historical_odds_raw.csv"

    done_dates = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done_dates = set(existing["snapshot_date"].astype(str))
        print(f"Ya hay {len(done_dates)} dias guardados en {out_path} -- se van a saltar.")

    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    credits_spent = 0
    rows_buffer = []
    days_done = 0
    days_skipped = 0

    for day in _daterange(start, end, 1):
        date_str = day.strftime("%Y-%m-%d")
        if date_str in done_dates:
            days_skipped += 1
            continue

        if credits_spent + cost_per_snapshot > max_credits:
            print(f"\n[LIMITE] Parar aca -- el siguiente snapshot ({cost_per_snapshot} creditos) "
                  f"excederia --max-credits={max_credits} (gastado hasta ahora: {credits_spent}). "
                  f"Correr de nuevo con un --max-credits mayor para seguir desde {date_str}.")
            break

        snapshot_dt = day.replace(hour=snapshot_hour_utc)
        iso_ts = snapshot_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        resp = _get_with_retries(
            f"{BASE_URL}/historical/sports/{sport_key}/odds",
            {
                "apiKey": _api_key(),
                "regions": regions,
                "markets": markets,
                "date": iso_ts,
                "oddsFormat": "american",
            },
        )
        data = resp.json()
        used = resp.headers.get("x-requests-used")
        remaining = resp.headers.get("x-requests-remaining")
        credits_spent += cost_per_snapshot

        events = data.get("data", []) if isinstance(data, dict) else data
        n_events = len(events) if events else 0

        for event in (events or []):
            for bm in event.get("bookmakers", []):
                for market in bm.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        rows_buffer.append({
                            "snapshot_date": date_str,
                            "snapshot_timestamp": iso_ts,
                            "event_id": event.get("id"),
                            "commence_time": event.get("commence_time"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "bookmaker": bm.get("key"),
                            "bookmaker_last_update": bm.get("last_update"),
                            "market": market.get("key"),
                            "outcome_name": outcome.get("name"),
                            "outcome_price": outcome.get("price"),
                            "outcome_point": outcome.get("point"),
                        })

        print(f"{date_str}: {n_events} partidos, {cost_per_snapshot} creditos "
              f"(usados={used}, restantes={remaining})")
        days_done += 1

        # Guardado incremental -- mismo criterio que thestatsapi_xg_loader.py:
        # si esto se corta, no se pierde lo ya bajado.
        if rows_buffer:
            chunk = pd.DataFrame(rows_buffer)
            file_exists = out_path.exists()
            chunk.to_csv(out_path, mode="a", header=not file_exists, index=False)
            rows_buffer = []

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{days_done} dias nuevos descargados, {days_skipped} ya estaban, "
          f"{credits_spent} creditos gastados en esta corrida -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-sports", action="store_true")
    parser.add_argument("--check-credits", action="store_true")
    parser.add_argument("--sport", choices=list(SPORT_KEYS), help="NBA/NFL/MLS")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--markets", default="h2h")
    parser.add_argument("--max-credits", type=int, default=2000)
    args = parser.parse_args()

    if args.check_sports:
        check_sports()
    elif args.check_credits:
        check_credits()
    elif args.sport and args.start_date and args.end_date:
        download_historical_range(
            args.sport, args.start_date, args.end_date,
            regions=args.regions, markets=args.markets, max_credits=args.max_credits,
        )
    else:
        parser.print_help()
