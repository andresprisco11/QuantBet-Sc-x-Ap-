"""
Fase 10 (NBA) / riesgo operativo de cuotas en vivo (roadmap) -- probe real
de The Odds API antes de construir nada sobre ella.

**Motivo urgente de este probe, no opcional**: existe una CONTRADICCION
real sin resolver entre dos fuentes de este mismo proyecto sobre si The
Odds API incluye Pinnacle (el libro sharp del que depende toda la
metodologia de CLV):
- Una evaluacion anterior (roadmap, sesion previa a la compra de esta API)
  concluyo que NO la incluye.
- El WebFetch de la pagina de marketing hecho el dia de la compra dice que
  SI la incluye.

Ninguna de las dos es una respuesta real de la API. Este script pide la
lista real de bookmakers disponibles para NBA y la imprime tal cual viene
-- sin interpretar, sin resumir -- para zanjar la duda con un dato real en
vez de dos paginas que se contradicen.

Requiere: THEODDSAPI_KEY como variable de entorno (nunca hardcodear la key
en este archivo ni commitearla).

Uso: python -m src.ingestion.theoddsapi_loader --probe
"""
import os
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"


def _api_key() -> str:
    key = os.environ.get("THEODDSAPI_KEY")
    if not key:
        raise EnvironmentError(
            "Falta la variable de entorno THEODDSAPI_KEY -- setearla antes de correr esto. "
            "NUNCA hardcodear la key en este archivo ni commitearla."
        )
    return key


def probe() -> None:
    """Pide odds reales de NBA y muestra la lista de bookmakers tal cual
    viene en la respuesta -- responde directamente si Pinnacle esta o no,
    sin depender de lo que diga ninguna pagina de marketing."""
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": _api_key(),
            "regions": "us,us2,eu,uk,au",  # todas las regiones -- Pinnacle suele estar en 'eu'
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if not data:
        print("[AVISO] La respuesta vino vacia -- puede ser que no haya partidos de NBA "
              "programados en este momento (fuera de temporada). Igual deberia poder "
              "confirmarse la lista de bookmakers con al menos un partido si hay alguno.")
        return

    all_bookmakers = set()
    for event in data:
        for bm in event.get("bookmakers", []):
            all_bookmakers.add(bm.get("key"))

    print(f"Partidos de NBA devueltos: {len(data)}")
    print(f"\n=== Bookmakers reales encontrados en la respuesta ===")
    for bm in sorted(all_bookmakers):
        print(f"  - {bm}")

    has_pinnacle = "pinnacle" in all_bookmakers
    print(f"\n[RESULTADO REAL] Pinnacle {'SI' if has_pinnacle else 'NO'} esta en la respuesta real de la API.")
    if not has_pinnacle:
        print("[AVISO] Esto confirmaria la evaluacion anterior del roadmap (sin libros sharp) -- "
              "revisar si existe algun parametro de region/plan que lo habilite antes de descartar "
              "The Odds API para CLV, o documentar que este gasto sirve para NBA en general pero NO "
              "para la metodologia de CLV especifica del proyecto.")

    # Chequeo extra, credit cost -- el header x-requests-used/x-requests-remaining
    # confirma cuanto gasto este UNICO request de los 20,000 creditos del plan.
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")
    print(f"\nCreditos usados hasta ahora (segun la propia API): {used}")
    print(f"Creditos restantes: {remaining}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        probe()
    else:
        parser.print_help()
