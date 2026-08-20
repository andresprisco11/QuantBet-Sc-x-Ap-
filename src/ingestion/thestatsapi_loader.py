"""
Fase 10 (decision de datos, 2026-08-20 noche) -- candidato nuevo para
destrabar la pausa de xG en futbol (Fase 8, punto 2: Sportmonks pausado por
cobertura historica insuficiente, no por precio). Motivado por el
comentario del usuario: los $50 disponibles son "para llegar a lo que
queremos antes de apostar plata" -- es decir, para mejorar la CALIDAD del
modelo antes de arriesgar capital, no para bankroll de apuestas ni para
cuotas en vivo (eso es Fase 6, todavia no).

**Candidato: TheStatsAPI** (https://www.thestatsapi.com/), plan Starter
$50/mes (100,000 requests/mes), TODOS los endpoints de futbol incluidos,
trial gratuito de 7 dias en todos los planes.

**MISMO CRITERIO DE DISCIPLINA que ya se aplico a Sportmonks: nunca gastar
un peso antes de confirmar con datos reales la cobertura HISTORICA
especifica de xG.** La pagina de marketing de TheStatsAPI dice "10 anios
de datos historicos" en general y "xG disponible para la mayoria de
competiciones top" -- exactamente el mismo tipo de afirmacion vaga que
Sportmonks hacia antes de que el probe real revelara que su xG solo cubre
desde 2024/25. NO se asume que sea distinto esta vez -- se confirma con
una corrida real, aprovechando el trial gratuito de 7 dias ANTES de pagar
los $50.

**Plan de accion, en orden**:
1. El usuario crea una cuenta de trial gratis en thestatsapi.com (sin
   pagar nada todavia) y consigue un API key.
2. Corre este script (`--probe`) con el key -- confirma competition_id de
   las 4 ligas europeas del proyecto y el RANGO REAL de fechas con xG no
   nulo para cada una, exactamente el mismo criterio que
   `sportmonks_loader.py.probe()`/`check_xg_coverage()` ya aplico.
3. Decision de gastar los $50/mes SOLO si la cobertura real cubre al
   menos las mismas ~7 temporadas que ya usa el pipeline de futbol (no
   solo la temporada en curso) -- si no, se documenta como otro intento
   fallido y se sigue sin xG, mismo criterio honesto que con Sportmonks.

Autenticacion (2026-08-20, CONFIRMADO via el "Code Snippets" tab del API
Tester propio de thestatsapi.com, generado por la plataforma con la key
real del usuario -- no es un supuesto sacado de marketing):
`Authorization: Bearer <key>`. El primer 403 NO fue un problema de
autenticacion -- el header ya estaba bien. El sospechoso real es el PATH:
el snippet confirmado por la plataforma pega contra `/api/health`, no
`/api/football/matches` (ese ultimo path era un supuesto mio, nunca
confirmado contra una respuesta real). Por eso este script ahora prueba
`--health` primero (path 100% confirmado) antes de tocar `--probe`
(path de matches, todavia sin confirmar).

Requiere: THESTATSAPI_KEY como variable de entorno (nunca hardcodear la
key en el codigo ni commitearla).

Uso:
    python -m src.ingestion.thestatsapi_loader --health   (correr primero)
    python -m src.ingestion.thestatsapi_loader --probe    (despues, si --health da 200)
"""
import os
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

BASE_URL = "https://api.thestatsapi.com/api"

# Confirmado solo contra la documentacion publica, NO contra una respuesta
# real todavia -- nombres de liga a completar/confirmar con probe() (el
# competition_id real de cada una no esta confirmado, es un placeholder).
TARGET_LEAGUES = {
    "EPL": None,       # TODO: completar con el competition_id real, confirmado via /football/matches
    "LALIGA": None,
    "SERIEA": None,
    "BUNDESLIGA": None,
}


def _headers() -> dict:
    key = os.environ.get("THESTATSAPI_KEY")
    if not key:
        raise EnvironmentError(
            "Falta la variable de entorno THESTATSAPI_KEY -- conseguir un API key del trial "
            "gratuito de 7 dias en thestatsapi.com antes de correr esto. NUNCA hardcodear el key "
            "en este archivo ni commitearlo."
        )
    return {"Authorization": f"Bearer {key}"}


def health() -> None:
    """Path 100% confirmado por la propia plataforma (Code Snippets tab del
    API Tester, generado con la key real del usuario). Sirve para aislar
    definitivamente: si esto da 200, el header/auth esta bien y el 403
    original de /football/matches era un problema de PATH, no de key."""
    resp = requests.get(f"{BASE_URL}/health", headers=_headers(), timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Respuesta: {resp.text}")
    if resp.status_code == 200:
        print(
            "\n[OK] Auth confirmada. El header Authorization: Bearer <key> funciona. "
            "El 403 anterior era el path /football/matches, no la key. Siguiente paso: en el "
            "API Tester, revisar las categorias de endpoint (Health / Competitions / Teams, etc. "
            "-- las que se ven en la pantalla) para encontrar el path REAL de partidos/fixtures, "
            "y pasarmelo para corregir probe() antes de correrlo."
        )
    else:
        print(
            "\n[AVISO] Si esto tambien da error, el problema es mas profundo que un path "
            "equivocado (key invalida, trial no activado, plan sin acceso, etc.) -- pegar la "
            "respuesta completa de arriba (sin la key) para seguir diagnosticando."
        )


def probe() -> None:
    """Confirma, con una llamada real, lo que la pagina de marketing NO dice:
    el rango real de fechas con xG no nulo, liga por liga. Mismo criterio de
    disciplina que sportmonks_loader.py.probe()/check_xg_coverage().

    ADVERTENCIA: el path /football/matches usado aca es TODAVIA un supuesto,
    no esta confirmado contra una respuesta real (a diferencia de /health,
    que si esta confirmado). Correr --health primero; si funciona pero esto
    sigue dando error, el path de matches necesita corregirse con la info
    real del API Tester (categorias de endpoint visibles en la pantalla)
    antes de seguir."""
    resp = requests.get(f"{BASE_URL}/football/matches", headers=_headers(), params={"per_page": 5}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print("Respuesta real de /football/matches (muestra):")
    print(data)
    print(
        "\n[SIGUIENTE PASO MANUAL] Con esta respuesta real, identificar el competition_id de "
        "EPL/LaLiga/SerieA/Bundesliga (buscar el nombre de la liga en el campo correspondiente) y "
        "completar TARGET_LEAGUES arriba. Despues, extender este script para pedir "
        "/football/xg filtrando por cada competition_id y temporada, y confirmar desde que "
        "temporada real hay valores de xG no nulos -- NO asumir que 'la mayoria de competiciones "
        "top' significa las 4 ligas de este proyecto ni que cubre las 7 temporadas que usa el "
        "pipeline de futbol."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true", help="Correr primero -- path confirmado.")
    parser.add_argument("--probe", action="store_true", help="Correr despues de --health.")
    args = parser.parse_args()
    if args.health:
        health()
    elif args.probe:
        probe()
    else:
        parser.print_help()