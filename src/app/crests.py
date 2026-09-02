"""
Resolutor de escudos. Nombre de equipo -> URL del badge, con cache en disco.

### El bug que hubo aca, porque vale la pena que quede escrito

La primera version cacheaba como `null` cualquier equipo que no resolviera.
En una corrida de 50 equipos, los primeros ~31 salieron bien y del 32 en
adelante fallaron TODOS -- incluidos Boca Juniors y River Plate, que
resuelven perfecto cuando se los consulta de a uno.

No era que no tuvieran escudo: era **rate limiting** del tier gratuito. Pero
el codigo no distinguia "la API dice que no existe" de "la API no me
contesto", y guardo las dos cosas como el mismo `null`. Resultado: media
liga argentina quedaba condenada al monograma para siempre, porque el cache
nunca reintentaba.

Es la misma clase de error que ya nos mordio dos veces en este proyecto: un
fallo transitorio disfrazado de resultado definitivo. Ahora:

  - un fallo de red / HTTP -> NO se cachea, se reintenta con backoff
  - la API responde y no hay equipo -> ESO si se cachea como null
  - si tras los reintentos sigue fallando, se deja fuera del cache y se
    avisa por pantalla, para que la proxima corrida lo intente de nuevo
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent.parent / "app" / "crests.json"
API = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t="

PAUSA = 1.2        # el tier gratuito corta cerca de 30 seguidas: hay que ir lento
REINTENTOS = 3
PAUSA_LOTE = 6.0   # respiro adicional cada LOTE consultas
LOTE = 20

ALIAS = {
    "Ath Madrid": "Atletico Madrid",
    "Vallecano": "Rayo Vallecano",
    "Internazionale": "Inter Milan",
    "Nott'm Forest": "Nottingham Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Paris Saint Germain": "Paris Saint-Germain",
    "Velez Sarsfield BA": "Velez Sarsfield",
    "CA Tigre BA": "Tigre",
    "Newells Old Boys": "Newell's Old Boys",
    "Union Santa Fe": "Union de Santa Fe",
    "Atlético Huracán": "Huracan",
    "Belgrano de Cordoba": "Belgrano",
    "Instituto de Córdoba": "Instituto",
    "Estudiantes de Río Cuarto": "Estudiantes de Rio Cuarto",
    "Aldosivi Mar del Plata": "Aldosivi",
    "Gimnasia Mendoza": "Gimnasia y Esgrima de Mendoza",
    "Sarmiento de Junin": "Sarmiento",
    "Hiroshima Sanfrecce FC": "Sanfrecce Hiroshima",
    "Kyoto Purple Sanga": "Kyoto Sanga",
    "V-Varen Nagasaki": "V-Varen Nagasaki",
}


def _cargar() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _guardar(d: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True),
                     encoding="utf-8")


def _buscar(nombre: str):
    """Devuelve (url, definitivo).

    definitivo=True  -> la API contesto; el resultado es confiable (url o None)
    definitivo=False -> no se pudo consultar; NO cachear, reintentar despues
    """
    consulta = ALIAS.get(nombre, nombre)
    for intento in range(REINTENTOS):
        try:
            with urllib.request.urlopen(API + urllib.parse.quote(consulta), timeout=20) as r:
                data = json.load(r)
        except Exception:
            time.sleep(2.0 * (intento + 1))   # backoff lineal
            continue
        for t in (data.get("teams") or []):
            if t.get("strSport") == "Soccer" and t.get("strBadge"):
                return t["strBadge"], True
        return None, True          # la API contesto y no hay equipo: definitivo
    return None, False             # se agotaron los reintentos: transitorio


def resolver(nombres, verbose: bool = True, reintentar_fallidos: bool = False) -> dict:
    """{nombre: url_o_None}. Solo consulta los que faltan.

    reintentar_fallidos=True vuelve a pedir los que estan cacheados como null
    -- util despues de un rate limit, o tras agregar un ALIAS nuevo."""
    cache = _cargar()
    faltan = [n for n in dict.fromkeys(nombres)
              if n not in cache or (reintentar_fallidos and cache.get(n) is None)]
    if not faltan:
        return cache

    if verbose:
        print(f"Resolviendo {len(faltan)} escudos ({len(cache)} en cache). "
              f"Va lento a proposito: el servicio gratuito corta si se lo apura.")
    transitorios = []
    for i, n in enumerate(faltan, 1):
        url, definitivo = _buscar(n)
        if definitivo:
            cache[n] = url
            estado = "ok" if url else "no existe en la API -> monograma"
        else:
            transitorios.append(n)
            estado = "sin respuesta -> se reintenta en la proxima corrida"
        if verbose:
            print(f"   [{i}/{len(faltan)}] {n:<34} {estado}")
        time.sleep(PAUSA)
        if i % LOTE == 0 and i < len(faltan):
            if verbose:
                print(f"   ... pausa de {PAUSA_LOTE:.0f}s para no gatillar el limite")
            time.sleep(PAUSA_LOTE)

    _guardar(cache)
    if transitorios and verbose:
        print(f"\n[AVISO] {len(transitorios)} sin respuesta (no cacheados). "
              f"Volve a correr el exportador y se resuelven.")
    return cache
