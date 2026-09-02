"""
Resolutor de escudos. Nombre de equipo -> URL del badge, con cache en disco.

### Por que hay cache y no se consulta cada vez

TheSportsDB es gratuito y sin key para el tier publico, pero es un servicio
de terceros: puede tener rate limit, caerse, o cambiar. Un escudo no cambia
nunca, asi que se consulta UNA vez por equipo en la vida del proyecto y se
guarda en app/crests.json. A partir de ahi la app funciona sin red.

Los fallos tambien se cachean (como null) para no reintentar en cada corrida
un nombre que la API no conoce -- pasa con equipos de ligas chicas y con
nombres que The Odds API escribe distinto. Para esos, la app cae al
monograma con el color del club, que se ve bien igual.

Si un equipo quedo mal resuelto, se corrige a mano en app/crests.json y no
se vuelve a tocar: el archivo es la fuente de verdad, la API es solo el
sembrador.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent.parent / "app" / "crests.json"
API = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t="
PAUSA = 0.35   # cortesia con un servicio gratuito

# The Odds API y TheSportsDB no siempre escriben igual. Estos son los que ya
# se vieron fallar; se agrega a mano cuando aparezca otro.
ALIAS = {
    "Atletico Madrid": "Atletico Madrid",
    "Ath Madrid": "Atletico Madrid",
    "Vallecano": "Rayo Vallecano",
    "Internazionale": "Inter Milan",
    "Nott'm Forest": "Nottingham Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Paris Saint Germain": "Paris Saint-Germain",
    "Bayern Munich": "Bayern Munich",
    "Sporting Kansas City": "Sporting Kansas City",
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


def _buscar(nombre: str) -> str | None:
    consulta = ALIAS.get(nombre, nombre)
    try:
        url = API + urllib.parse.quote(consulta)
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
    except Exception:
        return None
    for t in (data.get("teams") or []):
        if t.get("strSport") == "Soccer" and t.get("strBadge"):
            return t["strBadge"]
    return None


def resolver(nombres, verbose: bool = True) -> dict:
    """Devuelve {nombre: url_o_None}. Solo consulta los que no estan en cache."""
    cache = _cargar()
    faltan = [n for n in dict.fromkeys(nombres) if n not in cache]
    if faltan and verbose:
        print(f"Resolviendo {len(faltan)} escudos nuevos "
              f"({len(cache)} ya en cache)...")
    for i, n in enumerate(faltan, 1):
        cache[n] = _buscar(n)
        if verbose:
            estado = "ok" if cache[n] else "sin escudo -> monograma"
            print(f"   [{i}/{len(faltan)}] {n:<32} {estado}")
        time.sleep(PAUSA)
    if faltan:
        _guardar(cache)
    return cache
