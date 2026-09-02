"""
Fase 18 -- Informacion que el mercado NO esta leyendo.

### Por que esto es distinto de las cuatro tesis muertas

Las cuatro fracasaron por la misma razon de fondo, y recien ahora se ve
completa: **usabamos los mismos datos que usa el mercado**. Cuotas,
resultados, xG. Pinnacle tiene eso, mejor y antes. Intentabamos ganarle
leyendo su propio libro.

El texto es otra cosa. Una rueda de prensa en espanol, un periodista local
de Rosario diciendo que el arquero titular no viaja, una declaracion
postpartido -- eso no esta estructurado, no esta en ningun feed de cuotas, y
sobre todo **no lo lee el mismo numero de ojos** segun el idioma y la liga.

Ahi esta la condicion de Benter, por primera vez en el proyecto: informacion
que otros no tienen, no un modelo mas listo sobre informacion compartida.

### Lo que NO va a funcionar, dicho antes de intentarlo

El "analisis de sentimiento de noticias" es de las ideas mas intentadas y
mas fracasadas del sector. Falla por tres motivos concretos:

  1. **Las noticias grandes ya estan en el precio.** Si Mbappe se lesiona,
     el mercado lo sabe en minutos -- lee los mismos cables que nosotros.
  2. **El sentimiento es casi siempre resultado reciente reciclado.** "El
     equipo esta en racha" es informacion que ya movio la linea hace tres
     dias.
  3. **El backtest de texto miente.** Los articulos se editan despues del
     partido, los timestamps de publicacion se actualizan, y se termina
     usando informacion del futuro sin darse cuenta. Es la trampa numero
     uno de este terreno.

### El diseño que evita las tres

**Se registra `visto_utc`, no `publicado_utc`.** El momento en que NOSOTROS
bajamos la nota, sellado por nuestro reloj. Un medio puede reescribir su
articulo y cambiar la fecha de publicacion; no puede cambiar cuando lo
tuvimos en el disco. Todo analisis usa `visto_utc` y compara SOLO contra
cuotas capturadas despues de ese instante. Sin esto, cualquier resultado
positivo es sospechoso de mirar el futuro.

**Se buscan HECHOS, no animo.** "El tecnico confirmo que el 9 no viaja" es
un hecho verificable con consecuencia sobre el partido. "El equipo llega
motivado" es ruido. La extraccion apunta a: bajas, lesiones, suspensiones,
alineaciones confirmadas, cambios de tecnico, fichajes de ultima hora.

**Se priorizan fuentes en espanol y portugues.** No por gusto: es la unica
ventaja estructural real que tiene esta operacion. Un sindicato de Londres
lee BBC y Sky. Ole, Globo y la prensa local de Argentina y Brasil las lee
mucha menos gente con dinero puesto -- y justo Argentina Primera fue la
competicion que gano el ranking de dispersion en la fase 16.

### La prueba, declarada antes de tener un solo dato

La pregunta NO es "¿el equipo del que se habla bien gana mas?". Es:

    ¿Una noticia vista en el instante T predice hacia donde se mueve la
    linea de Pinnacle DESPUES de T?

Si una nota que vimos a las 14:00 anticipa que Pinnacle bajara el precio del
local a las 18:00, tenemos informacion antes que el mercado sharp. Eso es
edge informacional y **no se puede fabricar por construccion** -- el
movimiento futuro de un tercero no lo controlamos.

Y no hace falta apostar un peso para medirlo.

Uso:
    python -m src.news.news_loader --fetch
    python -m src.news.news_loader --fetch --solo es,pt
    python -m src.news.news_loader --estado
"""
import argparse
import hashlib
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.tracking.run_logger import RUNS_DIR

LOG = RUNS_DIR / "news_log.csv"
UA = {"User-Agent": "Mozilla/5.0 (compatible; QuantBet/1.0)"}
PAUSA = 0.4
DIAS_FEED_MUERTO = 7

# idioma: cuanto menos gente con dinero puesto lo lee, mas vale.
# 'alcance' distingue prensa nacional de cobertura global.
FEEDS = {
    # --- espanol / portugues: la ventaja estructural ---
    "ole_ar":        ("https://www.ole.com.ar/rss/futbol-primera/", "es", "argentina"),
    "globo_br":      ("https://ge.globo.com/rss/ge/futebol/", "pt", "brasil"),
    "marca_liga":    ("https://e00-marca.uecdn.es/rss/futbol/primera-division.xml", "es", "espana"),
    # --- ingles: lo que el mercado ya leyo. Sirve de CONTROL, no de ventaja ---
    "bbc_football":  ("https://feeds.bbci.co.uk/sport/football/rss.xml", "en", "global"),
    "sky_football":  ("https://www.skysports.com/rss/12040", "en", "global"),
    "guardian":      ("https://www.theguardian.com/football/rss", "en", "global"),
    "espn_soccer":   ("https://www.espn.com/espn/rss/soccer/news", "en", "global"),
}

COLUMNAS = ["id", "visto_utc", "publicado_utc", "fuente", "idioma", "alcance",
            "titulo", "resumen", "url", "equipos", "tipo_hecho"]

# Patrones de HECHO, no de animo. Cada uno describe algo verificable con
# consecuencia directa sobre el partido.
HECHOS = {
    "baja": r"\b(baja|bajas|no viaja|no estar[aá]|se pierde|descartad[oa]|ausencia)\b",
    "lesion": r"\b(lesi[oó]n|lesionad[oa]|desgarr|rotura|molestias|f[ií]sic[oa]mente)\b",
    "suspension": r"\b(suspendid[oa]|sanci[oó]n|expulsad[oa]|tarjeta roja|apercibid)\b",
    "alineacion": r"\b(alineaci[oó]n|once|titular|formaci[oó]n confirmada|escalaç[aã]o)\b",
    "tecnico": r"\b(destitu|despedid|cesad|nuevo t[eé]cnico|nuevo entrenador|interin)\b",
    "fichaje": r"\b(fichaje|firm[oó]|traspaso|cedid[oa]|refuerzo|contrataç[aã]o)\b",
    "rueda_prensa": r"\b(rueda de prensa|conferencia de prensa|coletiva|declar[oó])\b",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def _cargar() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=COLUMNAS)


def _texto(el, *tags):
    for t in tags:
        v = el.findtext(t)
        if v:
            return re.sub(r"<[^>]+>", " ", v).strip()
    return ""


def _fecha(s):
    try:
        return parsedate_to_datetime(s).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def bajar_feed(nombre, url, idioma, alcance, ahora):
    req = urllib.request.Request(url, headers=UA)
    raw = urllib.request.urlopen(req, timeout=25).read()
    root = ET.fromstring(raw)
    filas = []
    for it in root.findall(".//item"):
        link = _texto(it, "link", "guid")
        if not link:
            continue
        titulo = _texto(it, "title")
        resumen = _texto(it, "description", "summary")[:400]
        filas.append({
            # id por URL: si el medio reescribe el articulo, sigue siendo el
            # mismo item y NO se registra de nuevo con fecha nueva.
            "id": hashlib.sha1(link.encode()).hexdigest()[:16],
            "visto_utc": ahora,                       # nuestro reloj, sellado
            "publicado_utc": _fecha(_texto(it, "pubDate", "published")),
            "fuente": nombre, "idioma": idioma, "alcance": alcance,
            "titulo": titulo, "resumen": resumen, "url": link,
            "equipos": "", "tipo_hecho": clasificar(f"{titulo} {resumen}"),
        })
    return filas


def clasificar(texto: str) -> str:
    t = _norm(texto)
    return "|".join(k for k, pat in HECHOS.items() if re.search(pat, t)) or ""


def marcar_equipos(df: pd.DataFrame, equipos: list[str]) -> pd.DataFrame:
    """Marca que equipos conocidos menciona cada nota.

    El matching es por nombre normalizado sin acentos. Es deliberadamente
    conservador: prefiere no marcar a marcar mal, porque una nota atribuida
    al equipo equivocado es peor que una nota sin atribuir -- ensucia la
    señal en vez de solo perderla."""
    mapa = {}
    for e in equipos:
        n = _norm(e)
        mapa[n] = e
        # variantes cortas utiles: "Real Sociedad" -> "sociedad" no sirve,
        # pero "Boca Juniors" -> "boca" si. Solo si la palabra tiene >4 letras
        # y no es generica.
        partes = [p for p in n.split() if len(p) > 4 and p not in
                  {"real", "club", "deportivo", "atletico", "futbol", "united", "city"}]
        if partes:
            mapa.setdefault(partes[0], e)
    def buscar(row):
        t = _norm(f"{row['titulo']} {row['resumen']}")
        hits = sorted({v for k, v in mapa.items() if re.search(rf"\b{re.escape(k)}\b", t)})
        return ";".join(hits)
    df = df.copy()
    df["equipos"] = df.apply(buscar, axis=1)
    return df


def fetch(idiomas=None, equipos=None):
    ahora = datetime.now(timezone.utc).isoformat()
    log = _cargar()
    vistos = set(log["id"]) if not log.empty else set()

    nuevas, muertos = [], []
    for nombre, (url, idioma, alcance) in FEEDS.items():
        if idiomas and idioma not in idiomas:
            continue
        try:
            filas = bajar_feed(nombre, url, idioma, alcance, ahora)
        except Exception as e:
            print(f"   {nombre:<16} [ERROR] {str(e)[:50]}")
            continue

        # feed congelado: si lo mas nuevo tiene semanas, la fuente esta muerta
        # y no debe contarse como cobertura. (as.com devolvia notas de 2022.)
        fechas = [f["publicado_utc"] for f in filas if f["publicado_utc"]]
        if fechas:
            mas_nuevo = max(fechas)
            edad = (datetime.now(timezone.utc) - pd.Timestamp(mas_nuevo).to_pydatetime()).days
            if edad > DIAS_FEED_MUERTO:
                muertos.append((nombre, edad))
                print(f"   {nombre:<16} FEED MUERTO: lo mas nuevo tiene {edad} dias")
                continue

        n = [f for f in filas if f["id"] not in vistos]
        nuevas += n
        vistos.update(f["id"] for f in n)
        print(f"   {nombre:<16} {len(filas):>3} items, {len(n):>3} nuevos  [{idioma}]")
        time.sleep(PAUSA)

    if not nuevas:
        print("\nSin notas nuevas.")
        return

    d = pd.DataFrame(nuevas)
    if equipos:
        d = marcar_equipos(d, equipos)
    log = pd.concat([log, d], ignore_index=True)[COLUMNAS]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False)

    con_hecho = (d["tipo_hecho"] != "").sum()
    print(f"\n{len(d)} notas nuevas ({con_hecho} con hecho identificable). "
          f"Total en el log: {len(log)}")
    if equipos:
        print(f"con equipo reconocido: {(d['equipos'] != '').sum()}")
    print(f"-> {LOG}")

    if con_hecho:
        print("\nUltimas con hecho identificable:")
        for _, r in d[d["tipo_hecho"] != ""].head(8).iterrows():
            print(f"   [{r['idioma']}] {r['tipo_hecho'][:24]:<25} {r['titulo'][:60]}")


def estado():
    log = _cargar()
    if log.empty:
        print("Log vacio. Corre --fetch.")
        return
    print(f"{len(log)} notas registradas\n")
    print(log.groupby(["idioma", "fuente"]).size().to_string())
    print(f"\nCon hecho identificable: {(log['tipo_hecho'].fillna('') != '').sum()}")
    tipos = log["tipo_hecho"].fillna("").str.split("|").explode()
    tipos = tipos[tipos != ""]
    if not tipos.empty:
        print("\nPor tipo de hecho:")
        print(tipos.value_counts().to_string())
    v = pd.to_datetime(log["visto_utc"], errors="coerce", utc=True)
    print(f"\nVentana de captura: {v.min()}  ->  {v.max()}")
    print(f"Corridas distintas: {v.nunique()}")
    print("\n[RECORDATORIO] El analisis usa visto_utc, no publicado_utc. Hacen falta")
    print("               varias corridas al dia durante dias para tener con que medir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--solo", default=None, help="idiomas: es,pt,en")
    ap.add_argument("--sin-equipos", action="store_true",
                    help="no intenta reconocer equipos (mas rapido)")
    args = ap.parse_args()

    if args.estado:
        estado()
        return
    if not args.fetch:
        ap.print_help()
        return

    equipos = None
    if not args.sin_equipos:
        try:
            import json
            crests = Path(__file__).resolve().parent.parent.parent / "app" / "crests.json"
            equipos = sorted(json.loads(crests.read_text(encoding="utf-8")).keys())
            print(f"{len(equipos)} equipos conocidos desde crests.json\n")
        except Exception:
            print("[AVISO] sin crests.json, no se marcan equipos\n")

    idiomas = [i.strip() for i in args.solo.split(",")] if args.solo else None
    fetch(idiomas, equipos)


if __name__ == "__main__":
    main()
