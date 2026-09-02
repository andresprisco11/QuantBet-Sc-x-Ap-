"""
Fase 20 -- Enriquecer la app con lo que el mercado no muestra.

### Que se agrega y de donde sale

  historial   enfrentamientos previos entre los dos equipos, con marcador,
              tiros al arco y corners. Sale de data/raw/<LIGA>/*.csv, que son
              6 temporadas en formato football-data.co.uk.
  forma       ultimos 5 partidos de cada equipo: goles a favor y en contra,
              tiros al arco propios y concedidos. Misma fuente.
  noticias    notas de data/runs/news_log.csv que mencionan a alguno de los
              dos equipos, con su sello `visto_utc`.

### Lo que NO se agrega, y por que

**Tiros al arco POR JUGADOR en los ultimos 5 partidos: no existe en los datos
que tenemos.** Los CSV historicos traen tiros al arco a nivel EQUIPO (HST /
AST), no por futbolista. Para tenerlo por jugador hace falta un proveedor de
estadisticas de evento (API-Football, Opta, StatsBomb) que es de pago.

Se deja anotado en vez de improvisar algo parecido: un promedio por equipo
disfrazado de dato por jugador seria peor que no tenerlo, porque se veria
igual de convincente en pantalla.

### Cobertura desigual, declarada

Solo 5 ligas tienen historico: EPL, LaLiga, Serie A, Bundesliga y MLS.
Ligue 1, Argentina y Japon no lo tienen todavia. Los partidos de esas ligas
quedan sin historial ni forma, y la interfaz simplemente no muestra la
seccion en vez de mostrarla vacia.

Uso:
    python -m src.app.enrich_app_data
    python -m src.app.enrich_app_data --h2h 6 --forma 5 --dias-noticias 4
"""
import argparse
import difflib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(RAIZ))
from src.tracking.run_logger import RUNS_DIR

APP = RAIZ / "app"
RAW = RAIZ / "data" / "raw"
NEWS = RUNS_DIR / "news_log.csv"

# liga de The Odds API -> carpeta del historico
HIST = {
    "soccer_epl": "EPL", "soccer_spain_la_liga": "LALIGA",
    "soccer_italy_serie_a": "SERIEA", "soccer_germany_bundesliga": "BUNDESLIGA",
    "soccer_usa_mls": "MLS",
}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(fc|cf|afc|ac|as|ca|sc|cd|ud|rc|club|de|the)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def cargar_historico(liga_dir: str) -> pd.DataFrame:
    carpeta = RAW / liga_dir
    if not carpeta.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(carpeta.glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1")
        except Exception:
            continue
        if "HomeTeam" not in d.columns:
            continue
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    d = pd.concat(dfs, ignore_index=True).dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return d.sort_values("Date")


def emparejar(nombre: str, catalogo: dict) -> str | None:
    """Nombre de The Odds API -> nombre del historico.

    Primero exacto normalizado; si no, el mas parecido por encima de 0.78.
    El umbral es alto a proposito: emparejar mal a dos equipos distintos
    mete historial ajeno en una tarjeta, que es peor que no mostrar nada."""
    n = _norm(nombre)
    if n in catalogo:
        return catalogo[n]
    cerca = difflib.get_close_matches(n, list(catalogo), n=1, cutoff=0.78)
    return catalogo[cerca[0]] if cerca else None


def h2h(d: pd.DataFrame, local: str, visitante: str, n: int) -> list:
    m = d[((d.HomeTeam == local) & (d.AwayTeam == visitante)) |
          ((d.HomeTeam == visitante) & (d.AwayTeam == local))].tail(n)
    out = []
    for _, r in m.iloc[::-1].iterrows():
        out.append({
            "fecha": r["Date"].strftime("%d/%m/%y"),
            "local": r["HomeTeam"], "visitante": r["AwayTeam"],
            "gl": int(r["FTHG"]), "gv": int(r["FTAG"]),
            "tal": int(r["HST"]) if pd.notna(r.get("HST")) else None,
            "tav": int(r["AST"]) if pd.notna(r.get("AST")) else None,
        })
    return out


def forma(d: pd.DataFrame, equipo: str, n: int) -> dict | None:
    m = d[(d.HomeTeam == equipo) | (d.AwayTeam == equipo)].tail(n)
    if m.empty:
        return None
    gf = ga = tf = ta = 0
    racha = []
    for _, r in m.iterrows():
        casa = r["HomeTeam"] == equipo
        pf, pc = (r["FTHG"], r["FTAG"]) if casa else (r["FTAG"], r["FTHG"])
        gf += pf; ga += pc
        sf, sc = (r.get("HST"), r.get("AST")) if casa else (r.get("AST"), r.get("HST"))
        if pd.notna(sf): tf += sf
        if pd.notna(sc): ta += sc
        racha.append("G" if pf > pc else ("E" if pf == pc else "P"))
    k = len(m)
    return {"n": k, "racha": racha[::-1],
            "gf": round(gf / k, 2), "gc": round(ga / k, 2),
            "tirosf": round(tf / k, 1), "tirosc": round(ta / k, 1)}


def noticias_por_equipo(dias: int) -> dict:
    if not NEWS.exists():
        return {}
    d = pd.read_csv(NEWS)
    d["visto"] = pd.to_datetime(d["visto_utc"], utc=True, errors="coerce")
    corte = datetime.now(timezone.utc) - timedelta(days=dias)
    d = d[(d["visto"] >= corte) & (d["equipos"].fillna("") != "")]
    mapa = {}
    for _, r in d.sort_values("visto", ascending=False).iterrows():
        item = {"titulo": str(r["titulo"])[:150], "fuente": r["fuente"],
                "idioma": r["idioma"], "hecho": str(r.get("tipo_hecho") or ""),
                "url": r["url"],
                "visto": pd.Timestamp(r["visto"]).strftime("%d/%m %H:%M")}
        for e in str(r["equipos"]).split(";"):
            if e:
                mapa.setdefault(e, [])
                if len(mapa[e]) < 4 and item["titulo"] not in [x["titulo"] for x in mapa[e]]:
                    mapa[e].append(item)
    return mapa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2h", type=int, default=6)
    ap.add_argument("--forma", type=int, default=5)
    ap.add_argument("--dias-noticias", type=int, default=4)
    args = ap.parse_args()

    js = APP / "data.js"
    txt = js.read_text(encoding="utf-8")
    data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])

    hist_cache, cat_cache = {}, {}
    noticias = noticias_por_equipo(args.dias_noticias)
    print(f"noticias con equipo en los ultimos {args.dias_noticias} dias: "
          f"{sum(len(v) for v in noticias.values())} sobre {len(noticias)} equipos\n")

    con_h2h = con_forma = con_news = 0
    sin_hist = set()
    for p in data["partidos"]:
        liga = p["league"]
        # --- noticias: hay para todas las ligas ---
        ns = (noticias.get(p["home"]["name"], []) + noticias.get(p["away"]["name"], []))[:4]
        if ns:
            p["noticias"] = ns
            con_news += 1

        carpeta = HIST.get(liga)
        if not carpeta:
            sin_hist.add(liga)
            continue
        if carpeta not in hist_cache:
            d = cargar_historico(carpeta)
            hist_cache[carpeta] = d
            cat_cache[carpeta] = ({_norm(t): t for t in
                                   set(d.HomeTeam) | set(d.AwayTeam)} if not d.empty else {})
            print(f"{carpeta:<12} {len(d):>5} partidos historicos, "
                  f"{len(cat_cache[carpeta])} equipos")
        d, cat = hist_cache[carpeta], cat_cache[carpeta]
        if d.empty:
            continue

        hl = emparejar(p["home"]["name"], cat)
        vs = emparejar(p["away"]["name"], cat)
        if not hl or not vs:
            continue
        h = h2h(d, hl, vs, args.h2h)
        if h:
            p["h2h"] = h
            con_h2h += 1
        fl, fv = forma(d, hl, args.forma), forma(d, vs, args.forma)
        if fl and fv:
            p["forma"] = {"local": fl, "visitante": fv}
            con_forma += 1

    js.write_text("// Generado por src/app/export_app_data.py + enrich_app_data.py\n"
                  "window.QB_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n",
                  encoding="utf-8")
    n = len(data["partidos"])
    print(f"\n{n} partidos | historial {con_h2h} | forma {con_forma} | noticias {con_news}")
    if sin_hist:
        print(f"\nSin historico (no hay CSV para esa liga): "
              f"{', '.join(sorted(l.replace('soccer_','') for l in sin_hist))}")
    print(f"-> {js}")
    print("\n[NO INCLUIDO] Tiros al arco POR JUGADOR: los CSV traen el dato a nivel")
    print("              EQUIPO (HST/AST), no por futbolista. Hace falta un proveedor")
    print("              de estadisticas de evento, que es de pago.")


if __name__ == "__main__":
    main()
