"""
Puente entre el pipeline y la interfaz: genera app/data.js.

### Decision de diseño que hay que entender antes de tocar esto

Tu mockup tenia una columna MODEL con la probabilidad del modelo propio.
**Esa columna no se llena.** El modelo v4 se midio contra el mercado y el
peso optimo del blend dio 1.00 -- o sea, el modelo no aporta informacion
sobre el precio. Poner su salida en pantalla seria mostrar un numero que ya
demostramos que no vale, y peor: haria que la app se vea como si supiera
algo.

Lo que SI se muestra es lo unico que esta medido y es real:

  consenso    probabilidad implicita mediana entre casas, desvigueada.
              Es lo que el mercado agregado cree.
  mejor       la mejor cuota disponible y en que casa esta.
  premio      cuanto paga esa casa sobre la mediana. Esto es capturable
              sin ningun modelo: es buscar mejor precio, no predecir.
  dispersion  desacuerdo entre casas sobre ese resultado. Alta = alguien
              esta equivocado, aunque no sepamos quien.

Cuando exista un modelo con CLV positivo demostrado, se agrega la columna
`model` a cada outcome y la interfaz la dibuja sola -- ya esta programada
para eso. Hasta entonces queda vacia a proposito.

Uso:
    python -m src.app.export_app_data --liga soccer_argentina_primera_division
    python -m src.app.export_app_data --liga soccer_usa_mls --max-partidos 12
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.theoddsapi_live_odds_loader import fetch_upcoming_odds
from src.evaluation.soft_book_edge import (devig_shin, EXCLUDED_BOOKS, OPERATOR_GROUP)
from src.app.crests import resolver as resolver_escudos
from src.app.poisson_mercado import ajustar, derivar
from src.ingestion.theoddsapi_live_odds_loader import MARKETS_CON_TOTALES

# Atajos para no tener que escribir el sport_key completo.
GRUPOS = {
    "top5": ["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
             "soccer_germany_bundesliga", "soccer_france_ligue_one"],
    "latam": ["soccer_argentina_primera_division", "soccer_brazil_campeonato",
              "soccer_chile_campeonato", "soccer_mexico_ligamx",
              "soccer_brazil_serie_b"],
    "todo": None,   # se resuelve con discover_active
}

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
MIN_CASAS = 4

# Colores de club para el monograma cuando no hay escudo. Se completa a mano
# con el tiempo; lo que no este aca cae a verde por defecto.
COLORES = {
    "Barcelona": "#a50044", "Real Madrid": "#febe10", "Boca Juniors": "#1e3a8a",
    "River Plate": "#e01e2d", "Inter Miami": "#f7b5cd", "Flamengo": "#e01e2d",
}


def _abbr(nombre: str) -> str:
    partes = [p for p in nombre.replace("-", " ").split() if p]
    if len(partes) >= 2:
        return (partes[0][0] + partes[1][0] + partes[-1][0])[:3].upper()
    return nombre[:3].upper()


def _totales_consenso(tot: pd.DataFrame) -> dict:
    """{linea: prob_over} mediana entre casas, desvigueada por casa.

    Se desviguea el par Over/Under de CADA casa por separado y despues se
    toma la mediana. Al reves (mediana de cuotas y despues desvig) mezclaria
    margenes distintos y sesgaria el resultado."""
    salida = {}
    for linea, g in tot.groupby("outcome_point"):
        probs = []
        for _, gg in g.groupby("op"):
            pr = dict(zip(gg["outcome_name"], gg["outcome_price_decimal"]))
            if len(pr) != 2 or any(not v or v <= 1 for v in pr.values()):
                continue
            inv = {k: 1.0 / v for k, v in pr.items()}
            tot_inv = sum(inv.values())
            over = next((k for k in inv if str(k).lower().startswith("over")), None)
            if over and tot_inv > 0:
                probs.append(inv[over] / tot_inv)
        if len(probs) >= MIN_CASAS:
            salida[float(linea)] = float(np.median(probs))
    return salida


def construir_partido(grp: pd.DataFrame, tot: pd.DataFrame | None = None) -> dict | None:
    grp = grp.copy()
    grp["op"] = grp["bookmaker"].map(lambda b: OPERATOR_GROUP.get(b, b))
    grp = grp.drop_duplicates(subset=["op", "outcome_name"])
    vias = sorted(grp["outcome_name"].unique())
    if len(vias) < 2:
        return None

    # --- probabilidad justa por casa (Shin, mismo metodo que la deteccion) ---
    justas = {}
    for casa, g in grp.groupby("op"):
        precios = dict(zip(g["outcome_name"], g["outcome_price_decimal"]))
        if set(precios) != set(vias) or any(not p or p <= 1 for p in precios.values()):
            continue
        try:
            justas[casa] = devig_shin(precios)
        except Exception:
            continue
    if len(justas) < MIN_CASAS:
        return None

    home = grp["home_team"].iloc[0]
    away = grp["away_team"].iloc[0]
    apostables = grp[~grp["op"].isin(EXCLUDED_BOOKS)]

    outcomes = []
    for via in vias:
        ps = [j[via] for j in justas.values() if via in j]
        if len(ps) < MIN_CASAS:
            continue
        g = apostables[apostables["outcome_name"] == via]
        cuotas = g[["op", "outcome_price_decimal"]].dropna()
        cuotas = cuotas[cuotas["outcome_price_decimal"] > 1.0]
        mejor_casa, mejor_cuota, premio = None, None, None
        if len(cuotas) >= MIN_CASAS:
            i = cuotas["outcome_price_decimal"].idxmax()
            mejor_casa = cuotas.at[i, "op"]
            mejor_cuota = float(cuotas.at[i, "outcome_price_decimal"])
            premio = mejor_cuota / float(cuotas["outcome_price_decimal"].median()) - 1.0
        # nombre corto legible
        etiqueta = ("empate" if via.lower() == "draw"
                    else f"{_abbr(via)} win" if via in (home, away) else via)
        outcomes.append({
            "name": etiqueta,
            "_via": via,          # nombre crudo: necesario para saber quien es local

            "model": None,          # se llena cuando exista modelo validado
            "mkt": round(float(np.median(ps)), 4),
            "dispersion": round(float(np.std(ps, ddof=1)), 4),
            "mejor_cuota": mejor_cuota,
            "mejor_casa": mejor_casa,
            "premio": round(premio, 4) if premio is not None else None,
            "n_casas": len(ps),
        })
    if not outcomes:
        return None

    try:
        fecha = pd.to_datetime(grp["commence_time"].iloc[0], utc=True).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        fecha = ""

    # --- traduccion del mercado a grilla de marcadores ---
    #     NO es prediccion: es el precio del mercado reexpresado.
    # BUG CORREGIDO 2026-09-02: antes se tomaban las probabilidades en el orden
    # en que quedaban en la lista, y esa lista viene de `sorted(outcome_name)`
    # -- o sea ORDEN ALFABETICO, no local/visitante. En Valencia vs Barcelona
    # el orden alfabetico pone Barcelona primero, asi que su 75% entraba como
    # probabilidad del LOCAL siendo el visitante. Resultado: la grilla entera
    # espejada (xg 2.63-0.89 al reves, "visitante gana por 2+" 3% en vez de 53%).
    # Ahora se mapea por nombre de equipo explicitamente.
    extra = {}
    try:
        por_via = {o["_via"]: o["mkt"] for o in outcomes}
        p_home = por_via.get(home)
        p_away = por_via.get(away)
        p_draw = next((v for k, v in por_via.items() if str(k).lower() == "draw"), None)
        if None not in (p_home, p_away, p_draw):
            totales = _totales_consenso(tot) if tot is not None and not tot.empty else {}
            lh, la, err = ajustar(p_home, p_draw, p_away, totales)
            if err < 0.05:          # ajuste malo -> no se muestra nada
                extra = derivar(lh, la, local=_abbr(home), visitante=_abbr(away))
                extra["ajuste_err"] = round(err, 4)
                extra["n_totales"] = len(totales)
    except Exception:
        extra = {}

    for o in outcomes:
        o.pop("_via", None)
    premios = [o["premio"] for o in outcomes if o["premio"] is not None]
    return {
        "event_id": grp["event_id"].iloc[0],   # lo necesita results_archive
        "league": grp["league"].iloc[0],
        "date": fecha,
        "ts": str(grp["commence_time"].iloc[0]),
        "home": {"name": home, "abbr": _abbr(home), "color": COLORES.get(home, "#4ea87c")},
        "away": {"name": away, "abbr": _abbr(away), "color": COLORES.get(away, "#4ea87c")},
        "outcomes": outcomes,
        "max_premio": round(max(premios), 4) if premios else None,
        "max_disp": round(max(o["dispersion"] for o in outcomes), 4),
        **extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligas", default="top5",
                    help="claves separadas por coma, o un grupo: top5 / latam / todo")
    ap.add_argument("--liga", default=None, help="alias de --ligas para una sola")
    ap.add_argument("--max-partidos", type=int, default=200)
    ap.add_argument("--sin-escudos", action="store_true",
                    help="salta la resolucion de escudos (no toca la red)")
    ap.add_argument("--reintentar-escudos", action="store_true",
                    help="vuelve a pedir los escudos cacheados como fallidos")
    ap.add_argument("--reemplazar", action="store_true",
                    help="borra data.js y deja SOLO las ligas de esta corrida")
    args = ap.parse_args()

    # BUG CORREGIDO: antes se comparaba la cadena ENTERA contra GRUPOS, asi que
    # "top5,soccer_japan..." no matcheaba y "top5" se mandaba como sport_key.
    # Los grupos se expanden token por token.
    pedido = args.liga or args.ligas
    claves = []
    for tok in [t.strip() for t in pedido.split(",") if t.strip()]:
        if tok == "todo":
            from src.ingestion.theoddsapi_live_odds_loader import discover_active
            claves += list(discover_active(deportes=("futbol",)).values())
        elif tok in GRUPOS:
            claves += GRUPOS[tok]
        else:
            claves.append(tok)
    claves = list(dict.fromkeys(claves))

    print(f"Exportando {len(claves)} competicion(es) (~{len(claves)*6} creditos).\n")

    partidos = []
    for k in claves:
        try:
            raw = fetch_upcoming_odds(k, markets=MARKETS_CON_TOTALES)
        except Exception as e:
            print(f"[ERROR] {k}: {e}")
            continue
        if raw.empty:
            continue
        raw = raw.copy()
        raw["op"] = raw["bookmaker"].map(lambda b: OPERATOR_GROUP.get(b, b))
        h2h = raw[raw["market"] == "h2h"]
        tot_all = raw[raw["market"] == "totals"]
        antes = len(partidos)
        for ev, grp in h2h.groupby("event_id"):
            p = construir_partido(grp, tot_all[tot_all["event_id"] == ev])
            if p:
                partidos.append(p)
        print(f"   {k:<45} {len(partidos)-antes:>3} partidos")

    partidos.sort(key=lambda p: p["ts"])
    partidos = partidos[:args.max_partidos]

    # --- escudos: una consulta por equipo en la vida del proyecto ---
    if partidos and not args.sin_escudos:
        nombres = [t["name"] for p in partidos for t in (p["home"], p["away"])]
        escudos = resolver_escudos(nombres, reintentar_fallidos=args.reintentar_escudos)
        for p in partidos:
            for lado in ("home", "away"):
                u = escudos.get(p[lado]["name"])
                if u:
                    p[lado]["crest"] = u

    # --- FUSION con lo que ya estaba en data.js ---
    # Antes se reescribia el archivo entero, asi que correr `--ligas top5`
    # BORRABA Japon y Argentina de la app sin avisar. Ahora se reemplazan
    # SOLO las ligas de esta corrida y el resto se conserva: es el
    # comportamiento que uno espera al actualizar una parte.
    ligas_nuevas = {p["league"] for p in partidos}
    conservados = []
    destino_js = APP_DIR / "data.js"
    if not args.reemplazar and destino_js.exists():
        try:
            txt = destino_js.read_text(encoding="utf-8")
            crudo = txt[txt.index("{"):txt.rindex("}") + 1]
            previo = json.loads(crudo)
            conservados = [p for p in previo.get("partidos", [])
                           if p.get("league") not in ligas_nuevas]
        except Exception as e:
            print(f"[AVISO] no se pudo leer data.js previo ({str(e)[:40]}); "
                  f"se escribe solo lo de esta corrida.")
    if conservados:
        print(f"conservadas {len({p['league'] for p in conservados})} ligas previas "
              f"({len(conservados)} partidos) que no venian en esta corrida")
    partidos = sorted(partidos + conservados, key=lambda p: p.get("ts", ""))

    data = {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ligas": sorted({p["league"] for p in partidos}),
        "partidos": partidos,
        "salud": {"n": 105, "mov": "-5.06%", "t_mov": "-3.49", "clv": "+0.06%"},
    }
    APP_DIR.mkdir(parents=True, exist_ok=True)
    destino = APP_DIR / "data.js"
    destino.write_text(
        "// Generado por src/app/export_app_data.py -- no editar a mano.\n"
        "window.QB_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8")
    print(f"\n{len(partidos)} partidos de {len(data['ligas'])} ligas -> {destino}")
    con_escudo = sum(1 for p in partidos for l in ("home","away") if p[l].get("crest"))
    con_grilla = sum(1 for p in partidos if p.get("scores"))
    print(f"escudos resueltos : {con_escudo}/{len(partidos)*2}")
    print(f"grilla de marcadores: {con_grilla}/{len(partidos)} partidos")

    mejores = sorted([p for p in partidos if p["max_premio"]],
                     key=lambda p: -p["max_premio"])[:5]
    if mejores:
        print("\nMayor premio por buscar mejor precio (NO es edge, es buscar mejor precio):")
        for p in mejores:
            o = max([o for o in p["outcomes"] if o["premio"] is not None],
                    key=lambda o: o["premio"])
            print(f"   +{o['premio']:>5.1%}  {p['home']['name'][:18]:<19} v "
                  f"{p['away']['name'][:18]:<19} {o['name']:<12} "
                  f"{o['mejor_cuota']} en {o['mejor_casa']}")


if __name__ == "__main__":
    main()
