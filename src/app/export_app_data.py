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


def construir_partido(grp: pd.DataFrame) -> dict | None:
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

    return {
        "league": grp["league"].iloc[0],
        "date": fecha,
        "home": {"name": home, "abbr": _abbr(home), "color": COLORES.get(home, "#4ea87c")},
        "away": {"name": away, "abbr": _abbr(away), "color": COLORES.get(away, "#4ea87c")},
        "outcomes": outcomes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liga", required=True)
    ap.add_argument("--max-partidos", type=int, default=20)
    args = ap.parse_args()

    raw = fetch_upcoming_odds(args.liga)
    if raw.empty:
        print("Feed vacio.")
        return
    raw = raw[raw["market"] == "h2h"]

    partidos = []
    for _, grp in raw.groupby("event_id"):
        p = construir_partido(grp)
        if p:
            partidos.append(p)
    partidos.sort(key=lambda p: p["date"])
    partidos = partidos[:args.max_partidos]

    data = {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + f" · {args.liga}",
        "partidos": partidos,
        "salud": {"n": 105, "mov": "-5.06%", "t_mov": "-3.49", "clv": "+0.06%"},
    }
    APP_DIR.mkdir(parents=True, exist_ok=True)
    destino = APP_DIR / "data.js"
    destino.write_text(
        "// Generado por src/app/export_app_data.py -- no editar a mano.\n"
        "window.QB_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8")
    print(f"{len(partidos)} partidos exportados -> {destino}")
    if partidos:
        p = partidos[0]
        print(f"\nEjemplo: {p['home']['name']} vs {p['away']['name']}")
        for o in p["outcomes"]:
            print(f"   {o['name']:<14} consenso {o['mkt']:>6.1%}  disp {o['dispersion']:.4f}  "
                  f"mejor {o['mejor_cuota']} en {o['mejor_casa']} (+{o['premio']:.1%})")


if __name__ == "__main__":
    main()
