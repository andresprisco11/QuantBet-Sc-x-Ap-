"""
Fase 19a -- Almacen de fotos del mercado a lo largo del tiempo.

### Por que hace falta guardar la pelicula y no la foto

Todo lo que medimos hasta ahora fue **una foto**: un instante del mercado.
Con una foto solo se puede preguntar "¿quien tiene el mejor precio ahora?".

La pregunta que nadie se hace a este nivel necesita la PELICULA:

    ¿Quien se mueve PRIMERO?

Si una casa blanda cambia su precio y quince minutos despues Pinnacle se
mueve en la misma direccion, esa casa tenia informacion antes que el libro
mas sharp del mundo. Eso es descubrir **por donde entra la informacion al
mercado**, que es una pregunta de microestructura, no de prediccion.

Y conecta directo con el misterio que nos quedo abierto: en la fase 14
Pinnacle se movio -5.06% sistematicamente EN CONTRA de lo que detectabamos
(t=-3.49) y nunca supimos por que. Asumimos todo el tiempo que Pinnacle era
el que sabia. **Nunca testeamos si Pinnacle es realmente el que lidera.**

### Que guarda

Una fila por (instante, evento, casa, resultado). Sin desviguear ni agregar
nada: crudo, para que cualquier analisis futuro pueda recalcular sin volver
a gastar creditos. El archivo crece; se parte por dia.

Uso:
    python -m src.discovery.odds_snapshots --capturar --ligas soccer_epl,soccer_spain_la_liga
    python -m src.discovery.odds_snapshots --estado
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.theoddsapi_live_odds_loader import fetch_upcoming_odds
from src.evaluation.soft_book_edge import OPERATOR_GROUP
from src.tracking.run_logger import RUNS_DIR

DIR = RUNS_DIR / "snapshots"
GRUPOS = {
    "top5": ["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
             "soccer_germany_bundesliga", "soccer_france_ligue_one"],
    "latam": ["soccer_argentina_primera_division", "soccer_brazil_campeonato",
              "soccer_mexico_ligamx", "soccer_chile_campeonato"],
}


def capturar(ligas: list[str]) -> None:
    ahora = datetime.now(timezone.utc)
    filas = []
    for k in ligas:
        try:
            raw = fetch_upcoming_odds(k)
        except Exception as e:
            print(f"[ERROR] {k}: {str(e)[:60]}")
            continue
        if raw.empty:
            continue
        d = raw[raw["market"] == "h2h"].copy()
        d["casa"] = d["bookmaker"].map(lambda b: OPERATOR_GROUP.get(b, b))
        filas.append(d[["event_id", "commence_time", "home_team", "away_team",
                        "casa", "outcome_name", "outcome_price_decimal"]])
        print(f"   {k:<45} {d['event_id'].nunique():>3} eventos, {len(d):>5} cuotas")

    if not filas:
        print("Nada capturado.")
        return
    d = pd.concat(filas, ignore_index=True)
    d.insert(0, "snap_utc", ahora.isoformat())

    DIR.mkdir(parents=True, exist_ok=True)
    archivo = DIR / f"odds_{ahora:%Y%m%d}.csv"
    d.to_csv(archivo, mode="a", header=not archivo.exists(), index=False)
    print(f"\n{len(d)} filas -> {archivo}")

    # Cuantas fotos distintas lleva el dia: es lo que determina si ya se
    # puede hacer lead-lag (hacen falta al menos 6-8 para que signifique algo)
    try:
        prev = pd.read_csv(archivo, usecols=["snap_utc"])
        n = prev["snap_utc"].nunique()
        print(f"Fotos de hoy: {n}" + ("" if n >= 6 else
              f"  (hacen falta ~6+ para medir quien lidera)"))
    except Exception:
        pass


def estado() -> None:
    if not DIR.exists():
        print("Sin snapshots todavia.")
        return
    for f in sorted(DIR.glob("odds_*.csv")):
        try:
            d = pd.read_csv(f, usecols=["snap_utc", "event_id", "casa"])
        except Exception:
            continue
        t = pd.to_datetime(d["snap_utc"], utc=True, errors="coerce")
        print(f"{f.name}  {d['snap_utc'].nunique():>3} fotos  "
              f"{d['event_id'].nunique():>4} eventos  {d['casa'].nunique():>3} casas  "
              f"{t.min():%H:%M} -> {t.max():%H:%M}  ({len(d)} filas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capturar", action="store_true")
    ap.add_argument("--estado", action="store_true")
    ap.add_argument("--ligas", default="top5")
    args = ap.parse_args()

    if args.estado:
        estado()
        return
    if not args.capturar:
        ap.print_help()
        return

    claves = []
    for tok in [t.strip() for t in args.ligas.split(",") if t.strip()]:
        claves += GRUPOS.get(tok, [tok])
    claves = list(dict.fromkeys(claves))
    print(f"Capturando {len(claves)} ligas (~{len(claves)*6} creditos)\n")
    capturar(claves)


if __name__ == "__main__":
    main()
