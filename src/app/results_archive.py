"""
Fase 21 -- Archivo de resultados: el partido jugado no se borra, se guarda.

### Por que esto es el cierre del circuito

Hasta ahora `data.js` era una foto del futuro: los partidos por jugar. Cuando
un partido arrancaba, simplemente desaparecia en la siguiente exportacion y
con el se iba **el precio que el mercado le habia puesto**.

Eso es tirar a la basura justo lo que hace falta para aprender. Un resultado
sin el precio previo no vale casi nada: todo el mundo sabe que Barcelona le
gano a Valencia. Lo que informa es que el mercado le daba 74% y gano -- o que
le daba 74% y perdio.

Guardar (precio previo, resultado) partido a partido construye lo unico que
permite responder la pregunta que ningun modelo nuestro pudo:

    ¿Las probabilidades del mercado estan bien calibradas?

Cuando el mercado dice 70%, ¿pasa el 70% de las veces? Si hay un tramo de
probabilidad donde falla sistematicamente, ese es un sesgo explotable -- y se
mide sin modelo propio, solo acumulando.

### Como se obtienen los resultados

The Odds API tiene `/v4/sports/{key}/scores?daysFrom=N`, que devuelve los
partidos terminados de los ultimos N dias con el marcador. Cuesta 2 creditos
por llamada. No hace falta scrapear nada.

### Que guarda cada fila

El marcador Y el estado del mercado antes del partido: probabilidad de cada
via, mejor cuota disponible y en que casa, dispersion entre casas, y el xG
implicito del mercado. Todo lo que la app ya calculaba y se perdia.

Uso:
    python -m src.app.results_archive --actualizar
    python -m src.app.results_archive --reporte
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(RAIZ))
from src.ingestion.theoddsapi_live_odds_loader import (BASE_URL, _api_key,
                                                       _get_with_retries, ALL_KEYS)
from src.tracking.run_logger import RUNS_DIR

APP = RAIZ / "app"
ARCHIVO = RUNS_DIR / "resultados.csv"

COLUMNAS = ["event_id", "league", "commence_time", "cerrado_utc", "local", "visitante",
            "gl", "gv", "resultado", "p_local", "p_empate", "p_visitante",
            "cuota_local", "cuota_empate", "cuota_visitante",
            "casa_local", "casa_empate", "casa_visitante",
            "disp_media", "xg_local", "xg_visitante", "acerto_favorito"]


def _cargar_archivo() -> pd.DataFrame:
    if ARCHIVO.exists():
        return pd.read_csv(ARCHIVO)
    return pd.DataFrame(columns=COLUMNAS)


def bajar_marcadores(liga: str, dias: int) -> dict:
    """{event_id: (gl, gv, completado)} de los partidos terminados."""
    key = ALL_KEYS.get(liga, liga)
    r = _get_with_retries(f"{BASE_URL}/sports/{key}/scores",
                          {"apiKey": _api_key(), "daysFrom": dias})
    salida = {}
    for ev in r.json():
        if not ev.get("completed"):
            continue
        sc = {s.get("name"): s.get("score") for s in (ev.get("scores") or [])}
        h, a = ev.get("home_team"), ev.get("away_team")
        if h in sc and a in sc:
            try:
                salida[ev["id"]] = (int(sc[h]), int(sc[a]))
            except (TypeError, ValueError):
                continue
    return salida


def _vias(p: dict) -> tuple:
    """Devuelve (local, empate, visitante) en ese orden, mapeando por nombre.

    Los outcomes vienen alfabeticos -- el mismo problema que ya espejo la
    grilla de marcadores. Se mapea por abreviatura del equipo, nunca por
    posicion."""
    o = p.get("outcomes") or []
    def es(x, t):
        n = str(x.get("name", "")).lower()
        ab = str(t.get("abbr", "")).lower()
        return bool(ab) and n.startswith(ab)
    d = next((x for x in o if str(x.get("name", "")).lower() in ("empate", "draw")), None)
    h = next((x for x in o if es(x, p["home"])), None)
    a = next((x for x in o if x is not h and x is not d), None)
    return h, d, a


def actualizar(dias: int) -> None:
    js = APP / "data.js"
    if not js.exists():
        print("No hay app/data.js. Corre export_app_data primero.")
        return
    txt = js.read_text(encoding="utf-8")
    data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    arch = _cargar_archivo()
    ya = set(arch["event_id"]) if not arch.empty else set()

    ahora = datetime.now(timezone.utc)
    # solo se consultan ligas que tienen algun partido ya arrancado
    pendientes = {}
    for p in data["partidos"]:
        try:
            ini = pd.Timestamp(p["ts"]).tz_convert("UTC")
        except Exception:
            continue
        if ini < ahora and p.get("event_id", "") not in ya:
            pendientes.setdefault(p["league"], []).append(p)
    # data.js no guardaba event_id hasta ahora; se cae al par (equipos, fecha)
    if not pendientes:
        print("Ningun partido de data.js ya arranco. Nada que archivar.")
        return

    print(f"{len(pendientes)} ligas con partidos terminados "
          f"(~{len(pendientes)*2} creditos)\n")
    nuevas = []
    for liga, ps in pendientes.items():
        try:
            marc = bajar_marcadores(liga, dias)
        except Exception as e:
            print(f"[ERROR] {liga}: {str(e)[:60]}")
            continue
        n = 0
        for p in ps:
            eid = p.get("event_id")
            gm = marc.get(eid)
            if gm is None:
                continue
            gl, gv = gm
            h, d, a = _vias(p)
            if not (h and d and a):
                continue
            res = "L" if gl > gv else ("E" if gl == gv else "V")
            probs = {"L": h.get("mkt"), "E": d.get("mkt"), "V": a.get("mkt")}
            fav = max(probs, key=lambda k: probs[k] or 0)
            nuevas.append({
                "event_id": eid, "league": liga, "commence_time": p.get("ts"),
                "cerrado_utc": ahora.isoformat(),
                "local": p["home"]["name"], "visitante": p["away"]["name"],
                "gl": gl, "gv": gv, "resultado": res,
                "p_local": h.get("mkt"), "p_empate": d.get("mkt"), "p_visitante": a.get("mkt"),
                "cuota_local": h.get("mejor_cuota"), "cuota_empate": d.get("mejor_cuota"),
                "cuota_visitante": a.get("mejor_cuota"),
                "casa_local": h.get("mejor_casa"), "casa_empate": d.get("mejor_casa"),
                "casa_visitante": a.get("mejor_casa"),
                "disp_media": round(np.mean([x.get("dispersion") or 0
                                             for x in (h, d, a)]), 5),
                "xg_local": (p.get("xg") or [None, None])[0],
                "xg_visitante": (p.get("xg") or [None, None])[1],
                "acerto_favorito": int(res == fav),
            })
            n += 1
        print(f"   {liga:<42} {n:>3} archivados")

    if not nuevas:
        print("\nSin resultados nuevos (los partidos pueden no estar cerrados aun).")
        return
    arch = pd.concat([arch, pd.DataFrame(nuevas)], ignore_index=True)[COLUMNAS]
    ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    arch.to_csv(ARCHIVO, index=False)

    # los archivados salen de data.js: ya no son "por jugar"
    ids = {r["event_id"] for r in nuevas}
    data["partidos"] = [p for p in data["partidos"] if p.get("event_id") not in ids]
    data["ligas"] = sorted({p["league"] for p in data["partidos"]})
    js.write_text("// Generado por src/app/export_app_data.py\n"
                  "window.QB_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n",
                  encoding="utf-8")
    print(f"\n{len(nuevas)} resultados archivados. Total historico: {len(arch)}")
    print(f"-> {ARCHIVO}")


def reporte() -> None:
    """Calibracion: cuando el mercado dice X%, ¿pasa X% de las veces?"""
    d = _cargar_archivo()
    if d.empty:
        print("Archivo vacio.")
        return
    print(f"{len(d)} partidos archivados | {d['league'].nunique()} ligas")
    print(f"acierto del favorito del mercado: {d['acerto_favorito'].mean():.1%}\n")

    # una fila por (partido, via): probabilidad dicha vs si ocurrio
    filas = []
    for _, r in d.iterrows():
        for via, col in (("L", "p_local"), ("E", "p_empate"), ("V", "p_visitante")):
            if pd.notna(r[col]):
                filas.append({"p": float(r[col]), "ocurrio": int(r["resultado"] == via)})
    f = pd.DataFrame(filas)
    if len(f) < 30:
        print(f"[MUESTRA CHICA] {len(f)} observaciones. La calibracion necesita")
        print("                cientos de partidos para decir algo.")
        return

    f["banda"] = pd.cut(f["p"], [0, .1, .2, .3, .4, .5, .65, .8, 1.0])
    g = f.groupby("banda", observed=True).agg(n=("p", "size"), dijo=("p", "mean"),
                                              paso=("ocurrio", "mean"))
    g["dif"] = g["paso"] - g["dijo"]
    g["t"] = g.apply(lambda r: r["dif"] / np.sqrt(max(r["paso"] * (1 - r["paso"]), 1e-9)
                                                  / r["n"]) if r["n"] > 1 else np.nan, axis=1)
    print("CALIBRACION DEL MERCADO -- ¿lo que dice es lo que pasa?\n")
    print(f"{'banda':<16}{'n':>6}{'dijo':>9}{'paso':>9}{'dif':>9}{'t':>8}")
    print("-" * 57)
    for b, r in g.iterrows():
        print(f"{str(b):<16}{int(r['n']):>6}{r['dijo']:>9.1%}{r['paso']:>9.1%}"
              f"{r['dif']:>+9.1%}{r['t']:>8.2f}")
    print("-" * 57)
    print("dif > 0 = el resultado ocurre MAS de lo que el mercado decia (paga de mas)")
    print("Una banda con |t| > 2 sostenida en el tiempo es un sesgo explotable.")
    print("\n[AVISO] Con pocos partidos esto es ruido garantizado. La calibracion es")
    print("        una medicion de meses, no de dias.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actualizar", action="store_true")
    ap.add_argument("--reporte", action="store_true")
    ap.add_argument("--dias", type=int, default=3, help="dias hacia atras (max 3)")
    args = ap.parse_args()
    if args.reporte:
        reporte()
    elif args.actualizar:
        actualizar(min(args.dias, 3))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
