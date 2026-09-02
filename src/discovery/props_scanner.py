"""
Fase 17 -- ¿Los mercados secundarios estan peor preciados que el 1X2?

### De donde sale esta pregunta

No de la teoria: de una observacion del usuario. Sus boletos ganadores no
eran de resultado sino de **corners, saves del arquero y totales por equipo**.
La hipotesis que se desprende es estructural y no una corazonada:

    Los mercados secundarios no los cotiza el mismo equipo ni con la misma
    atencion que el 1X2. Menos liquidez, menos dinero sharp corrigiendolos,
    mas error del propio libro. Si eso es cierto, ahi deberia haber mas
    desacuerdo entre casas.

Si se confirma, es donde el conocimiento de estilo de juego de un equipo
(posesion alta -> mas corners) puede valer algo, porque el libro le presta
menos atencion a esa dimension que a quien gana.

### Por que la comparacion es PAREADA y no entre ligas

El escaner de mercados comparaba competiciones entre si, y ahi el lead time
casi arruina el resultado: Conference League parecia ineficiente y solo
estaba a 43 dias.

Aca se compara **el mismo partido consigo mismo**: dispersion del mercado
secundario menos dispersion del 1X2, evento por evento. Eso controla de un
saque liga, equipos, hora, numero de casas y lead time -- todo lo que podria
confundir queda dentro del par y se cancela en la resta.

El estadistico es un t pareado sobre esas diferencias. Es la misma disciplina
del experimento CLV: una metrica que no se puede satisfacer por construccion.

### Advertencia que hay que leer antes de festejar

Dispersion mas alta en props **no significa que se pueda ganar ahi**. Los
mercados secundarios tambien cargan MAS margen (overround), asi que el peaje
sube junto con el desacuerdo. Por eso se reporta el overround al lado: si la
dispersion sube 50% pero el overround sube 100%, el terreno es peor, no mejor.

Uso:
    python -m src.discovery.props_scanner --probe --liga soccer_epl
    python -m src.discovery.props_scanner --liga soccer_epl
    python -m src.discovery.props_scanner --liga soccer_spain_la_liga --mercados totals,btts
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.theoddsapi_live_odds_loader import (
    BASE_URL, _api_key, _get_with_retries, ALL_KEYS, FETCH_REGIONS)
from src.evaluation.soft_book_edge import OPERATOR_GROUP
from src.tracking.run_logger import RUNS_DIR

# Candidatos a mercado secundario en futbol. No todos existen en todas las
# competiciones ni en todos los planes -- por eso el modo --probe: la API
# devuelve 422 para el que no soporta, y se descubre en vez de suponerse.
CANDIDATOS = [
    "totals", "spreads", "btts", "draw_no_bet", "double_chance",
    "team_totals", "alternate_totals", "alternate_spreads",
]

MIN_CASAS = 4
SALIDA = RUNS_DIR / "props_dispersion.csv"


def _resolver(liga: str) -> str:
    return ALL_KEYS.get(liga, liga)


def bajar(liga: str, mercados: str) -> pd.DataFrame:
    """Una sola llamada con todos los mercados juntos: la API cobra por
    (mercado x region), asi que pedirlos juntos cuesta igual que separados
    pero devuelve todo en un snapshot coherente -- importante, porque
    comparar dispersiones tomadas en momentos distintos seria trampa."""
    resp = _get_with_retries(
        f"{BASE_URL}/sports/{_resolver(liga)}/odds",
        {"apiKey": _api_key(), "regions": FETCH_REGIONS,
         "markets": mercados, "oddsFormat": "decimal"})
    filas = []
    for ev in resp.json():
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                for oc in mk.get("outcomes", []):
                    filas.append({
                        "event_id": ev.get("id"),
                        "commence_time": ev.get("commence_time"),
                        "partido": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                        "casa": OPERATOR_GROUP.get(bm.get("key"), bm.get("key")),
                        "mercado": mk.get("key"),
                        "via": oc.get("name"),
                        "linea": oc.get("point"),
                        "cuota": oc.get("price"),
                        # 'description' distingue el equipo en team_totals
                        "desc": oc.get("description"),
                    })
    print(f"   restantes: {resp.headers.get('x-requests-remaining')}")
    return pd.DataFrame(filas)


def medir(g: pd.DataFrame) -> dict | None:
    """Dispersion y overround de un (evento, mercado, linea, descripcion)."""
    vias = sorted(g["via"].dropna().unique())
    if len(vias) < 2:
        return None
    por_casa, overrounds = {}, []
    for casa, gg in g.groupby("casa"):
        pr = dict(zip(gg["via"], gg["cuota"]))
        if set(pr) != set(vias) or any(not v or v <= 1 for v in pr.values()):
            continue
        inv = {k: 1.0 / v for k, v in pr.items()}
        s = sum(inv.values())
        if s <= 0:
            continue
        overrounds.append(s - 1.0)
        por_casa[casa] = {k: v / s for k, v in inv.items()}
    if len(por_casa) < MIN_CASAS:
        return None
    disp = [np.std([p[v] for p in por_casa.values()], ddof=1) for v in vias]
    return {"dispersion": float(np.mean(disp)),
            "overround": float(np.median(overrounds)),
            "n_casas": len(por_casa)}


def analizar(raw: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (evento, mercado). Las lineas de un mismo mercado se
    promedian: over 2.5 y over 3.5 son apuestas distintas pero pertenecen al
    mismo mercado, y lo que se compara es el MERCADO contra el 1X2."""
    raw = raw.copy()
    raw["_linea"] = raw["linea"].fillna(-999.0)
    raw["_desc"] = raw["desc"].fillna("")
    filas = []
    for (ev, mercado, _l, _d), g in raw.groupby(
            ["event_id", "mercado", "_linea", "_desc"]):
        m = medir(g)
        if m:
            m.update({"event_id": ev, "mercado": mercado,
                      "partido": g["partido"].iloc[0]})
            filas.append(m)
    if not filas:
        return pd.DataFrame()
    d = pd.DataFrame(filas)
    return (d.groupby(["event_id", "partido", "mercado"])
             .agg(dispersion=("dispersion", "mean"),
                  overround=("overround", "median"),
                  n_casas=("n_casas", "median"),
                  n_lineas=("dispersion", "size"))
             .reset_index())


def reportar(t: pd.DataFrame) -> None:
    base = t[t["mercado"] == "h2h"].set_index("event_id")
    if base.empty:
        print("Sin 1X2 de referencia, no se puede parear.")
        return

    print(f"\n{'='*92}")
    print("PROPS vs 1X2 -- comparacion PAREADA dentro del mismo partido")
    print(f"{'='*92}")
    print(f"{'mercado':<20}{'pares':>7}{'disp props':>12}{'disp 1X2':>11}"
          f"{'ratio':>8}{'t pareado':>11}{'overr props':>13}{'overr 1X2':>11}")
    print("-" * 92)

    for mercado, g in t[t["mercado"] != "h2h"].groupby("mercado"):
        g = g.set_index("event_id")
        comun = g.index.intersection(base.index)
        if len(comun) < 5:
            print(f"{mercado:<20}{len(comun):>7}   (menos de 5 pares, se omite)")
            continue
        dp = g.loc[comun, "dispersion"].astype(float)
        dh = base.loc[comun, "dispersion"].astype(float)
        dif = (dp - dh).dropna()
        t_stat = dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif))) if len(dif) > 1 else np.nan
        print(f"{mercado:<20}{len(dif):>7}{dp.mean():>12.4f}{dh.mean():>11.4f}"
              f"{dp.mean()/dh.mean():>8.2f}{t_stat:>11.2f}"
              f"{g.loc[comun,'overround'].median():>12.1%}"
              f"{base.loc[comun,'overround'].median():>11.1%}")

    print("-" * 92)
    print("ratio > 1  = el mercado secundario dispersa MAS que el 1X2 del mismo partido")
    print("t pareado  = |t| > 2 significa que la diferencia no es casualidad de la muestra")
    print("\n[LECTURA] Un ratio alto NO alcanza. Compara tambien las dos columnas de")
    print("          overround: si el peaje sube mas que el desacuerdo, el terreno es")
    print("          PEOR aunque disperse mas. Y dispersion alta dice que alguien se")
    print("          equivoca, no que seas vos quien tiene razon -- eso lo decide el CLV.")


def _pedir_sin_reintentos(url: str, params: dict):
    """Consulta directa, SIN los reintentos del loader.

    Por que: `_get_with_retries` reintenta 4 veces con backoff ante cualquier
    error. Eso esta bien para un fallo de red, pero un **422 significa "este
    mercado no existe"** -- es una respuesta definitiva del servidor, no un
    tropiezo. Reintentarla 4 veces perdia 30 segundos por cada mercado no
    soportado, y el probe tardaba 3 minutos en decir lo que sabe al primer
    intento.

    Es la misma familia de error que ya nos mordio con el cache de escudos:
    tratar un "no existe" como si fuera un "no pude preguntar".
    """
    r = requests.get(url, params=params, timeout=25)
    if r.status_code == 422:
        return None, "no soportado en este endpoint"
    if r.status_code == 429:
        return None, "rate limit (reintentable)"
    if not r.ok:
        return None, f"HTTP {r.status_code}"
    return r, None


def probe(liga: str, evento: bool = False) -> None:
    """Descubre que mercados soporta la competicion.

    The Odds API parte los mercados en dos endpoints: los principales van en
    /sports/{key}/odds, y los "adicionales" (btts, team_totals, lineas
    alternativas, props de jugador) SOLO existen en el endpoint por evento
    /sports/{key}/events/{id}/odds. Un 422 en el primero no significa que el
    mercado no exista: puede estar en el segundo. Por eso se prueban los dos.
    """
    sk = _resolver(liga)
    print(f"--- endpoint de liga: {len(CANDIDATOS)} mercados ---\n")
    ok = []
    for m in CANDIDATOS:
        r, err = _pedir_sin_reintentos(
            f"{BASE_URL}/sports/{sk}/odds",
            {"apiKey": _api_key(), "regions": FETCH_REGIONS,
             "markets": m, "oddsFormat": "decimal"})
        if err:
            print(f"   {m:<22} {err}")
            continue
        data = r.json()
        n = sum(len(mk.get("outcomes", []))
                for e in data for b in e.get("bookmakers", [])
                for mk in b.get("markets", []))
        if n:
            ok.append(m)
        print(f"   {m:<22} {len(data)} eventos, {n} cuotas"
              if n else f"   {m:<22} soportado pero VACIO")

    if evento:
        # un solo evento de muestra: los adicionales se cobran por evento,
        # asi que no se barre la liga entera para averiguar que existe
        r, err = _pedir_sin_reintentos(
            f"{BASE_URL}/sports/{sk}/events", {"apiKey": _api_key()})
        eventos = r.json() if r else []
        if eventos:
            ev = eventos[0]
            print(f"\n--- endpoint por evento: {ev.get('home_team')} vs "
                  f"{ev.get('away_team')} ---\n")
            for m in CANDIDATOS:
                if m in ok:
                    continue
                r2, err2 = _pedir_sin_reintentos(
                    f"{BASE_URL}/sports/{sk}/events/{ev['id']}/odds",
                    {"apiKey": _api_key(), "regions": FETCH_REGIONS,
                     "markets": m, "oddsFormat": "decimal"})
                if err2:
                    print(f"   {m:<22} {err2}")
                    continue
                d2 = r2.json()
                n2 = sum(len(mk.get("outcomes", []))
                         for b in d2.get("bookmakers", [])
                         for mk in b.get("markets", []))
                casas = len({b.get("key") for b in d2.get("bookmakers", [])
                             for mk in b.get("markets", []) if mk.get("outcomes")})
                print(f"   {m:<22} {n2} cuotas en {casas} casas"
                      if n2 else f"   {m:<22} soportado pero VACIO")
        else:
            print("\n[AVISO] no se pudo listar eventos para el probe por evento.")

    print(f"\nUsables en el endpoint de liga: {','.join(ok) if ok else 'ninguno'}")
    if ok:
        print(f"\n   python -m src.discovery.props_scanner --liga {liga} "
              f"--mercados {','.join(ok)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liga", required=True)
    ap.add_argument("--mercados", default="totals,spreads,btts,double_chance",
                    help="mercados secundarios a comparar contra h2h")
    ap.add_argument("--probe", action="store_true",
                    help="descubre que mercados soporta la liga y termina")
    ap.add_argument("--probe-evento", action="store_true",
                    help="ademas prueba el endpoint por evento (mercados adicionales)")
    args = ap.parse_args()

    if args.probe or args.probe_evento:
        probe(args.liga, evento=args.probe_evento)
        return

    mercados = "h2h," + args.mercados
    n_mk = len(mercados.split(","))
    print(f"Bajando {n_mk} mercados x 3 regiones (~{n_mk*3} creditos)...")
    raw = bajar(args.liga, mercados)
    if raw.empty:
        print("Feed vacio.")
        return
    print(f"{len(raw)} cuotas, {raw['event_id'].nunique()} eventos, "
          f"{raw['mercado'].nunique()} mercados con datos")

    t = analizar(raw)
    if t.empty:
        print("Sin material suficiente (hacen falta >=4 casas por mercado).")
        return
    reportar(t)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    t.insert(0, "liga", args.liga)
    t.to_csv(SALIDA, mode="a", header=not SALIDA.exists(), index=False)
    print(f"\n-> {SALIDA}")


if __name__ == "__main__":
    main()
