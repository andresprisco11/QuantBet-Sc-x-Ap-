"""
Fase 16 -- Escaner de ineficiencia de mercado. El primer organo del sistema vivo.

### Por que existe

Las tres tesis anteriores murieron por el mismo error de fondo: **elegimos
donde jugar antes de medir donde habia algo que ganar**. Se asumio que EPL,
LaLiga, Serie A y Bundesliga eran el terreno, y resultaron ser los cuatro
mercados de futbol mas eficientes del planeta.

Este modulo invierte el orden. No predice partidos ni busca apuestas: mide
el TERRENO. Responde una sola pregunta, para cada competicion y deporte que
la API tenga:

    ¿Donde discrepan las casas entre si?

### El cambio de diseño que importa

`soft_book_edge.py` usaba Pinnacle como verdad. Esa fue la falla que mato la
tesis 3: cuando la referencia se mueve -5% en contra, el "edge" era un error
de la referencia, no un error del mercado.

**Este modulo no usa ninguna referencia de verdad.** La dispersion entre
casas es una propiedad observable del mercado: si diez casas cotizan el
mismo resultado a la misma probabilidad, el mercado incorporo la informacion
disponible y no hay nada que buscar ahi. Si discrepan, alguien esta
equivocado -- y no hace falta saber quien para saber que hay algo.

Esa es la condicion de Benter, medida directamente: el mercado del Hong Kong
Jockey Club era batible porque la cuota la ponia el publico, no un
formador de precios profesional. La firma observable de eso es dispersion
alta y ausencia de sharp.

### Las cinco metricas, y que significa cada una

  casas_por_evento   cobertura. Pocas casas = poca presion arbitrajista
  pct_con_sharp      % de eventos donde Pinnacle esta presente.
                     BAJO ES BUENO: sin formador sharp, la linea la pone
                     el publico. Es la condicion estructural de Benter.
  overround          margen total de la casa mediana. Es el COSTO de jugar.
                     Alto = mercado desatendido, pero tambien peaje mas caro.
  dispersion         *** LA METRICA CENTRAL ***
                     desvio estandar de la probabilidad implicita
                     normalizada entre casas, para el mismo resultado.
                     Alta = las casas no se ponen de acuerdo = hay
                     informacion sin incorporar.
  premio_mejor       cuanto paga la mejor casa por encima de la mediana.
                     Es el spread capturable, en terminos directos.

### Memoria: por que cada corrida se guarda

Un mercado ineficiente hoy puede cerrarse en tres meses, y uno eficiente
puede abrirse cuando una casa cambia de proveedor de precios. Cada corrida
se apendea a un historico con timestamp. Eso convierte una foto en una
serie, y es lo que permite que el sistema note por si solo que su terreno
se esta secando -- antes de perder plata averiguandolo.

Uso:
    python -m src.discovery.market_efficiency_scanner --deportes futbol
    python -m src.discovery.market_efficiency_scanner --deportes futbol,nba,tenis
    python -m src.discovery.market_efficiency_scanner --max-ligas 10   # prueba barata
    python -m src.discovery.market_efficiency_scanner --historico      # sin gastar creditos
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.theoddsapi_live_odds_loader import discover_active, fetch_upcoming_odds
from src.evaluation.soft_book_edge import EXCLUDED_BOOKS, OPERATOR_GROUP
from src.tracking.run_logger import RUNS_DIR

SHARP_BOOKS = {"pinnacle"}

# Un evento necesita al menos esto para que la dispersion signifique algo.
# Con 2 casas el desvio estandar es basicamente ruido de una sola comparacion.
MIN_CASAS = 4

# Un ranking sobre 3 eventos no es un ranking. Con pocos eventos, un solo
# partido raro mueve toda la metrica -- se vio en la primera corrida:
# brazil_serie_b dio 0.0075 y 0.0212 en dos scans separados por minutos.
MIN_EVENTOS = 8

SNAPSHOT = RUNS_DIR / "market_efficiency_latest.csv"
HISTORICO = RUNS_DIR / "market_efficiency_historico.csv"


def _normalizar(probs: dict) -> dict:
    """Quita el overround repartiendo proporcionalmente. Aca SI vale usar el
    metodo proporcional, aunque sesgue longshots: no estamos estimando la
    probabilidad verdadera de nada, solo poniendo a todas las casas en la
    misma escala para poder compararlas entre si. El sesgo es identico para
    todas y se cancela en la comparacion."""
    total = sum(probs.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in probs.items()}


def medir_evento(grp: pd.DataFrame) -> dict | None:
    """Mide un (evento, mercado, linea). Devuelve None si no hay material."""
    # Una casa = un operador. unibet_se y unibet_nl son el mismo precio
    # dos veces y inflarian la cobertura sin aportar una opinion distinta.
    grp = grp.copy()
    grp["operador"] = grp["bookmaker"].map(lambda b: OPERATOR_GROUP.get(b, b))
    grp = grp.drop_duplicates(subset=["operador", "outcome_name"])

    vias = sorted(grp["outcome_name"].unique())
    if len(vias) < 2:
        return None

    por_casa = {}
    for casa, g in grp.groupby("operador"):
        precios = dict(zip(g["outcome_name"], g["outcome_price_decimal"]))
        # Solo casas que cotizan el mercado COMPLETO. Una casa con 2 de 3
        # vias no se puede normalizar y ensuciaria la dispersion.
        if set(precios) != set(vias):
            continue
        if any((not p) or p <= 1.0 for p in precios.values()):
            continue
        por_casa[casa] = _normalizar({k: 1.0 / v for k, v in precios.items()})

    n_casas = len(por_casa)
    if n_casas < MIN_CASAS:
        return None

    # --- dispersion: cuanto discrepan las casas sobre el MISMO resultado ---
    dispersiones = []
    for via in vias:
        ps = [p[via] for p in por_casa.values() if via in p]
        if len(ps) >= MIN_CASAS:
            dispersiones.append(np.std(ps, ddof=1))
    if not dispersiones:
        return None

    # --- premio de la mejor casa vs la mediana, en cuota bruta ---
    #     Esto es lo capturable de verdad: lo que pagas de mas por buscar.
    #     Se excluyen exchanges (comision distinta, no comparable).
    premios = []
    apostables = grp[~grp["operador"].isin(EXCLUDED_BOOKS)]
    for via, g in apostables.groupby("outcome_name"):
        cuotas = g["outcome_price_decimal"].dropna()
        cuotas = cuotas[cuotas > 1.0]
        if len(cuotas) >= MIN_CASAS:
            premios.append(cuotas.max() / cuotas.median() - 1.0)

    # --- overround de la casa mediana ---
    overrounds = []
    for casa, g in grp.groupby("operador"):
        precios = dict(zip(g["outcome_name"], g["outcome_price_decimal"]))
        if set(precios) == set(vias) and all(p and p > 1.0 for p in precios.values()):
            overrounds.append(sum(1.0 / p for p in precios.values()) - 1.0)

    return {
        "n_casas": n_casas,
        "n_vias": len(vias),
        "tiene_sharp": int(bool(set(por_casa) & SHARP_BOOKS)),
        "dispersion": float(np.mean(dispersiones)),
        "premio_mejor": float(np.mean(premios)) if premios else np.nan,
        "overround": float(np.median(overrounds)) if overrounds else np.nan,
    }


def escanear_liga(raw: pd.DataFrame, liga: str) -> dict | None:
    if raw.empty:
        return None
    raw = raw.copy()
    raw["_linea"] = raw["outcome_point"].fillna(-999.0)
    ahora = pd.Timestamp.now(tz="UTC")
    filas = []
    for _, grp in raw.groupby(["event_id", "market", "_linea"]):
        m = medir_evento(grp)
        if m:
            # CONFOUND CENTRAL: un partido a 20 dias tiene dispersion alta
            # porque las casas todavia no convergieron, no porque el mercado
            # sea ineficiente. Sin esta columna el ranking premia lejania.
            try:
                ini = pd.to_datetime(grp["commence_time"].iloc[0], utc=True)
                m["horas"] = (ini - ahora).total_seconds() / 3600.0
            except Exception:
                m["horas"] = float("nan")
            filas.append(m)
    if not filas:
        return None
    d = pd.DataFrame(filas)
    return {
        "liga": liga,
        "eventos": len(d),
        "horas_al_evento": round(d["horas"].median(), 1),
        "casas_por_evento": round(d["n_casas"].median(), 1),
        "pct_con_sharp": round(d["tiene_sharp"].mean() * 100, 1),
        "overround": round(d["overround"].median(), 4),
        # MEDIANA, no media: con 10 eventos un partido raro secuestra la media.
        "dispersion": round(d["dispersion"].median(), 5),
        "premio_mejor": round(d["premio_mejor"].median(), 4),
    }


def puntuar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Ranking transparente: se muestran los componentes, no solo el numero.

    La dispersion pesa doble porque es la unica metrica que indica
    INFORMACION SIN INCORPORAR. El premio de la mejor casa es lo capturable.
    La ausencia de sharp es la condicion estructural de Benter.

    No se pondera el overround: un overround alto acompaña a los mercados
    desatendidos (señal buena) pero es al mismo tiempo el peaje (señal mala).
    Mezclarlo en el score esconderia esa tension -- se deja a la vista.
    """
    t = tabla.copy()
    for col, peso in [("dispersion", 2.0), ("premio_mejor", 1.0)]:
        v = t[col]
        rango = v.max() - v.min()
        t[f"_{col}"] = ((v - v.min()) / rango if rango > 0 else 0.0) * peso
    t["_sin_sharp"] = (1.0 - t["pct_con_sharp"] / 100.0) * 1.0
    t["score"] = (t["_dispersion"] + t["_premio_mejor"] + t["_sin_sharp"]).round(3)
    return t.drop(columns=[c for c in t.columns if c.startswith("_")])


def mostrar(t: pd.DataFrame) -> None:
    print(f"\n{'='*100}")
    print("MAPA DE INEFICIENCIA -- donde discrepan las casas entre si")
    print(f"{'='*100}")
    print(f"{'competicion':<38}{'ev':>4}{'dias':>6}{'casas':>7}{'sharp%':>8}"
          f"{'overr':>8}{'dispers':>9}{'premio':>8}{'score':>7}")
    print("-" * 100)
    for _, r in t.iterrows():
        dias = r["horas_al_evento"] / 24.0
        alerta = " <" if dias > 7 else ""
        print(f"{str(r['liga'])[:37]:<38}{int(r['eventos']):>4}{dias:>6.1f}"
              f"{r['casas_por_evento']:>7.1f}"
              f"{r['pct_con_sharp']:>8.0f}{r['overround']:>8.1%}"
              f"{r['dispersion']:>9.4f}{r['premio_mejor']:>8.2%}{r['score']:>7.2f}{alerta}")
    print("-" * 100)
    print("dispersion = desacuerdo entre casas sobre el mismo resultado. ES LA METRICA CENTRAL.")
    print("sharp%     = presencia de Pinnacle. BAJO ES BUENO (sin formador, la linea la pone el publico).")
    print("premio     = cuanto paga la mejor casa sobre la mediana. Es el spread capturable.")
    print("dias       = mediana de dias hasta el evento. *** CONFOUND PRINCIPAL ***")
    print("             Las filas marcadas con '<' estan a mas de 7 dias: su dispersion alta")
    print("             puede ser simplemente que las casas todavia no convergieron.")
    print("             NO se pueden comparar contra ligas que juegan pasado mañana.")
    print("overround  = peaje de la casa mediana. Alto = desatendido, pero tambien mas caro.")
    print("\n[AVISO] Esto mide DONDE MIRAR, no que apostar. Dispersion alta dice que hay")
    print("        informacion sin incorporar; NO dice que vos la tengas. El siguiente paso")
    print("        es probar si en el top del ranking aparece CLV positivo -- con el mismo")
    print("        criterio ex-ante y las mismas dos metricas a prueba de construccion.")


def guardar(t: pd.DataFrame) -> None:
    """Snapshot + historico. El historico es la memoria del organismo: sin
    serie temporal no se puede detectar que un terreno se seco."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(SNAPSHOT, index=False)
    h = t.copy()
    h.insert(0, "scan_utc", datetime.now(timezone.utc).isoformat())
    h.to_csv(HISTORICO, mode="a", header=not HISTORICO.exists(), index=False)
    print(f"\n-> {SNAPSHOT}")
    print(f"-> {HISTORICO}  (apendeado: {len(h)} filas)")


def ver_historico() -> None:
    """Compara el scan mas reciente contra el anterior. No gasta creditos."""
    if not HISTORICO.exists():
        print("Sin historico todavia. Corre un scan primero.")
        return
    h = pd.read_csv(HISTORICO)
    scans = sorted(h["scan_utc"].unique())
    print(f"Scans guardados: {len(scans)}")
    for s in scans:
        print(f"   {s}  ({(h['scan_utc']==s).sum()} competiciones)")
    if len(scans) < 2:
        print("\nHace falta un segundo scan para comparar. Volve en unos dias.")
        return
    a = h[h["scan_utc"] == scans[-2]].set_index("liga")
    b = h[h["scan_utc"] == scans[-1]].set_index("liga")
    comun = a.index.intersection(b.index)
    if comun.empty:
        print("\nSin competiciones en comun entre los dos ultimos scans.")
        return
    d = pd.DataFrame({
        "dispersion_antes": a.loc[comun, "dispersion"],
        "dispersion_ahora": b.loc[comun, "dispersion"],
    })
    d["cambio_%"] = ((d["dispersion_ahora"] / d["dispersion_antes"] - 1) * 100).round(1)
    d = d.sort_values("cambio_%")
    print(f"\nCambio de dispersion entre los dos ultimos scans (n={len(d)}):")
    print("Los de arriba se estan CERRANDO (perdiendo ineficiencia).\n")
    print(d.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deportes", default="futbol",
                    help="futbol,nba,tenis,basquet,nfl (separados por coma)")
    ap.add_argument("--max-ligas", type=int, default=None,
                    help="limita cuantas competiciones escanear (para probar barato)")
    ap.add_argument("--historico", action="store_true",
                    help="compara los dos ultimos scans. NO gasta creditos.")
    args = ap.parse_args()

    if args.historico:
        ver_historico()
        return

    deportes = tuple(d.strip() for d in args.deportes.split(",") if d.strip())
    ligas = discover_active(deportes=deportes)
    # discover_active devuelve {nombre: sport_key}. Iterar el dict daria las
    # CLAVES -- el bug que en su momento rompio 43 de 44 ligas en silencio.
    claves = list(ligas.values())
    if args.max_ligas:
        claves = claves[:args.max_ligas]

    print(f"Escaneando {len(claves)} competiciones (~{len(claves)*6} creditos).\n")

    filas = []
    for i, k in enumerate(claves, 1):
        try:
            raw = fetch_upcoming_odds(k)
            r = escanear_liga(raw, k)
            if r:
                filas.append(r)
                print(f"[{i:>3}/{len(claves)}] {k:<45} {r['eventos']:>3} ev  "
                      f"dispersion {r['dispersion']:.4f}")
            else:
                print(f"[{i:>3}/{len(claves)}] {k:<45} sin material suficiente")
        except Exception as e:
            print(f"[{i:>3}/{len(claves)}] {k:<45} [ERROR] {e}")

    if not filas:
        print("\nNingun mercado con material suficiente.")
        return

    bruto = pd.DataFrame(filas)
    descartadas = bruto[bruto["eventos"] < MIN_EVENTOS]
    t = bruto[bruto["eventos"] >= MIN_EVENTOS].copy()
    if descartadas.empty is False:
        print(f"\n[{len(descartadas)} competiciones descartadas por tener menos de "
              f"{MIN_EVENTOS} eventos: {', '.join(descartadas['liga'].head(8))}"
              f"{'...' if len(descartadas) > 8 else ''}]")
    if t.empty:
        print("Ninguna competicion supera el minimo de eventos.")
        return
    t = puntuar(t).sort_values("score", ascending=False)
    mostrar(t)
    guardar(t)


if __name__ == "__main__":
    main()
