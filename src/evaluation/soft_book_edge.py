"""
Fase 12 -- CAMBIO DE OBJETIVO: dejar de intentar ganarle a Pinnacle con
nuestro modelo, y empezar a usar a Pinnacle como REFERENCIA DE VALOR JUSTO
para detectar precios malos en las casas blandas (retail).

### Por que este cambio (medicion real, no intuicion)

El 2026-08-22 se midio por primera vez la pregunta de fondo del proyecto:
¿el blend Benter Boost (modelo v4 + mercado) le gana al mercado SOLO, fuera
de muestra? Resultado sobre 6,529 partidos OOS de las 4 ligas:

  Liga        n      Brier v4   Brier Pinnacle   Brier blend   peso optimo
  EPL       1730     0.588920      0.560801        0.568619       w=1.00
  LALIGA    1708     0.588734      0.572426        0.576078       w=0.96
  SERIEA    1718     0.600537      0.577595        0.584011       w=1.00
  BUNDES    1373     0.608798      0.576732        0.584806       w=1.00

El peso optimo del mercado es 1.00 -- es decir, **la mezcla optima ignora
por completo a nuestro modelo**. Y contra la cuota de APERTURA (la que uno
realmente puede tomar, no la de cierre) el peso optimo es 1.00 en las
CUATRO ligas, con mejora de 0.000%.

Conclusion honesta: v4 no aporta NINGUNA informacion que la linea de
Pinnacle no tenga ya. El blend al peso actual (w=0.515) esta empeorando la
estimacion respecto de simplemente leer el precio de Pinnacle. Todo "edge"
calculado contra esa estimacion es ruido.

Esto tambien explica los 5 intentos fallidos consecutivos de mejorar v4
(v5 ataque/defensa, v6 corners, v7 Elo, v8 xG, totales): el techo no
estaba en las features, estaba en que un Poisson sobre datos publicos no
puede superar al libro mas eficiente del mundo en el mercado 1X2 de las 5
grandes ligas.

### La via que SI tiene fundamento

No hace falta ganarle a nadie con un modelo propio. La asimetria real del
mercado es entre casas: Pinnacle es sharp (margen bajo, limites altos,
acepta ganadores), y las casas retail copian con retraso y con sesgos
comerciales. Si Pinnacle dice que algo vale 2.00 y una casa blanda lo paga
3.10, ese +EV existe **sin que nuestro modelo opine nada**.

Este script hace exactamente eso: desvig-ea a Pinnacle para obtener la
probabilidad justa, y la compara contra el precio de TODAS las demas casas
del feed (~40-50 por partido, ya confirmadas disponibles el 2026-08-22).

### El desvig NO es un detalle: es la mitad del resultado

La primera version usaba desvig PROPORCIONAL y reporto 129 oportunidades.
Al revisarlas aparecio una señal de alarma: **81% eran sobre tapados**
(prob < 25%) y solo 1 de 129 sobre un favorito, con el edge creciendo
cuanto mas tapado el resultado. Esa es la firma exacta del
favorite-longshot bias: el libro carga mas margen sobre los tapados, y el
desvig proporcional le saca a todos la misma fraccion, dejando la
probabilidad del tapado inflada -- y por lo tanto el edge inflado.

Medido sobre el feed real del 2026-08-22 (45 partidos x 53 casas):

  umbral    proporcional   power   Shin
  3%             129         32      41
  5%              54         14      24
  8%              26          2       6
  10%             18          0       0

**Las 18 oportunidades de edge >10% eran TODAS artefacto del desvig.** El
edge medio de las que pasaban el 3% con proporcional cae de 5.8% a 1.3%
con Shin.

Por eso el default es **Shin** (Shin 1993, estandar de la literatura para
esta correccion). `--devig proportional` queda disponible solo para poder
reproducir la comparacion, nunca para operar.

### Limitaciones reales, dichas de frente (no son menores)

1. **Las casas blandas limitan**: es el riesgo operativo central de esta
   estrategia, no un detalle. Una cuenta que gana consistentemente termina
   limitada o cerrada en semanas o meses. Es un negocio real pero con vida
   util por cuenta.
2. **Estas apuestas se pierden la mayoria de las veces.** La probabilidad
   justa media de las oportunidades que sobreviven a Shin es ~22%: se
   espera perder ~78% de ellas CON el sistema funcionando bien. La metrica
   que manda es el ROI acumulado y el CLV, jamas el porcentaje de aciertos.
3. **El precio tiene que estar disponible cuando lo tomas**: el feed es un
   snapshot. Una cuota que se ve +EV puede haber cambiado.
4. **Shin tampoco es la verdad**: es un modelo, mejor fundado que el
   proporcional pero igualmente un supuesto. La unica validacion real es
   medir CLV sobre apuestas registradas -- pendiente.
5. **Esto NO valida ni invalida a v4 como modelo de futbol** -- simplemente
   deja de depender de el para encontrar valor.

Uso:
    python -m src.evaluation.soft_book_edge --fetch
    python -m src.evaluation.soft_book_edge --file data/raw/THEODDSAPI/football_live_odds_latest.csv
    python -m src.evaluation.soft_book_edge --fetch --min-edge 0.05
    python -m src.evaluation.soft_book_edge --file <feed.csv> --devig proportional  # solo para comparar
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import RAW_DATA_DIR
from src.ingestion.theoddsapi_live_odds_loader import (
    fetch_upcoming_odds, ALL_KEYS, MARKETS_CON_TOTALES, discover_active_soccer)

SHARP_BOOK = "pinnacle"

# Mismas reglas de staking que economic_backtest.py -- se reusan tal cual.
# El umbral por defecto es MAS BAJO que el de economic_backtest (8%) a
# proposito: aquel media edge contra una estimacion propia ruidosa, donde
# hacia falta un margen grande para separar señal de ruido. Aca el edge se
# mide contra el precio de Pinnacle, que es una referencia mucho mas firme,
# asi que un 3% ya es accionable. No es "aflojar el filtro": es que la
# incertidumbre de la referencia cambio.
DEFAULT_MIN_EDGE = 0.03
KELLY_FRACTION = 0.10
MAX_STAKE_FRACTION = 0.05

# --- Dos filtros de cordura, ambos por hallazgos REALES del 2026-08-22 ----
#
# MIN_MINUTOS_ANTES: solo partidos que TODAVIA no arrancaron. En la primera
# corrida a 44 competencias, **268 de 370 oportunidades (72%) eran de
# partidos ya empezados**, con edge medio 108% y maximo 1101% (una casa
# pagaba 501.0 lo que Pinnacle tenia a 28.10, con el partido 1.6 horas en
# juego). Son lineas en vivo desactualizadas: la referencia pre-partido de
# Pinnacle ya no es valida, y de todos modos ese precio no se puede tomar.
# Envenenarian por completo la medicion de CLV.
# Las 102 restantes (no empezadas) daban edge medio 5.96% -- el rango creible.
MIN_MINUTOS_ANTES = 5

# MAX_EDGE_SANITY: un edge pre-partido de mas de 25% contra el libro mas
# sharp del mundo casi nunca es una oportunidad -- es linea suspendida,
# congelada, o un error de la fuente. Se descarta y se AVISA cuantas cayeron,
# en vez de dejarlas pasar como si fueran valor.
MAX_EDGE_SANITY = 0.25

# Casas que NO son objetivo de esta estrategia: exchanges (cobran comision
# sobre ganancias, el precio bruto no es comparable) y el propio Pinnacle
# (es la referencia, compararlo consigo mismo daria edge 0 por definicion).
EXCLUDED_BOOKS = {
    "pinnacle",
    "betfair_ex_uk", "betfair_ex_eu", "smarkets", "matchbook",  # exchanges
}


# --- Grupos de operador ---------------------------------------------------
# MEDIDO el 2026-08-22: 'unibet_se' y 'unibet_nl' devolvian precios
# IDENTICOS (10.00/10.00, 6.75/6.75) porque son la misma empresa (Kindred)
# operando bajo licencias distintas. Lo mismo 'betsson'/'nordicbet' (Betsson
# Group). Contarlas como oportunidades separadas INFLA el conteo: es un
# unico precio, no dos señales independientes. Ademas, en la practica es
# una sola cuenta que se puede limitar.
OPERATOR_GROUP = {
    **{b: "Kindred" for b in ["unibet_se", "unibet_nl", "unibet_uk", "unibet_fr",
                               "leovegas", "leovegas_se", "32red"]},
    **{b: "Betsson" for b in ["betsson", "nordicbet"]},
    **{b: "Entain" for b in ["ladbrokes_uk", "coral", "bwin"]},
    **{b: "Flutter" for b in ["paddypower", "betfair_sb_uk", "fanduel", "sportsbet"]},
    **{b: "WilliamHill" for b in ["williamhill", "williamhill_us"]},
    **{b: "BetOnline" for b in ["betonlineag", "lowvig", "betus", "betanysports"]},
}


def devig_proportional(prices: dict) -> dict:
    """Normalizacion proporcional. SESGADA -- ver devig() abajo. Se mantiene
    solo para poder comparar contra el metodo bueno, no para usarla."""
    raw = {k: 1.0 / v for k, v in prices.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def devig_power(prices: dict) -> dict:
    """Wheatcroft: prob_i = (1/cuota_i)^k, con k tal que sumen 1. Al elevar
    a k>1, los tapados se achican MAS que los favoritos -- que es
    exactamente la correccion del favorite-longshot bias."""
    names = list(prices)
    pi = np.array([1.0 / prices[n] for n in names])
    try:
        k = brentq(lambda k: np.sum(pi ** k) - 1.0, 0.5, 5.0)
    except ValueError:
        return devig_proportional(prices)
    return dict(zip(names, pi ** k))


def devig_shin(prices: dict) -> dict:
    """Shin (1993): modela una proporcion z de apostadores informados y
    deriva de ahi cuanto margen carga el libro sobre cada resultado. Es el
    estandar de la literatura para corregir el favorite-longshot bias."""
    names = list(prices)
    pi = np.array([1.0 / prices[n] for n in names])
    P = pi.sum()

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / P) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: probs(z).sum() - 1.0, 1e-9, 0.5)
    except ValueError:
        return devig_proportional(prices)
    return dict(zip(names, probs(z)))


DEVIG_METHODS = {
    "shin": devig_shin,
    "power": devig_power,
    "proportional": devig_proportional,
}


def _filtrar_ya_empezados(raw: pd.DataFrame) -> pd.DataFrame:
    """Descarta partidos cuyo arranque ya paso (o esta a menos de
    MIN_MINUTOS_ANTES). Ver MIN_MINUTOS_ANTES para el hallazgo que lo
    motiva -- sin esto, el 72% de lo detectado es basura de linea en vivo."""
    if "commence_time" not in raw.columns:
        return raw
    inicio = pd.to_datetime(raw["commence_time"], utc=True, errors="coerce")
    corte = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=MIN_MINUTOS_ANTES)
    vale = inicio.notna() & (inicio > corte)
    n_fuera = int((~vale).sum())
    if n_fuera:
        print(f"  [FILTRO] {n_fuera} filas descartadas por partido ya empezado "
              f"o a menos de {MIN_MINUTOS_ANTES} min del arranque (linea en vivo, "
              f"no es valor pre-partido).")
    return raw[vale]


def find_edges(raw: pd.DataFrame, min_edge: float, metodo: str = "shin",
               dedupe_operator: bool = True) -> pd.DataFrame:
    """Para cada partido: desvig-ea Pinnacle y busca en el resto de las
    casas precios que superen esa probabilidad justa.

    Soporta h2h (3 vias) y totals (2 vias). CLAVE en totales: se agrupa por
    (partido, LINEA), porque un Over 2.5 y un Over 3.0 son apuestas
    DISTINTAS -- compararlas entre si daria un edge inventado. Solo se
    compara contra la MISMA linea que cotiza Pinnacle."""
    if metodo not in DEVIG_METHODS:
        raise ValueError(f"metodo de desvig desconocido: {metodo} (opciones: {list(DEVIG_METHODS)})")
    if "outcome_point" not in raw.columns:
        raw = raw.assign(outcome_point=np.nan)
    raw = _filtrar_ya_empezados(raw)
    datos = raw[raw["market"].isin(["h2h", "totals"])].copy()
    # La linea entra en la clave de agrupacion. Para h2h queda constante.
    datos["_linea"] = datos["outcome_point"].fillna(-999.0)
    filas = []
    descartadas_absurdas = []

    for (event_id, mercado, linea), grp in datos.groupby(["event_id", "market", "_linea"]):
        home = grp["home_team"].iloc[0]
        away = grp["away_team"].iloc[0]
        liga = grp["league"].iloc[0]
        inicio = grp["commence_time"].iloc[0]
        sharp = grp[grp["bookmaker"] == SHARP_BOOK]
        precios_sharp = dict(zip(sharp["outcome_name"], sharp["outcome_price_decimal"]))

        # NUMERO DE VIAS INFERIDO DEL DATO, no hardcodeado (2026-08-25).
        # Antes era "3 si es h2h, si no 2", lo cual es correcto SOLO para
        # futbol. En NBA y tenis el h2h tiene 2 vias (no hay empate), asi
        # que la regla vieja habria descartado en silencio TODOS los
        # partidos de esos deportes -- un fallo mudo, sin error ni aviso.
        # Se toma la cantidad de resultados distintos que cotiza el mercado
        # completo y se exige que el libro sharp los tenga todos.
        n_vias = grp["outcome_name"].nunique()
        if n_vias < 2:
            continue
        # Sin TODAS las vias del libro sharp no hay referencia -- se salta,
        # no se estima la faltante.
        if len(precios_sharp) != n_vias:
            continue
        justas = DEVIG_METHODS[metodo](precios_sharp)
        justas_prop = devig_proportional(precios_sharp)  # solo para reportar la diferencia

        for _, r in grp.iterrows():
            book = r["bookmaker"]
            if book in EXCLUDED_BOOKS:
                continue
            resultado = r["outcome_name"]
            cuota = r["outcome_price_decimal"]
            if resultado not in justas or pd.isna(cuota):
                continue
            p_justa = justas[resultado]
            edge = p_justa * cuota - 1.0
            if edge < min_edge:
                continue
            if edge > MAX_EDGE_SANITY:
                descartadas_absurdas.append((f"{home} vs {away}", resultado, book, cuota, edge))
                continue
            kelly_full = (p_justa * cuota - 1.0) / (cuota - 1.0)
            stake = min(max(kelly_full * KELLY_FRACTION, 0.0), MAX_STAKE_FRACTION)
            # La linea forma parte de la IDENTIDAD de la apuesta en totales:
            # "Over" a secas seria ambiguo entre 2.5 y 3.0.
            etiqueta = resultado if mercado == "h2h" else f"{resultado} {linea:g}"
            filas.append({
                "league": liga, "commence_time": inicio,
                "match": f"{home} vs {away}", "market": mercado,
                "line": (np.nan if mercado == "h2h" else linea),
                "outcome": etiqueta,
                "book": book, "operator": OPERATOR_GROUP.get(book, book),
                "book_odds": cuota,
                "pinnacle_odds": precios_sharp[resultado],
                "fair_prob": p_justa, "edge": edge,
                "edge_proporcional": justas_prop[resultado] * cuota - 1.0,
                "kelly_stake_frac": stake,
            })

    if descartadas_absurdas:
        print(f"  [CORDURA] {len(descartadas_absurdas)} descartadas por edge > "
              f"{MAX_EDGE_SANITY:.0%} contra Pinnacle -- casi seguro linea suspendida "
              f"o error de la fuente, no valor. Ejemplos:")
        for m, o, b, c, e in sorted(descartadas_absurdas, key=lambda x: -x[4])[:3]:
            print(f"     {m} / {o} @ {b} cuota {c:.2f} -> edge {e:.0%}")

    columnas = ["league", "commence_time", "match", "market", "line", "outcome",
                "book", "operator", "book_odds", "pinnacle_odds", "fair_prob",
                "edge", "edge_proporcional", "kelly_stake_frac"]
    if not filas:
        # DataFrame vacio PERO con columnas -- que no haya oportunidades es un
        # resultado legitimo y frecuente, no un error; devolver un frame sin
        # columnas hace explotar a cualquiera que lo consuma.
        return pd.DataFrame(columns=columnas)
    out = pd.DataFrame(filas)[columnas].sort_values("edge", ascending=False)
    if dedupe_operator:
        # Un mismo precio en 2 marcas del mismo grupo es UNA oportunidad.
        out = out.drop_duplicates(["match", "outcome", "operator"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Bajar cuotas frescas de las 4 ligas")
    ap.add_argument("--file", type=str, default=None, help="Usar un CSV de feed ya guardado")
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--devig", choices=list(DEVIG_METHODS), default="shin",
                    help="Metodo de desvig. Default 'shin' -- 'proportional' esta SESGADO "
                         "hacia tapados, usar solo para comparar.")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="No deduplicar marcas del mismo grupo operador (infla el conteo).")
    args = ap.parse_args()

    if args.fetch:
        dfs = []
        ligas = discover_active_soccer(excluir_femenino=True)
        # Iterar los VALORES (sport_key), no las claves -- ver el mismo bug
        # documentado en clv_tracker._bajar_feed().
        claves = list(ligas.values())
        print(f"Competencias activas descubiertas: {len(claves)}")
        for liga in claves:
            try:
                d = fetch_upcoming_odds(liga, markets=MARKETS_CON_TOTALES)
                if not d.empty:
                    dfs.append(d)
            except Exception as e:
                print(f"[ERROR] {liga}: {e}")
        if not dfs:
            print("[AVISO] No se bajo nada.")
            return
        raw = pd.concat(dfs, ignore_index=True)
        out_dir = RAW_DATA_DIR / "THEODDSAPI"
        out_dir.mkdir(parents=True, exist_ok=True)
        snap = out_dir / "football_live_odds_latest.csv"
        raw.to_csv(snap, index=False)
        print(f"\nFeed guardado -> {snap} ({len(raw)} filas)")
    elif args.file:
        raw = pd.read_csv(args.file)
    else:
        ap.print_help()
        return

    n_books = raw["bookmaker"].nunique()
    n_events = raw["event_id"].nunique()
    print(f"\nAnalizando {n_events} partidos x {n_books} casas "
          f"(referencia: {SHARP_BOOK}, desvig: {args.devig}, exchanges excluidos)")

    # Comparacion explicita entre metodos -- para que nunca se pierda de vista
    # cuanto del "edge" es artefacto del desvig. Medido el 2026-08-22 sobre el
    # feed real: 129 oportunidades con proporcional -> 41 con Shin, y las 18
    # de edge >10% se cayeron TODAS.
    comp = {m: len(find_edges(raw, args.min_edge, m, dedupe_operator=False)) for m in DEVIG_METHODS}
    print(f"  Sensibilidad al metodo de desvig (sin deduplicar): " +
          " | ".join(f"{m}={n}" for m, n in comp.items()))

    edges = find_edges(raw, args.min_edge, args.devig, dedupe_operator=not args.no_dedupe)
    if edges.empty:
        print(f"\nNinguna cuota supera el umbral de edge de {args.min_edge:.1%}. "
              f"Resultado legitimo -- significa que hoy las casas blandas estan alineadas "
              f"con Pinnacle en estos partidos.")
        return

    print(f"\n{len(edges)} oportunidades DISTINTAS con edge > {args.min_edge:.1%} "
          f"(top {min(args.top, len(edges))}):\n")
    print(f"{'liga':<11}{'partido':<40}{'apuesta':<24}{'operador':<14}"
          f"{'cuota':>7}{'pinn':>7}{'edge':>7}{'prob':>7}{'stake':>7}")
    print("-" * 131)
    for _, r in edges.head(args.top).iterrows():
        print(f"{r['league']:<11}{r['match'][:39]:<40}{r['outcome'][:23]:<24}{r['operator'][:13]:<14}"
              f"{r['book_odds']:>7.2f}{r['pinnacle_odds']:>7.2f}{r['edge']:>6.1%}"
              f"{r['fair_prob']:>7.1%}{r['kelly_stake_frac']:>7.2%}")

    # La probabilidad media importa MAS que el edge: dice cuantas de estas
    # apuestas se pierden aunque el sistema funcione perfecto.
    p = edges["fair_prob"].mean()
    print(f"\nProbabilidad justa MEDIA de estas apuestas: {p:.1%} -- se espera PERDER "
          f"~{1-p:.0%} de ellas aun con el sistema funcionando bien. "
          f"La metrica que manda es el ROI acumulado, nunca el % de aciertos.")

    print(f"\nResumen por casa (cuantas oportunidades aporta cada una):")
    print(edges["operator"].value_counts().head(12).to_string())

    out_dir = RAW_DATA_DIR / "THEODDSAPI"
    out_path = out_dir / "soft_book_edges_latest.csv"
    edges.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print(f"\n[RECORDATORIO] Ver limitaciones en el docstring: desvig proporcional sesga a "
          f"favoritos, las casas blandas limitan cuentas ganadoras, y el precio puede haberse "
          f"movido desde el snapshot.")


if __name__ == "__main__":
    main()
