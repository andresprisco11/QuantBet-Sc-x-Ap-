"""
Fase 14 -- CLV tracker: el JUEZ del sistema.

### Por que esto es lo mas importante que tiene el proyecto ahora

Todo lo demas descansa sobre supuestos. El desvig de Shin es un modelo, no
la verdad. Que Pinnacle sea "el precio justo" es una hipotesis razonable
pero hipotesis al fin. Y el porcentaje de aciertos NO sirve para validar
nada: las apuestas que detecta soft_book_edge.py tienen ~22% de
probabilidad justa, o sea que se pierden ~78% de las veces incluso con el
sistema funcionando perfecto. Una racha de 40 apuestas perdidas seguidas es
perfectamente compatible con un edge real.

El CLV (Closing Line Value) es la unica metrica que resuelve esto sin
esperar años de muestra. La idea:

  Si tomaste una cuota de 6.75 y para cuando arranco el partido Pinnacle
  habia movido su linea a un valor justo de 6.10, capturaste valor: el
  mercado se movio EN TU DIRECCION. Eso pasa consistentemente solo si
  estabas comprando barato de verdad.

El CLV converge muchisimo mas rapido que el ROI porque no depende de que
la pelota entre: mide el PRECIO, no el resultado. Un jugador con CLV
positivo sostenido gana plata a largo plazo casi por definicion; uno con
CLV negativo esta perdiendo aunque venga de una racha ganadora afortunada.

### Como se mide aca, exactamente

Para cada apuesta registrada:
  - `odds_tomada`      = cuota que pagaba la casa blanda al detectarla
  - `p_cierre`         = probabilidad justa de Pinnacle cerca del arranque,
                         desvig-eada con Shin (mismo metodo que la deteccion)
  - `clv = odds_tomada * p_cierre - 1`

CLV > 0 significa que la apuesta seguia siendo +EV contra la linea de
cierre del libro mas sharp. Es el estandar de la industria.

### Limitacion honesta del "cierre" que usamos

No hay un endpoint de "cuota de cierre exacta" barato: se aproxima con el
ultimo snapshot de Pinnacle tomado antes del arranque. Cuanto mas cerca del
kickoff se corra `--update-closing`, mejor la aproximacion. Si el ultimo
snapshot quedo lejos del arranque, la fila queda marcada con
`horas_antes_cierre` alto para poder filtrarla despues -- no se oculta el
problema, se mide.

Uso:
    python -m src.tracking.clv_tracker --record          # detecta y registra
    python -m src.tracking.clv_tracker --update-closing  # correr cerca del kickoff
    python -m src.tracking.clv_tracker --report
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.ingestion.theoddsapi_live_odds_loader import (
    fetch_upcoming_odds, ALL_KEYS, MARKETS_CON_TOTALES, discover_active_soccer,
    discover_active)
from src.evaluation.soft_book_edge import find_edges, devig_shin, SHARP_BOOK
from src.tracking.run_logger import RUNS_DIR

CLV_LOG = RUNS_DIR / "clv_log.csv"

# Minutos durante los cuales un checkpoint de barrido interrumpido se
# considera reutilizable. Pasado ese tiempo las cuotas ya son viejas y
# conviene volver a bajarlas aunque cueste creditos.
CHECKPOINT_VALIDO_MIN = 30.0

# --- Casas donde el usuario PUEDE apostar de verdad ----------------------
# HALLAZGO CLAVE (2026-08-25): el usuario vive en Nueva York, y **76% de las
# oportunidades detectadas estaban en casas que no puede usar** (Unibet /
# Kindred, 1xBet, Coolbet dominan la deteccion pero no operan ahi).
# Sin esta distincion se pasa la semana validando un edge inalcanzable.
# El reporte ahora separa ambos grupos: el general responde "¿existe el
# fenomeno?" y el accesible responde "¿puedo capturarlo yo?".
BOOKS_ACCESIBLES = {
    # legales en NY
    "fanduel", "draftkings", "betmgm", "williamhill_us", "betrivers",
    "fanatics", "espnbet", "ballybet", "resortsworld",
    # offshore que aceptan clientes de EEUU (zona gris -- decision del usuario)
    "betonlineag", "lowvig", "betus", "bovada", "mybookieag",
    "betanysports", "gtbets", "everygame",
}

COLUMNAS = [
    "bet_id", "registrada_utc", "league", "commence_time", "match", "outcome",
    "book", "operator", "odds_tomada", "pinnacle_odds_apuesta", "fair_prob_apuesta",
    "edge_apuesta", "kelly_stake_frac",
    "pinnacle_odds_cierre", "fair_prob_cierre", "clv", "cierre_utc", "horas_antes_cierre",
    # --- Metricas de MOVIMIENTO (agregadas 2026-08-22) --------------------
    # Sin estas, el CLV es casi tautologico: clv = cuota * p_cierre - 1 y
    # edge = cuota * p_apuesta - 1, asi que si Pinnacle no mueve su linea,
    # clv == edge POR CONSTRUCCION. Un reporte de "CLV +10%" seria solo el
    # edge de entrada reflejado, no evidencia de haber comprado barato.
    #
    # movimiento_pinnacle = clv - edge  ->  ¿se movio el sharp hacia nosotros?
    # odds_casa_cierre                  ->  precio de LA MISMA casa al cierre
    # convergencia_casa                 ->  ¿corrigio la casa blanda su error?
    #   Esta ultima es la evidencia mas fuerte: si tomamos 6.75 donde Pinnacle
    #   valia 5.78 y la casa termina en 6.00, la casa esta ADMITIENDO que su
    #   precio estaba mal. Eso no se puede explicar por construccion.
    "odds_casa_cierre", "movimiento_pinnacle", "convergencia_casa",
]


def _ahora():
    return datetime.now(timezone.utc)


# Columnas que SIEMPRE deben ser texto. Sin esto hay un bug REAL, detectado
# probando el ciclo completo el 2026-08-22: una columna de texto guardada
# vacia vuelve del CSV como float64 (NaN), y al escribirle una fecha pandas
# tira TypeError. Se fuerza el dtype al cargar.
COLUMNAS_TEXTO = ["bet_id", "registrada_utc", "league", "commence_time", "match",
                  "outcome", "book", "operator", "cierre_utc"]


def _cargar_log() -> pd.DataFrame:
    if CLV_LOG.exists():
        df = pd.read_csv(CLV_LOG)
        for c in COLUMNAS_TEXTO:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        return df
    return pd.DataFrame(columns=COLUMNAS)


def _guardar_log(df: pd.DataFrame) -> None:
    CLV_LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLV_LOG, index=False)


# Deportes que se barren. Cambiar esta lista es TODO lo que hace falta para
# extender a NBA/tenis -- la deteccion sharp-vs-blandas no depende del
# deporte. Se deja en futbol hasta tener veredicto de CLV: sumar deportes
# ahora gastaria creditos sin acelerar la respuesta (ver roadmap 2026-08-23).
DEPORTES = ("futbol",)


def _bajar_feed(ligas=None, markets: str = MARKETS_CON_TOTALES, deportes=None) -> pd.DataFrame:
    """Baja el feed de TODAS las competencias configuradas (no solo las 4 con
    historico) y con h2h+totales. Esto ataca el cuello de botella real: el
    CLV necesita 100+ apuestas para decir algo, y con 4 ligas y solo 1X2 se
    juntaban ~16 por fin de semana."""
    # Checkpoint incremental REAL. Correccion de una promesa mal cumplida
    # (2026-08-25): la version anterior ESCRIBIA el checkpoint pero nunca lo
    # leia de vuelta, asi que no ahorraba ni un credito -- solo evitaba
    # perder los datos. Ahora si: si existe un checkpoint reciente (menos de
    # CHECKPOINT_VALIDO_MIN minutos), las ligas que ya figuran ahi NO se
    # vuelven a pedir. Un barrido cortado a mitad se retoma donde quedo.
    ckpt = RUNS_DIR / "_feed_checkpoint.csv"
    previo, ligas_ya = None, set()
    if ckpt.exists():
        edad_min = (_ahora().timestamp() - ckpt.stat().st_mtime) / 60.0
        if edad_min <= CHECKPOINT_VALIDO_MIN:
            try:
                previo = pd.read_csv(ckpt)
                ligas_ya = set(previo["league"].dropna().unique())
                print(f"[CHECKPOINT] Retomando un barrido cortado hace {edad_min:.0f} min: "
                      f"{len(ligas_ya)} liga(s) ya descargadas, no se vuelven a pagar.")
            except Exception:
                previo, ligas_ya = None, set()
        else:
            ckpt.unlink()   # viejo, no sirve
    dfs = [previo] if previo is not None else []
    ligas = ligas or discover_active(deportes or DEPORTES, excluir_femenino=True)
    # BUG REAL corregido (2026-08-22): discover_active_soccer() devuelve un
    # dict {NOMBRE: sport_key}, y iterar un dict en Python da las CLAVES
    # ('ARGENTINA_PRIMERA_DIVISION'), no los sport_key. Eso hacia fallar 43
    # de 44 competencias -- solo pasaba EPL, y de casualidad, porque 'EPL'
    # existe en ALL_KEYS. Hay que iterar los VALORES.
    claves = list(ligas.values()) if isinstance(ligas, dict) else list(ligas)
    print(f"Competencias activas descubiertas: {len(claves)}")
    ok = 0
    for liga in claves:
        if liga in ligas_ya:
            ok += 1
            continue
        try:
            d = fetch_upcoming_odds(liga, markets=markets)
            if not d.empty:
                dfs.append(d)
                ok += 1
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                d.to_csv(ckpt, mode="a", header=not ckpt.exists(), index=False)
        except Exception as e:
            print(f"[ERROR] {liga}: {e}")
    print(f"Competencias con partidos y cuotas: {ok}/{len(claves)}")
    if ckpt.exists():
        ckpt.unlink()   # barrido completo -> ya no hace falta el checkpoint
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def registrar(min_edge: float) -> None:
    """Detecta oportunidades y las agrega al log. Idempotente: una misma
    (partido, resultado, casa) no se registra dos veces, para que correrlo
    varias veces en el dia no infle la muestra."""
    raw = _bajar_feed()
    if raw.empty:
        print("[AVISO] feed vacio.")
        return

    edges = find_edges(raw, min_edge, metodo="shin", dedupe_operator=True)
    if edges.empty:
        print(f"Sin oportunidades > {min_edge:.1%} en esta corrida.")
        return

    log = _cargar_log()
    ya = set(log["bet_id"]) if not log.empty else set()
    ahora = _ahora().isoformat()

    nuevas = []
    for _, r in edges.iterrows():
        bet_id = f"{r['match']}|{r['outcome']}|{r['book']}".replace(",", ";")
        if bet_id in ya:
            continue
        nuevas.append({
            "bet_id": bet_id, "registrada_utc": ahora, "league": r["league"],
            "commence_time": r["commence_time"], "match": r["match"], "outcome": r["outcome"],
            "book": r["book"], "operator": r["operator"], "odds_tomada": r["book_odds"],
            "pinnacle_odds_apuesta": r["pinnacle_odds"], "fair_prob_apuesta": r["fair_prob"],
            "edge_apuesta": r["edge"], "kelly_stake_frac": r["kelly_stake_frac"],
            "pinnacle_odds_cierre": np.nan, "fair_prob_cierre": np.nan, "clv": np.nan,
            "cierre_utc": "", "horas_antes_cierre": np.nan,
            "odds_casa_cierre": np.nan, "movimiento_pinnacle": np.nan,
            "convergencia_casa": np.nan,
        })

    if not nuevas:
        print(f"{len(edges)} oportunidades detectadas, todas YA registradas antes. Log sin cambios.")
        return

    log = pd.concat([log, pd.DataFrame(nuevas)], ignore_index=True)[COLUMNAS]
    _guardar_log(log)
    print(f"{len(nuevas)} apuestas nuevas registradas ({len(edges)-len(nuevas)} ya estaban). "
          f"Total en el log: {len(log)}")
    print(f"-> {CLV_LOG}")


# Ventana por defecto para considerar un snapshot como "linea de cierre".
# DEFECTO DE DISEÑO corregido el 2026-08-23: actualizar_cierre() cerraba
# TODA apuesta que encontrara en el feed, sin importar cuanto faltaba para
# el partido. Resultado real: 122 filas cerradas a una mediana de 141.7
# HORAS antes del arranque (casi 6 dias). Eso no es una linea de cierre --
# es un segundo snapshot tomado antes de que el mercado haga ningun
# descubrimiento de precio, y mide ruido entre dos fotos casi identicas.
# Ahora solo se cierran las apuestas cuyo partido esta por empezar.
VENTANA_CIERRE_HORAS = 3.0


def actualizar_cierre(ventana_horas: float = VENTANA_CIERRE_HORAS) -> None:
    """Cierra SOLO las apuestas cuyo partido arranca dentro de
    `ventana_horas`. Las demas quedan pendientes a proposito, para que cada
    una se cierre cerca de SU propio kickoff -- que es lo que hace que el
    numero signifique algo. Correr este comando varias veces al dia."""
    log = _cargar_log()
    if log.empty:
        print("Log vacio, nada que actualizar.")
        return
    pend = log["clv"].isna()
    if not pend.any():
        print("No hay apuestas pendientes de cierre.")
        return

    # OPTIMIZACION DE CREDITOS (2026-08-23): no hace falta barrer las 45
    # competencias para cerrar apuestas. Solo se bajan las ligas que tienen
    # apuestas pendientes con partido DENTRO de la ventana (mas un margen).
    # Medido: un barrido completo cuesta 270 creditos; a 3 corridas diarias
    # el presupuesto mensual se agota en 9 dias. Filtrando por ligas
    # relevantes el costo cae a una fraccion.
    inicio_pend = pd.to_datetime(log.loc[pend, "commence_time"], utc=True, errors="coerce")
    horas_pend = (inicio_pend - _ahora()).dt.total_seconds() / 3600.0
    # Margen de 1h por encima de la ventana: si un partido esta a 3.5h, la
    # proxima corrida podria perderlo, asi que se incluye desde ya.
    relevantes = log.loc[pend].loc[(horas_pend > -1) & (horas_pend <= ventana_horas + 1)]
    ligas_necesarias = sorted(relevantes["league"].dropna().unique())

    if not ligas_necesarias:
        prox = horas_pend[horas_pend > 0].min()
        print(f"Ninguna apuesta pendiente arranca dentro de {ventana_horas}h. "
              f"No se gasta ni un credito.")
        if not pd.isna(prox):
            print(f"   El proximo partido pendiente arranca en {prox:.1f}h.")
        return

    print(f"Cerrando: {len(relevantes)} filas en {len(ligas_necesarias)} liga(s) "
          f"(en vez de barrer las 45). Costo ~{len(ligas_necesarias)*6} creditos.")
    raw = _bajar_feed(ligas=ligas_necesarias)
    if raw.empty:
        print("[AVISO] feed vacio, no se puede cerrar nada.")
        return

    # Probabilidad justa actual de Pinnacle por (partido, resultado).
    # DEBE cubrir h2h Y totals, y la etiqueta del resultado tiene que
    # construirse EXACTAMENTE igual que en find_edges() ("Over 2.5", no
    # "Over"), o las apuestas de totales nunca encontrarian su cierre y
    # quedarian pendientes para siempre.
    if "outcome_point" not in raw.columns:
        raw = raw.assign(outcome_point=np.nan)
    datos = raw[raw["market"].isin(["h2h", "totals"])].copy()
    datos["_linea"] = datos["outcome_point"].fillna(-999.0)

    justas, cuotas = {}, {}
    precios_casa = {}   # (partido, etiqueta, casa) -> cuota al cierre
    for (_, mercado, linea), g in datos.groupby(["event_id", "market", "_linea"]):
        home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
        sh = g[g["bookmaker"] == SHARP_BOOK]
        precios = dict(zip(sh["outcome_name"], sh["outcome_price_decimal"]))
        if len(precios) != (3 if mercado == "h2h" else 2):
            continue
        partido = f"{home} vs {away}"
        for k, v in devig_shin(precios).items():
            etiqueta = k if mercado == "h2h" else f"{k} {linea:g}"
            justas[(partido, etiqueta)] = v
            cuotas[(partido, etiqueta)] = precios[k]
        # Precio de cierre de TODAS las casas, no solo del sharp -- necesario
        # para medir si la casa blanda corrigio su propio error.
        for _, fila in g.iterrows():
            et = (fila["outcome_name"] if mercado == "h2h"
                  else f"{fila['outcome_name']} {linea:g}")
            precios_casa[(partido, et, fila["bookmaker"])] = fila["outcome_price_decimal"]

    ahora = _ahora()
    n = 0
    fuera_ventana = 0
    ya_empezados = 0
    for i in log.index[pend]:
        clave = (log.at[i, "match"], log.at[i, "outcome"])
        if clave not in justas:
            continue  # el partido ya no esta en el feed (arranco) -- queda pendiente
        # Solo cerrar si el partido esta por empezar. Ver VENTANA_CIERRE_HORAS.
        try:
            faltan = (pd.to_datetime(log.at[i, "commence_time"], utc=True)
                      - ahora).total_seconds() / 3600.0
        except Exception:
            faltan = np.nan
        # BUG REAL corregido (2026-08-23): se verificaba el limite SUPERIOR
        # (no cerrar demasiado pronto) pero NUNCA el inferior. Una apuesta
        # con faltan = -0.78 (partido 47 min en juego) pasaba el filtro y se
        # cerraba contra una linea EN VIVO. Caso concreto que lo delato:
        # Man City vs Bournemouth, cuota tomada 6.75, Pinnacle "cierre" 3.00,
        # CLV +113% -- un numero espectacular y completamente falso.
        # El cierre tiene que ser PRE-partido, siempre.
        if not np.isnan(faltan) and faltan < 0:
            ya_empezados += 1
            continue
        if not np.isnan(faltan) and faltan > ventana_horas:
            fuera_ventana += 1
            continue
        p_cierre = justas[clave]
        odds = float(log.at[i, "odds_tomada"])
        try:
            inicio = pd.to_datetime(log.at[i, "commence_time"], utc=True)
            horas = (inicio - ahora).total_seconds() / 3600.0
        except Exception:
            horas = np.nan
        log.at[i, "pinnacle_odds_cierre"] = cuotas[clave]
        log.at[i, "fair_prob_cierre"] = p_cierre
        log.at[i, "clv"] = odds * p_cierre - 1.0
        log.at[i, "cierre_utc"] = ahora.isoformat()
        log.at[i, "horas_antes_cierre"] = horas
        # Movimiento del sharp: cuanto mejoro (o empeoro) el valor respecto
        # del edge que teniamos al apostar.
        log.at[i, "movimiento_pinnacle"] = (odds * p_cierre - 1.0) - float(log.at[i, "edge_apuesta"])
        # Convergencia de la casa blanda hacia Pinnacle.
        cierre_casa = precios_casa.get((log.at[i, "match"], log.at[i, "outcome"],
                                        log.at[i, "book"]))
        if cierre_casa is not None and not pd.isna(cierre_casa):
            log.at[i, "odds_casa_cierre"] = cierre_casa
            # Positivo = la casa BAJO su cuota acercandose al valor justo,
            # es decir corrigio el error que le detectamos.
            log.at[i, "convergencia_casa"] = (odds - float(cierre_casa)) / odds
        n += 1

    _guardar_log(log)
    print(f"{n} apuestas cerradas (partido dentro de {ventana_horas}h). "
          f"{fuera_ventana} todavia lejos del arranque, quedan pendientes a proposito. "
          f"Pendientes totales: {int(log['clv'].isna().sum())}")
    if fuera_ventana:
        print(f"   -> volver a correr este comando mas cerca de esos partidos.")
    if ya_empezados:
        print(f"   [PERDIDAS] {ya_empezados} no se cerraron porque su partido YA arranco. "
              f"Un cierre contra linea en vivo no mide nada, asi que se descartan.")


def reporte(max_horas: float) -> None:
    """CLV agregado. `max_horas` filtra las que se cerraron demasiado lejos
    del arranque (aproximacion mala del cierre)."""
    log = _cargar_log()
    if log.empty:
        print("Log vacio.")
        return
    cerradas = log.dropna(subset=["clv"]).copy()
    print(f"Apuestas registradas: {len(log)} | con cierre medido: {len(cerradas)} "
          f"| pendientes: {len(log)-len(cerradas)}")
    if cerradas.empty:
        print("\nTodavia no hay ninguna cerrada. Correr --update-closing cerca del kickoff.")
        return

    buenas = cerradas[(cerradas["horas_antes_cierre"] >= 0) &
                      (cerradas["horas_antes_cierre"] <= max_horas)]
    print(f"Con cierre valido (0 a {max_horas}h antes del arranque): {len(buenas)}")
    acc = buenas[buenas["book"].isin(BOOKS_ACCESIBLES)]
    print(f"   de esas, en casas donde PODES apostar: {len(acc)} "
          f"({len(acc)/len(buenas)*100 if len(buenas) else 0:.0f}%)\n")

    for etiqueta, d in [(f"cierre valido -- TODAS las casas", buenas),
                        (f"cierre valido -- SOLO casas accesibles", acc)]:
        if d.empty:
            continue
        clv = d["clv"]
        print(f"--- {etiqueta} (n={len(d)}) ---")
        print(f"   CLV medio      : {clv.mean():+.2%}   <-- OJO: ver aviso abajo")
        print(f"   CLV mediano    : {clv.median():+.2%}")
        print(f"   % con CLV > 0  : {(clv > 0).mean():.1%}")
        print(f"   edge al apostar: {d['edge_apuesta'].mean():+.2%}")
        print(f"   [AVISO] Si Pinnacle no movio su linea, CLV == edge por CONSTRUCCION. "
              f"Un CLV\n           parecido al edge NO es evidencia -- es aritmetica. "
              f"Lo que informa es el movimiento.")
        # --- Correccion por AGRUPAMIENTO (pseudo-replicacion) --------------
        # Hallazgo real del 2026-08-22: 370 filas del log correspondian a solo
        # 67 pares (partido, resultado) distintos -- una misma discrepancia
        # aparecia hasta en 26 casas a la vez. Esas 26 filas NO son 26
        # observaciones independientes: comparten el MISMO cierre de Pinnacle,
        # asi que su CLV esta casi perfectamente correlacionado.
        # Tratarlas como independientes infla el t hasta 5x y haria "confirmar"
        # un edge que la muestra no soporta. Se promedia por cluster y el test
        # se hace sobre las medias de cluster -- el estandar para datos
        # agrupados.
        clusters = d.groupby(["match", "outcome"])["clv"].mean()
        n_ef = len(clusters)
        print(f"   observaciones independientes: {n_ef} pares (partido,resultado) "
              f"de {len(d)} filas -- {len(d)/n_ef:.1f} casas por apuesta")
        se = clusters.std(ddof=1) / np.sqrt(n_ef) if n_ef > 1 else np.nan
        if not np.isnan(se) and se > 1e-9:
            t = clusters.mean() / se
            print(f"   CLV medio por cluster: {clusters.mean():+.2%}  "
                  f"error estandar {se:.2%}  (t = {t:+.2f})")
            clv = clusters  # los veredictos se leen sobre la muestra efectiva
            if n_ef < 30:
                print(f"   [MUESTRA CHICA] con {n_ef} apuestas INDEPENDIENTES no se puede "
                      f"concluir nada todavia. Hacen falta 100+ pares distintos, no 100+ filas.")
            elif abs(t) < 2:
                print(f"   (CLV medio no se distingue de cero, pero eso NO es el test "
                      f"que importa -- ver abajo)")

        # ================= LO QUE DE VERDAD VALIDA EL MOTOR =================
        _veredicto_movimiento(d)
        print()

    if len(cerradas) >= 10:
        print("CLV medio por operador (donde conviene tener cuenta):")
        g = cerradas.groupby("operator")["clv"].agg(["size", "mean"]).sort_values("mean", ascending=False)
        g["mean"] = (g["mean"] * 100).round(2).astype(str) + "%"
        print(g.to_string())


def reabrir_tempranas(ventana_horas: float, aplicar: bool) -> None:
    """Borra el cierre de las apuestas que se cerraron DEMASIADO LEJOS del
    arranque, siempre que el partido todavia no se haya jugado.

    Recupera muestra en vez de perderla: un cierre tomado 6 dias antes no
    mide nada, pero si el partido aun no arranco se puede volver a cerrar
    bien mas adelante."""
    log = _cargar_log()
    if log.empty or "horas_antes_cierre" not in log.columns:
        print("Log vacio o sin datos de cierre.")
        return
    ahora = _ahora()
    inicio = pd.to_datetime(log["commence_time"], utc=True, errors="coerce")
    sin_jugar = inicio > ahora
    mal_cerrada = log["clv"].notna() & (log["horas_antes_cierre"] > ventana_horas)
    objetivo = sin_jugar & mal_cerrada

    print(f"Cerradas a mas de {ventana_horas}h del arranque Y todavia sin jugar: "
          f"{int(objetivo.sum())}")
    if objetivo.any():
        print(f"   se cerraron a una mediana de "
              f"{log.loc[objetivo,'horas_antes_cierre'].median():.0f}h del partido")
    ya_jugadas_mal = (~sin_jugar) & mal_cerrada
    if ya_jugadas_mal.any():
        print(f"[PERDIDAS] {int(ya_jugadas_mal.sum())} se cerraron mal y el partido YA se jugo "
              f"-- no se pueden recuperar.")
    if not aplicar:
        print("\n[SIMULACION] No se modifico nada. Usar --reabrir-aplicar para hacerlo.")
        return
    for col in ["pinnacle_odds_cierre", "fair_prob_cierre", "clv", "horas_antes_cierre",
                "odds_casa_cierre", "movimiento_pinnacle", "convergencia_casa"]:
        if col in log.columns:
            log.loc[objetivo, col] = np.nan
    log.loc[objetivo, "cierre_utc"] = ""
    _guardar_log(log)
    print(f"\n{int(objetivo.sum())} apuestas reabiertas. Volver a correr --update-closing "
          f"cerca del arranque de cada partido.")


def agenda(ventana_horas: float, tz_offset: float) -> None:
    """Responde la pregunta operativa que se repite TODOS los dias: ¿que
    partidos tengo pendientes y a que hora conviene correr el cierre?

    Existe porque resolverlo a mano requiere mirar el log y cruzar horarios,
    y una apuesta cuyo partido arranca sin haberse cerrado se pierde para
    siempre. El script propone horarios que maximizan cuantas apuestas
    entran en la ventana de cierre."""
    log = _cargar_log()
    if log.empty:
        print("Log vacio.")
        return
    ahora = _ahora()
    inicio = pd.to_datetime(log["commence_time"], utc=True, errors="coerce")
    pend = log["clv"].isna() & (inicio > ahora)
    d = log[pend].copy()
    if d.empty:
        print("No hay apuestas pendientes con partido por jugarse.")
        perdidas = int((log["clv"].isna() & (inicio <= ahora)).sum())
        if perdidas:
            print(f"[AVISO] {perdidas} quedaron sin cerrar y su partido ya arranco -- perdidas.")
        return

    d["inicio"] = inicio[pend]
    d["horas"] = (d["inicio"] - ahora).dt.total_seconds() / 3600.0
    d["local"] = (d["inicio"] + pd.Timedelta(hours=tz_offset)).dt.strftime("%a %d %I:%M %p")

    print(f"PENDIENTES: {len(d)} filas, {d.groupby(['match','outcome']).ngroups} apuestas "
          f"independientes\n")
    print(f"{'arranque (tu hora)':<22}{'liga':<38}{'filas':>6}{'en':>8}")
    print("-" * 76)
    for (loc, liga), g in d.groupby(["local", "league"], sort=False):
        h = g["horas"].iloc[0]
        print(f"{loc:<22}{str(liga)[:37]:<38}{len(g):>6}{h:>7.1f}h")

    # Horarios sugeridos: se recorre el dia en pasos de 30 min y se elige
    # greedy el momento que cierra mas apuestas todavia sin cubrir.
    hoy = d[d["horas"] <= 24].copy()
    print(f"\n{'='*76}")
    if hoy.empty:
        prox = d["horas"].min()
        print(f"Nada arranca en las proximas 24h. El proximo partido es en {prox:.0f}h "
              f"({d.loc[d['horas'].idxmin(),'local']}).")
        print("No hace falta correr --update-closing todavia: no gastaria creditos igual.")
        return

    print("HORARIOS SUGERIDOS para correr --update-closing (proximas 24h):\n")
    sin_cubrir = set(hoy.index)
    momentos = []
    for _ in range(6):
        if not sin_cubrir:
            break
        mejor, mejor_n = None, 0
        t = 0.0
        while t <= 24:
            cubre = {i for i in sin_cubrir if 0 < hoy.at[i, "horas"] - t <= ventana_horas}
            if len(cubre) > mejor_n:
                mejor, mejor_n = (t, cubre), len(cubre)
            t += 0.5
        if not mejor or mejor_n == 0:
            break
        t, cubre = mejor
        hora_local = (ahora + pd.Timedelta(hours=t + tz_offset)).strftime("%a %d %I:%M %p")
        ligas = hoy.loc[list(cubre), "league"].nunique()
        # El limite es el arranque MAS TEMPRANO del grupo: pasado ese momento
        # esa fila ya no se puede cerrar.
        t_limite = min(hoy.at[i, "horas"] for i in cubre)
        limite = (ahora + pd.Timedelta(hours=t_limite + tz_offset)).strftime("%I:%M %p")
        momentos.append((hora_local, len(cubre), ligas, limite))
        sin_cubrir -= cubre

    # Se muestra un RANGO, no una hora exacta. La ventana son `ventana_horas`
    # ANTES del arranque, asi que cualquier momento entre el inicio del rango
    # y el kickoff mas temprano del grupo sirve igual. Mostrar solo el punto
    # optimo generaba ansiedad innecesaria y corridas tarde por creer que
    # habia que dar en el clavo.
    for hora_local, n, ligas, limite in momentos:
        print(f"   entre {hora_local} y {limite}   cierra {n:>3} filas  (~{ligas*6} creditos)")
    print(f"\n   (cualquier momento DENTRO del rango sirve igual -- no hay que "
          f"acertar la hora exacta.\n    Pasado el limite, esas apuestas se pierden.)")
    if sin_cubrir:
        print(f"\n   [OJO] {len(sin_cubrir)} filas no entran en ningun horario razonable "
              f"-- arrancan muy pronto o muy dispersas.")


def _veredicto_movimiento(d: pd.DataFrame) -> None:
    """Los DOS tests que no se pueden satisfacer por construccion.

    1. MOVIMIENTO DE PINNACLE (clv - edge): ¿el sharp se movio hacia nuestra
       apuesta? Positivo = compramos antes de que el mercado reconociera el
       valor. Cero = Pinnacle no se movio (neutro). Negativo = estabamos
       leyendo precio rancio y el sharp nos corrigio.

    2. CONVERGENCIA DE LA CASA BLANDA: ¿la casa bajo su propia cuota hacia el
       valor justo antes del arranque? Esta es la evidencia mas fuerte que
       existe aca, porque es la casa ADMITIENDO que su precio estaba mal.
       No hay forma de que salga positiva por construccion."""
    print("\n   ### Los dos tests que NO se pueden cumplir por construccion ###")

    for col, nombre, explica in [
        ("movimiento_pinnacle", "MOVIMIENTO de Pinnacle (clv - edge)",
         "el sharp se movio hacia nuestra apuesta"),
        ("convergencia_casa", "CONVERGENCIA de la casa blanda",
         "la casa corrigio su propio precio hacia el valor justo"),
    ]:
        if col not in d.columns:
            print(f"   {nombre}: sin datos (log viejo, se llena desde la proxima corrida)")
            continue
        sub = d.dropna(subset=[col])
        if sub.empty:
            print(f"   {nombre}: sin datos todavia")
            continue
        cl = sub.groupby(["match", "outcome"])[col].mean()
        n = len(cl)
        se = cl.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        # Guardia: con varianza practicamente nula el t se dispara a valores
        # absurdos (1e16). Pasa con datos sinteticos o cuando todas las
        # apuestas comparten el mismo movimiento -- no es significancia real.
        if not np.isnan(se) and se < 1e-9:
            print(f"   {nombre}: {cl.mean():+.2%} (n={n}) -- varianza ~0, "
                  f"sin test posible (todas se movieron igual)")
            continue
        linea = f"   {nombre}: {cl.mean():+.2%} (n={n} independientes"
        if not np.isnan(se) and se > 0:
            t = cl.mean() / se
            linea += f", t={t:+.2f})"
        else:
            linea += ")"
        print(linea)
        if n < 20:
            print(f"      muestra chica ({n}), sin veredicto todavia.")
        elif not np.isnan(se) and se > 0:
            if t >= 2:
                print(f"      >>> POSITIVO Y SIGNIFICATIVO: {explica}. Esto SI es evidencia.")
            elif t <= -2:
                print(f"      >>> NEGATIVO Y SIGNIFICATIVO: el mercado se movio EN CONTRA. "
                      f"Señal de alarma, revisar antes de escalar.")
            else:
                print(f"      >>> Indistinguible de cero: no confirma ni desmiente.")


def limpiar(aplicar: bool) -> None:
    """Saca del log las apuestas que se registraron sobre partidos que YA
    habian empezado. Ver MIN_MINUTOS_ANTES en soft_book_edge: son lineas en
    vivo desactualizadas, no valor pre-partido, y arruinarian el CLV.

    Necesario porque el log acumulado antes del 2026-08-22 se lleno de
    ellas: 268 de 370 filas (72%), con edge medio 108% y maximo 1101%."""
    log = _cargar_log()
    if log.empty:
        print("Log vacio.")
        return
    inicio = pd.to_datetime(log["commence_time"], utc=True, errors="coerce")
    registro = pd.to_datetime(log["registrada_utc"], utc=True, errors="coerce")
    # Se compara contra el momento del REGISTRO, no contra ahora: la pregunta
    # es si el partido ya habia empezado cuando se detecto la oportunidad.
    mala = inicio.isna() | registro.isna() | (inicio <= registro)
    print(f"Filas totales: {len(log)}")
    print(f"Registradas sobre partido YA EMPEZADO: {int(mala.sum())} ({mala.mean():.0%})")
    if mala.any():
        print(f"   edge medio de esas    : {log.loc[mala,'edge_apuesta'].mean():.1%}")
        print(f"   edge maximo           : {log.loc[mala,'edge_apuesta'].max():.0%}")
    print(f"Filas que quedan (validas): {int((~mala).sum())}")
    if (~mala).any():
        print(f"   edge medio de esas    : {log.loc[~mala,'edge_apuesta'].mean():.2%}")
        pares = log[~mala].groupby(["match","outcome"]).ngroups
        print(f"   apuestas independientes: {pares} pares (partido,resultado)")

    if not aplicar:
        print("\n[SIMULACION] No se modifico nada. Correr con "
              "--limpiar-aplicar para borrarlas de verdad.")
        return
    _guardar_log(log[~mala].reset_index(drop=True))
    print(f"\nLog limpiado -> {CLV_LOG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--update-closing", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limpiar", action="store_true",
                    help="Ver cuantas filas del log son de partidos ya empezados (no borra).")
    ap.add_argument("--limpiar-aplicar", action="store_true",
                    help="Borrarlas de verdad.")
    ap.add_argument("--ventana", type=float, default=VENTANA_CIERRE_HORAS,
                    help="Horas antes del arranque dentro de las cuales se considera cierre.")
    ap.add_argument("--reabrir", action="store_true",
                    help="Ver cuantas se cerraron demasiado lejos del arranque (no modifica).")
    ap.add_argument("--agenda", action="store_true",
                    help="Que partidos hay pendientes y a que hora conviene cerrar. GRATIS.")
    ap.add_argument("--tz", type=float, default=-4.0,
                    help="Tu huso horario respecto de UTC (default -4).")
    ap.add_argument("--reabrir-aplicar", action="store_true",
                    help="Reabrirlas para volver a cerrarlas bien mas adelante.")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--max-horas", type=float, default=6.0)
    args = ap.parse_args()

    if args.agenda:
        agenda(args.ventana, args.tz)
    elif args.limpiar or args.limpiar_aplicar:
        limpiar(aplicar=args.limpiar_aplicar)
    elif args.record:
        registrar(args.min_edge)
    elif args.reabrir or args.reabrir_aplicar:
        reabrir_tempranas(args.ventana, aplicar=args.reabrir_aplicar)
    elif args.update_closing:
        actualizar_cierre(args.ventana)
    elif args.report:
        reporte(args.max_horas)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
