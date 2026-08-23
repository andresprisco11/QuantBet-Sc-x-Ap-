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
    fetch_upcoming_odds, ALL_KEYS, MARKETS_CON_TOTALES, discover_active_soccer)
from src.evaluation.soft_book_edge import find_edges, devig_shin, SHARP_BOOK
from src.tracking.run_logger import RUNS_DIR

CLV_LOG = RUNS_DIR / "clv_log.csv"

COLUMNAS = [
    "bet_id", "registrada_utc", "league", "commence_time", "match", "outcome",
    "book", "operator", "odds_tomada", "pinnacle_odds_apuesta", "fair_prob_apuesta",
    "edge_apuesta", "kelly_stake_frac",
    "pinnacle_odds_cierre", "fair_prob_cierre", "clv", "cierre_utc", "horas_antes_cierre",
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


def _bajar_feed(ligas=None, markets: str = MARKETS_CON_TOTALES) -> pd.DataFrame:
    """Baja el feed de TODAS las competencias configuradas (no solo las 4 con
    historico) y con h2h+totales. Esto ataca el cuello de botella real: el
    CLV necesita 100+ apuestas para decir algo, y con 4 ligas y solo 1X2 se
    juntaban ~16 por fin de semana."""
    dfs = []
    ligas = ligas or discover_active_soccer(excluir_femenino=True)
    # BUG REAL corregido (2026-08-22): discover_active_soccer() devuelve un
    # dict {NOMBRE: sport_key}, y iterar un dict en Python da las CLAVES
    # ('ARGENTINA_PRIMERA_DIVISION'), no los sport_key. Eso hacia fallar 43
    # de 44 competencias -- solo pasaba EPL, y de casualidad, porque 'EPL'
    # existe en ALL_KEYS. Hay que iterar los VALORES.
    claves = list(ligas.values()) if isinstance(ligas, dict) else list(ligas)
    print(f"Competencias activas descubiertas: {len(claves)}")
    ok = 0
    for liga in claves:
        try:
            d = fetch_upcoming_odds(liga, markets=markets)
            if not d.empty:
                dfs.append(d)
                ok += 1
        except Exception as e:
            print(f"[ERROR] {liga}: {e}")
    print(f"Competencias con partidos y cuotas: {ok}/{len(claves)}")
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
        })

    if not nuevas:
        print(f"{len(edges)} oportunidades detectadas, todas YA registradas antes. Log sin cambios.")
        return

    log = pd.concat([log, pd.DataFrame(nuevas)], ignore_index=True)[COLUMNAS]
    _guardar_log(log)
    print(f"{len(nuevas)} apuestas nuevas registradas ({len(edges)-len(nuevas)} ya estaban). "
          f"Total en el log: {len(log)}")
    print(f"-> {CLV_LOG}")


def actualizar_cierre() -> None:
    """Toma el snapshot actual de Pinnacle y lo usa como aproximacion de la
    linea de cierre para las apuestas todavia pendientes. Correr LO MAS
    CERCA POSIBLE del arranque de los partidos."""
    log = _cargar_log()
    if log.empty:
        print("Log vacio, nada que actualizar.")
        return
    pend = log["clv"].isna()
    if not pend.any():
        print("No hay apuestas pendientes de cierre.")
        return

    raw = _bajar_feed()
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

    ahora = _ahora()
    n = 0
    for i in log.index[pend]:
        clave = (log.at[i, "match"], log.at[i, "outcome"])
        if clave not in justas:
            continue  # el partido ya no esta en el feed (arranco) -- queda pendiente
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
        n += 1

    _guardar_log(log)
    print(f"{n} apuestas actualizadas con linea de cierre. "
          f"Pendientes: {int(log['clv'].isna().sum())}")


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

    buenas = cerradas[cerradas["horas_antes_cierre"].abs() <= max_horas]
    print(f"Con cierre tomado a menos de {max_horas}h del arranque: {len(buenas)}\n")

    for etiqueta, d in [("TODAS las cerradas", cerradas), (f"solo <={max_horas}h", buenas)]:
        if d.empty:
            continue
        clv = d["clv"]
        print(f"--- {etiqueta} (n={len(d)}) ---")
        print(f"   CLV medio      : {clv.mean():+.2%}")
        print(f"   CLV mediano    : {clv.median():+.2%}")
        print(f"   % con CLV > 0  : {(clv > 0).mean():.1%}")
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
        if not np.isnan(se) and se > 0:
            t = clusters.mean() / se
            print(f"   CLV medio por cluster: {clusters.mean():+.2%}  "
                  f"error estandar {se:.2%}  (t = {t:+.2f})")
            clv = clusters  # los veredictos se leen sobre la muestra efectiva
            if n_ef < 30:
                print(f"   [MUESTRA CHICA] con {n_ef} apuestas INDEPENDIENTES no se puede "
                      f"concluir nada todavia. Hacen falta 100+ pares distintos, no 100+ filas.")
            elif abs(t) < 2:
                print(f"   VEREDICTO: CLV medio NO se distingue de cero.")
            elif t >= 2:
                print(f"   VEREDICTO: CLV positivo estadisticamente distinguible de cero. "
                      f"Es la señal de que el edge es real.")
            else:
                print(f"   VEREDICTO: CLV NEGATIVO significativo -- el sistema esta comprando caro. "
                      f"Revisar el desvig o la seleccion antes de seguir.")
        print()

    if len(cerradas) >= 10:
        print("CLV medio por operador (donde conviene tener cuenta):")
        g = cerradas.groupby("operator")["clv"].agg(["size", "mean"]).sort_values("mean", ascending=False)
        g["mean"] = (g["mean"] * 100).round(2).astype(str) + "%"
        print(g.to_string())


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
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--max-horas", type=float, default=6.0)
    args = ap.parse_args()

    if args.limpiar or args.limpiar_aplicar:
        limpiar(aplicar=args.limpiar_aplicar)
    elif args.record:
        registrar(args.min_edge)
    elif args.update_closing:
        actualizar_cierre()
    elif args.report:
        reporte(args.max_horas)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
