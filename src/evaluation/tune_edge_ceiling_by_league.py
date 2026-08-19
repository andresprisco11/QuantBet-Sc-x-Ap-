"""
Fase 8, siguiente paso tras el techo de edge por liga (economic_backtest.py,
CONFIRMADO 2026-08-19 con Kelly real en La Liga/Serie A/Bundesliga -- ver
roadmap). Cierra una advertencia que el propio proyecto viene repitiendo
desde que se parametrizo por liga: `min_edge_threshold` y `kelly_fraction`
se tunearon EXCLUSIVAMENTE sobre EPL en Fase 3.5 (tune_staking_rules.py,
24 combinaciones), y se usan como default global "a validar, no un hecho"
-- la advertencia esta literal en el docstring de economic_backtest.py.

El techo de edge que se adopto ayer (max_edge<=25%/25%/30%) se probo
sosteniendo min_edge=8% y kelly=10% FIJOS (los valores de EPL) -- nunca se
verifico si, ahora que existe un techo, el piso de edge o la fraccion de
Kelly optimos para La Liga/Serie A/Bundesliga siguen siendo esos mismos
valores de EPL, o si cambian una vez que la cola de edge alto ya no forma
parte de la seleccion. Es la misma logica de "no asumir que una regla
tuneada en una liga transfiere a otra" que ya goberno todo Fase 8 -- ahora
aplicada a la interaccion entre el techo nuevo y el resto de la regla.

Pregunta que responde este script: para LALIGA/SERIEA/BUNDESLIGA (las 3
ligas con techo adoptado -- EPL no entra, ya se confirmo que el techo la
perjudica), ¿cual es la combinacion de (min_edge_threshold, kelly_fraction,
max_edge_ceiling) con mejor ROI real (Kelly, no flat), sosteniendo
max_odds=3.0 fijo (ya establecido como optimo en las 4 ligas europeas sin
excepcion desde Fase 3.5/8 -- no se vuelve a cuestionar aca)?

Reutiliza EXACTAMENTE la logica ya confirmada de economic_backtest.py
(_select_bets, _simulate_bankroll, load_eval_df) importandola directamente
-- no se reimplementa Kelly ni la seleccion de apuestas desde cero, para
no arriesgar una discrepancia silenciosa entre este script y el que ya
esta en produccion.

No se logguea en el sistema de tracking (barrido exploratorio) -- mismo
criterio que tune_staking_rules.py original.

Salida: data/runs/tune_edge_ceiling_by_league.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.evaluation.economic_backtest import (
    _select_bets, _simulate_bankroll, load_eval_df, INITIAL_BANKROLL, MAX_STAKE_FRACTION,
    MAX_EDGE_CEILING_BY_LEAGUE,
)

LEAGUES_TO_TUNE = ["LALIGA", "SERIEA", "BUNDESLIGA"]  # EPL excluida -- el techo ya se confirmo que la perjudica.

MAX_ODDS = 3.0  # fijo -- establecido como optimo sin excepcion desde Fase 3.5/8, no se re-cuestiona aca.

MIN_EDGE_GRID = [0.06, 0.08, 0.10, 0.12, 0.15]
KELLY_FRACTION_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
MAX_EDGE_CEILING_GRID = [None, 0.20, 0.25, 0.30, 0.35, 0.40]

MIN_N_BETS_RELIABLE = 80  # por debajo de esto, no se reporta como candidato -- mismo umbral de cautela
                           # que el proyecto ya aplico con La Liga n=28 en Tier 1 (ahi para un umbral de
                           # probabilidad, aca para volumen de apuestas -- misma logica de "muestra chica").

# FIX 2026-08-19: la version anterior comparaba contra el primer techo no-None
# de la grilla (siempre 20%, por el orden de MAX_EDGE_CEILING_GRID) en vez del
# techo REALMENTE adoptado por liga -- daba una "regla vigente" incorrecta
# para las 3 ligas. Se importa MAX_EDGE_CEILING_BY_LEAGUE directamente de
# economic_backtest.py para que esto nunca se desincronice de lo que esta en
# produccion.
MAX_DRAWDOWN_CONSERVADOR = 0.35  # tope para el ranking "conservador" -- ver FIX de drawdown abajo.


def _run_combo(df_eval: pd.DataFrame, min_edge: float, kelly_fraction: float, ceiling: float) -> dict:
    bets = _select_bets(df_eval, min_edge_threshold=min_edge, max_odds=MAX_ODDS, max_edge_ceiling=ceiling)
    n_bets = len(bets)
    if n_bets == 0:
        return {"n_bets": 0}

    bets = _simulate_bankroll(bets, kelly_fraction=kelly_fraction, max_stake_fraction=MAX_STAKE_FRACTION,
                               initial_bankroll=INITIAL_BANKROLL)
    total_staked = bets["stake"].sum()
    total_profit = bets["profit"].sum()
    roi = total_profit / total_staked if total_staked > 0 else float("nan")

    return {
        "n_bets": n_bets,
        "roi": roi,
        "final_bankroll": bets["bankroll_after"].iloc[-1],
        "max_drawdown": bets["drawdown"].max(),
        "win_rate": bets["won"].mean(),
    }


def tune_league(league_key: str) -> list:
    print(f"\n=== {league_key} ===")
    try:
        df_eval = load_eval_df(league_key)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return []

    rows = []
    for min_edge in MIN_EDGE_GRID:
        for kelly_fraction in KELLY_FRACTION_GRID:
            for ceiling in MAX_EDGE_CEILING_GRID:
                result = _run_combo(df_eval, min_edge, kelly_fraction, ceiling)
                if result["n_bets"] == 0:
                    continue
                rows.append({
                    "league_key": league_key,
                    "min_edge_threshold": min_edge,
                    "kelly_fraction": kelly_fraction,
                    "max_edge_ceiling": "sin techo" if ceiling is None else f"{ceiling:.0%}",
                    **result,
                })

    grid_df = pd.DataFrame(rows)
    reliable = grid_df[grid_df["n_bets"] >= MIN_N_BETS_RELIABLE].copy()
    if reliable.empty:
        print(f"[AVISO] {league_key}: ninguna combinacion alcanzo n>={MIN_N_BETS_RELIABLE} apuestas -- "
              f"no hay candidato confiable en esta grilla.")
        return rows

    reliable = reliable.sort_values("roi", ascending=False)

    # FIX: comparar contra el techo REALMENTE adoptado para esta liga, no
    # contra el primer techo de la grilla.
    adopted_ceiling = MAX_EDGE_CEILING_BY_LEAGUE.get(league_key)
    adopted_ceiling_label = "sin techo" if adopted_ceiling is None else f"{adopted_ceiling:.0%}"
    current_row = grid_df[
        (grid_df["min_edge_threshold"] == 0.08) & (grid_df["kelly_fraction"] == 0.10)
        & (grid_df["max_edge_ceiling"] == adopted_ceiling_label)
    ]
    current_roi = current_row["roi"].iloc[0] if not current_row.empty else None
    current_drawdown = current_row["max_drawdown"].iloc[0] if not current_row.empty else None

    print(f"Combinaciones probadas: {len(grid_df)}  |  con n>={MIN_N_BETS_RELIABLE} (confiables): {len(reliable)}")
    if current_roi is not None:
        print(f"Regla vigente hoy (min_edge=8%, kelly=10%, techo={adopted_ceiling_label}): "
              f"ROI {current_roi:.2%}, drawdown {current_drawdown:.2%}")
    else:
        print(f"[AVISO] no se encontro la combinacion exacta de la regla vigente (techo={adopted_ceiling_label}) "
              f"en la grilla -- revisar MIN_EDGE_GRID/KELLY_FRACTION_GRID.")

    print(f"\nTop 5 por ROI puro (n>={MIN_N_BETS_RELIABLE}) -- OJO: no filtra por riesgo, ver aviso abajo:")
    top5 = reliable.head(5)[["min_edge_threshold", "kelly_fraction", "max_edge_ceiling", "n_bets",
                              "roi", "max_drawdown", "win_rate"]]
    with pd.option_context("display.width", 160):
        print(top5.to_string(index=False,
              formatters={"min_edge_threshold": "{:.0%}".format, "kelly_fraction": "{:.0%}".format,
                          "roi": "{:.2%}".format, "max_drawdown": "{:.2%}".format, "win_rate": "{:.2%}".format}))

    # FIX: el top 5 por ROI puro puede premiar combinaciones con drawdown
    # disparado (ej. kelly_fraction alto) que en la practica nadie operaria.
    # Se agrega un segundo ranking que primero descarta drawdown > 35% y
    # recien ahi ordena por ROI -- para no repetir ese error de lectura.
    conservative = reliable[reliable["max_drawdown"] <= MAX_DRAWDOWN_CONSERVADOR]
    print(f"\nTop 5 conservador (drawdown <= {MAX_DRAWDOWN_CONSERVADOR:.0%}, ordenado por ROI dentro de ese filtro):")
    if conservative.empty:
        print(f"  [AVISO] ninguna combinacion con n>={MIN_N_BETS_RELIABLE} se mantiene bajo "
              f"{MAX_DRAWDOWN_CONSERVADOR:.0%} de drawdown -- no hay candidato conservador en esta grilla.")
    else:
        top5_conservative = conservative.head(5)[["min_edge_threshold", "kelly_fraction", "max_edge_ceiling",
                                                    "n_bets", "roi", "max_drawdown", "win_rate"]]
        with pd.option_context("display.width", 160):
            print(top5_conservative.to_string(index=False,
                  formatters={"min_edge_threshold": "{:.0%}".format, "kelly_fraction": "{:.0%}".format,
                              "roi": "{:.2%}".format, "max_drawdown": "{:.2%}".format, "win_rate": "{:.2%}".format}))

    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES_TO_TUNE:
        all_rows.extend(tune_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo tunear ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "tune_edge_ceiling_by_league.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: usar el top 5 CONSERVADOR (drawdown <=35%) como candidato real, no el top 5 por ROI "
          "puro -- el de ROI puro puede estar premiando un kelly_fraction alto que dispara el drawdown a "
          "niveles no operables (visto en Serie A: mejor ROI puro con 70% de drawdown, el doble del actual). "
          "Si el top 5 conservador de alguna liga usa un min_edge o kelly_fraction distinto de 8%/10%, es "
          "evidencia real de que, con el techo de edge ya en juego, la regla optima para esa liga especifica "
          "cambio. Cuidado con sobreajustar de todas formas: son 150 combinaciones probadas sobre el MISMO "
          "set OOS walk-forward -- con esa cantidad de intentos, encontrar algo que mejora mucho por puro "
          "azar es esperable, no la excepcion. Lo que hace a un candidato mas creible no es que sea el numero "
          "1 de la tabla, sino que varias combinaciones VECINAS (mismo techo, distinto kelly; o techo cercano) "
          "tambien mejoren -- una meseta, no un pico aislado. Ninguna de estas combinaciones deberia tratarse "
          "como regla nueva de produccion sin, como minimo, confirmarla con datos que todavia no existian "
          "cuando se corrio este barrido (la proxima temporada, o paper trading) -- mismo estandar que ya "
          "se aplico a la decision del techo de edge original.")
    print("\nNo se loggea en el sistema de tracking (barrido exploratorio) -- mismo criterio que "
          "tune_staking_rules.py original.")


if __name__ == "__main__":
    run()