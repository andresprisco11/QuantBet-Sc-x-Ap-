"""
Escalamiento a Tenis, paso 6 (tras economic_backtest_tennis.py, corrido y
CONFIRMADO 2026-08-19 -- ver roadmap). La regla de staking prestada de
fútbol (min_edge=8%, kelly=10%, max_odds=3.0) seleccionó CASI NADA:
ATP 2 apuestas (de 20,837 partidos evaluados), WTA 25 (de 19,315) -- muy
por debajo del volumen que la misma regla produjo en cualquier liga
europea (EPL sola: 256 de ~1,730). Con n=2 en ATP, el ROI de -100% no es
una conclusión, es ruido de muestra -- 2 apuestas no prueban ni refutan
nada.

Este script NO asume que 8%/10%/3.0 sea siquiera un punto de partida
razonable para tenis -- barre un rango de min_edge MÁS BAJO que el usado
en fútbol (el blend de tenis está mucho más pegado al mercado que en
fútbol, ver tennis_logistic_model.py: gap blend-mercado de apenas
+0.0008/+0.0011 en Brier, contra +0.0037 a +0.0081 en las 4 ligas
europeas -- con un blend tan cerca del mercado, es esperable que muy
pocos partidos crucen un umbral de edge tan alto como 8%). También barre
max_odds más alto, porque tenis tiene mucha más dispersión de cuotas que
fútbol (partidos de primera ronda entre un top-10 y un jugador de
clasificación pueden pagar 8-10, cuotas que en fútbol serían atípicas).

Reutiliza EXACTAMENTE la lógica ya confirmada de economic_backtest_tennis.py
(_select_bets, _simulate_bankroll, load_predictions) importándola
directamente -- mismo criterio que tune_edge_ceiling_by_league.py en
fútbol, para no arriesgar una discrepancia silenciosa entre este script y
el que ya está en producción.

No se loggea en el sistema de tracking (barrido exploratorio) -- mismo
criterio que tune_staking_rules.py / tune_staking_rules_mls.py originales.

Salida: data/runs/tune_staking_rules_tennis_<tour>.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.evaluation.economic_backtest_tennis import (
    _select_bets, _simulate_bankroll, load_predictions, INITIAL_BANKROLL, MAX_STAKE_FRACTION,
)

TOURS = ["ATP", "WTA"]

MIN_EDGE_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
KELLY_FRACTION_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
MAX_ODDS_GRID = [2.5, 3.0, 4.0, 6.0]

MIN_N_BETS_RELIABLE = 30  # mas bajo que el de futbol (80) -- se sabe de entrada que el volumen
                           # de tenis con esta regla es mucho mas chico, ver docstring. Con menos
                           # de esto ni siquiera se reporta como candidato "confiable".
MAX_DRAWDOWN_CONSERVADOR = 0.35  # mismo criterio que tune_edge_ceiling_by_league.py en futbol


def _run_combo(df: pd.DataFrame, min_edge: float, kelly_fraction: float, max_odds: float) -> dict:
    bets = _select_bets(df, min_edge_threshold=min_edge, max_odds=max_odds)
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


def tune_tour(tour: str) -> list:
    print(f"\n=== {tour.upper()} ===")
    try:
        df = load_predictions(tour)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return []

    rows = []
    for min_edge in MIN_EDGE_GRID:
        for kelly_fraction in KELLY_FRACTION_GRID:
            for max_odds in MAX_ODDS_GRID:
                result = _run_combo(df, min_edge, kelly_fraction, max_odds)
                if result["n_bets"] == 0:
                    continue
                rows.append({
                    "tour": tour, "min_edge_threshold": min_edge, "kelly_fraction": kelly_fraction,
                    "max_odds": max_odds, **result,
                })

    grid_df = pd.DataFrame(rows)
    print(f"Combinaciones probadas: {len(MIN_EDGE_GRID) * len(KELLY_FRACTION_GRID) * len(MAX_ODDS_GRID)}  |  "
          f"con al menos 1 apuesta: {len(grid_df)}")

    reliable = grid_df[grid_df["n_bets"] >= MIN_N_BETS_RELIABLE].copy()
    if reliable.empty:
        print(f"[AVISO] {tour}: ninguna combinación alcanzó n>={MIN_N_BETS_RELIABLE} apuestas en TODA la grilla -- "
              f"el mercado de tenis parece muchísimo más eficiente que fútbol para este modelo, o el modelo casi "
              f"no genera edge detectable. Máximo n_bets encontrado en la grilla: "
              f"{int(grid_df['n_bets'].max()) if not grid_df.empty else 0}.")
        return rows

    reliable = reliable.sort_values("roi", ascending=False)
    print(f"Con n>={MIN_N_BETS_RELIABLE} (candidatos con volumen mínimo reportable): {len(reliable)}")

    print(f"\nTop 5 por ROI puro -- OJO: no filtra por riesgo, ver ranking conservador abajo:")
    top5 = reliable.head(5)[["min_edge_threshold", "kelly_fraction", "max_odds", "n_bets",
                              "roi", "max_drawdown", "win_rate"]]
    with pd.option_context("display.width", 160):
        print(top5.to_string(index=False,
              formatters={"min_edge_threshold": "{:.0%}".format, "kelly_fraction": "{:.0%}".format,
                          "roi": "{:.2%}".format, "max_drawdown": "{:.2%}".format, "win_rate": "{:.2%}".format}))

    conservative = reliable[reliable["max_drawdown"] <= MAX_DRAWDOWN_CONSERVADOR]
    print(f"\nTop 5 conservador (drawdown <= {MAX_DRAWDOWN_CONSERVADOR:.0%}, ordenado por ROI dentro de ese filtro):")
    if conservative.empty:
        print(f"  [AVISO] ninguna combinación con n>={MIN_N_BETS_RELIABLE} se mantiene bajo "
              f"{MAX_DRAWDOWN_CONSERVADOR:.0%} de drawdown.")
    else:
        top5_cons = conservative.head(5)[["min_edge_threshold", "kelly_fraction", "max_odds", "n_bets",
                                           "roi", "max_drawdown", "win_rate"]]
        with pd.option_context("display.width", 160):
            print(top5_cons.to_string(index=False,
                  formatters={"min_edge_threshold": "{:.0%}".format, "kelly_fraction": "{:.0%}".format,
                              "roi": "{:.2%}".format, "max_drawdown": "{:.2%}".format, "win_rate": "{:.2%}".format}))

    return rows


def run() -> None:
    all_rows = []
    for tour in TOURS:
        all_rows.extend(tune_tour(tour))

    if not all_rows:
        print("\n[AVISO] No se pudo tunear ningún tour.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "tune_staking_rules_tennis.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: esto es un barrido EXPLORATORIO de un mercado que el proyecto recién empieza a conocer --")
    print("no se adopta ninguna combinación como regla de producción solo por aparecer primera en esta tabla.")
    print(f"Son {len(MIN_EDGE_GRID) * len(KELLY_FRACTION_GRID) * len(MAX_ODDS_GRID)} combinaciones probadas sobre el")
    print("MISMO set OOS walk-forward -- con esa cantidad de intentos, encontrar algo que parece rentable por puro")
    print("azar es esperable, no la excepción. Un candidato es más creíble si varias combinaciones VECINAS (edge")
    print("cercano, kelly cercano) también rinden bien -- una meseta, no un pico aislado. Si ninguna combinación")
    print("alcanza volumen mínimo reportable, la conclusión honesta es que este modelo, con estas features, no")
    print("detecta un edge explotable contra Pinnacle en tenis todavía -- no que 'hay que forzar un umbral más")
    print("bajo hasta que aparezca algo': un umbral tan bajo que casi cualquier partido califica deja de medir")
    print("edge real y empieza a medir ruido de selección.")
    print("\nNo se loggea en el sistema de tracking (barrido exploratorio) -- mismo criterio que tune_staking_rules.py.")


if __name__ == "__main__":
    run()