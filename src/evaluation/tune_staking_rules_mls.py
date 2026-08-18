"""
Fase 8b: barrido de reglas de staking para MLS -- mismo principio que
tune_staking_rules.py (24 combinaciones de min_edge/kelly_fraction/max_odds),
antes de concluir que el -4.92% de ROI de la primera corrida de
economic_backtest_mls.py es un resultado final. Igual que se hizo para
Serie A/Bundesliga/La Liga: no se ajusta el umbral a mano, se barre la
grilla completa y se elige con evidencia (o se documenta que NINGUNA
combinacion funciona, si es lo que sale).

Reutiliza _select_bets_mls/_simulate_bankroll/load_eval_df_mls de
economic_backtest_mls.py -- no entrena nada nuevo.

Recordatorio metodologico (ver economic_backtest_mls.py): la ejecucion es
a cuota de CIERRE (no hay apertura para MLS), asi que este barrido esta
respondiendo una pregunta mas dura que su equivalente en las 4 ligas
europeas -- "gana el modelo al precio ya eficiente", no "aprovecha una
linea temprana barata". Es esperable que el techo de ROI alcanzable ac
sea mas bajo. Documentar el resultado tal cual salga.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.evaluation.economic_backtest_mls import _select_bets_mls, load_eval_df_mls, LEAGUE_KEY
from src.evaluation.economic_backtest import _simulate_bankroll, INITIAL_BANKROLL

MIN_EDGE_GRID = [0.02, 0.05, 0.08, 0.12]
KELLY_FRACTION_GRID = [0.10, 0.25]
MAX_ODDS_GRID = [3.0, 5.0, None]
MIN_STAKE_FRACTION = 0.05


def run() -> None:
    print(f"\n=== {LEAGUE_KEY} ===")
    try:
        df_eval = load_eval_df_mls()
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    print(f"Partidos con blend disponible: {len(df_eval)}\n")

    results = []
    for min_edge in MIN_EDGE_GRID:
        for kelly_fraction in KELLY_FRACTION_GRID:
            for max_odds in MAX_ODDS_GRID:
                bets = _select_bets_mls(df_eval, min_edge_threshold=min_edge, max_odds=max_odds)
                n_bets = len(bets)
                if n_bets == 0:
                    results.append({
                        "min_edge": min_edge, "kelly_fraction": kelly_fraction,
                        "max_odds": max_odds if max_odds else "sin_tope",
                        "n_bets": 0, "final_bankroll_multiple": float("nan"),
                        "roi": float("nan"), "win_rate": float("nan"), "max_drawdown": float("nan"),
                    })
                    continue

                bets = _simulate_bankroll(bets, kelly_fraction=kelly_fraction,
                                           max_stake_fraction=MIN_STAKE_FRACTION,
                                           initial_bankroll=INITIAL_BANKROLL)
                final_bankroll = bets["bankroll_after"].iloc[-1]
                total_staked = bets["stake"].sum()
                total_profit = bets["profit"].sum()
                roi = total_profit / total_staked if total_staked > 0 else float("nan")

                results.append({
                    "min_edge": min_edge, "kelly_fraction": kelly_fraction,
                    "max_odds": max_odds if max_odds else "sin_tope",
                    "n_bets": n_bets,
                    "final_bankroll_multiple": final_bankroll / INITIAL_BANKROLL,
                    "roi": roi,
                    "win_rate": bets["won"].mean(),
                    "max_drawdown": bets["drawdown"].max(),
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("final_bankroll_multiple", ascending=False)

    print(f"=== [MLS] Barrido completo de reglas de staking (ordenado por bankroll final, mejor primero) ===")
    print(results_df.round(4).to_string(index=False))

    MIN_N_BETS_FOR_TRUST = 100
    robust = results_df[results_df["n_bets"] >= MIN_N_BETS_FOR_TRUST].copy()
    print(f"\n=== [MLS] Mismo barrido, filtrado a combinaciones con al menos {MIN_N_BETS_FOR_TRUST} "
          f"apuestas ===")
    print(robust.round(4).to_string(index=False))

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "staking_sweep_mls.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nGuardado barrido completo -> {out_path}")
    print("\nNo se loggea en el sistema de tracking (24 corridas de un mismo analisis, no 24 modelos "
          "distintos) -- mismo criterio que tune_staking_rules.py.")


if __name__ == "__main__":
    run()