"""
Fase 10 -- barrido de reglas de staking para NFL, mismo patron que
`tune_staking_rules.py` (futbol): la corrida diagnostica de
`economic_backtest_nfl.py` (min_edge=2%, kelly=25%, sin techo) salio muy
negativa (ROI -50.98%, drawdown 73.89%) y `nfl_selection_bias_check.py`
encontro la causa -- el acierto real CAE a medida que sube la
probabilidad/edge declarada por el modelo (sobreconfianza creciente, no
problema de seleccion). Mismo remedio que ya funciono para La Liga/Serie A/
Bundesliga en futbol: un TECHO de edge (max_edge) que descarta justamente
las apuestas de edge mas alto -- las que este diagnostico mostro que son
las MENOS confiables, no las mejores.

Barre min_edge x kelly_fraction x max_edge (grilla chica, sin refit del
modelo -- solo re-simula bankroll sobre las probabilidades ya calculadas en
spread_evaluation_v1.csv, mismo dato de entrada que economic_backtest_nfl.py).

Uso: python -m src.evaluation.tune_staking_rules_nfl
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.evaluation.economic_backtest_nfl import _select_bets, _simulate_bankroll, STARTING_BANKROLL

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "spread_evaluation_v1.csv"

MIN_EDGE_CANDIDATES = [0.02, 0.05, 0.08]
KELLY_CANDIDATES = [0.05, 0.10, 0.25]
MAX_EDGE_CANDIDATES = [None, 0.10, 0.15, 0.20]


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre 'python -m src.evaluation.backtest_nfl_spread' primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["gameday"] = pd.to_datetime(df["gameday"])

    results = []
    for min_edge in MIN_EDGE_CANDIDATES:
        for max_edge in MAX_EDGE_CANDIDATES:
            if max_edge is not None and max_edge <= min_edge:
                continue  # techo invalido si es menor o igual al piso
            bets = _select_bets(df, min_edge=min_edge, max_edge=max_edge)
            if bets.empty:
                continue
            for kelly_fraction in KELLY_CANDIDATES:
                res = _simulate_bankroll(bets, kelly_fraction=kelly_fraction, starting_bankroll=STARTING_BANKROLL)
                results.append({
                    "min_edge": min_edge,
                    "max_edge": max_edge if max_edge is not None else "sin techo",
                    "kelly_fraction": kelly_fraction,
                    "n_bets": res["n_bets"],
                    "win_rate": res["win_rate"],
                    "roi": res["roi"],
                    "max_drawdown": res["max_drawdown"],
                })

    results_df = pd.DataFrame(results).sort_values("roi", ascending=False)

    print(f"=== Barrido de staking NFL -- {len(results_df)} combinaciones evaluadas ===\n")
    print("Top 15 por ROI:")
    print(results_df.head(15).to_string(
        index=False,
        formatters={"win_rate": "{:.2%}".format, "roi": "{:+.2%}".format, "max_drawdown": "{:.2%}".format},
    ))

    print("\nPeores 5 por ROI (para ver el rango completo):")
    print(results_df.tail(5).to_string(
        index=False,
        formatters={"win_rate": "{:.2%}".format, "roi": "{:+.2%}".format, "max_drawdown": "{:.2%}".format},
    ))

    n_positive = int((results_df["roi"] > 0).sum())
    print(f"\nCombinaciones con ROI positivo: {n_positive} de {len(results_df)}.")

    out_path = PROCESSED_DATA_DIR.parent / "runs" / "staking_sweep_nfl.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\n[NOTA] Si ninguna combinacion da ROI positivo, el problema no es de sizing -- confirmaria que "
          "el modelo de un solo feature (elo_diff) necesita mas señal antes de que NFL sea viable "
          "economicamente, mismo diagnostico que ya se vio en Serie A/Bundesliga (el techo de edge mejora "
          "pero no siempre alcanza para dar vuelta el signo).")


if __name__ == "__main__":
    run()