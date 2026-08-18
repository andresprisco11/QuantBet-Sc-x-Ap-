"""
Fase 3.5d: barrido de reglas de staking sobre v4, en respuesta directa al
"winner's curse" confirmado por selection_bias_check.py -- el proceso de
apostar al resultado de mayor edge entre 3 por partido selecciona ruido,
no señal, en los 5 rangos de cuota medidos (mas severo en cuotas bajas y
altas, menos severo en 1.50-2.00, que fue el UNICO rango con ROI positivo
en la corrida original).

En vez de adivinar un umbral de edge/fraccion de Kelly/tope de cuota
nuevo, este script barre una grilla de combinaciones -- mismo principio
que tune_half_life.py -- y reporta bankroll final, ROI, drawdown y
cantidad de apuestas de cada una, para elegir la mejor CON evidencia.

No entrena nada nuevo: reutiliza _select_bets/_simulate_bankroll/
load_eval_df de economic_backtest.py (ya parametrizadas para esto),
sobre el mismo 'model_predictions_oos_walkforward_v4.csv'.

ADVERTENCIA METODOLOGICA (importante, léela): elegir la combinación
ganadora de ESTE MISMO barrido sobre el MISMO conjunto de partidos es, en
sí mismo, un ejercicio de ajustar un hiperparámetro sobre datos ya
vistos -- el mismo riesgo de sobreajuste que ya se documentó con
half-life (aunque ahí el rango de resultados fue chato, sin riesgo real).
Acá el resultado SÍ puede cambiar mucho entre combinaciones, así que la
combinación elegida debe tratarse como una regla de staking a validar con
más datos futuros (temporada en curso, o paper trading), no como un
resultado ya demostrado. Se prioriza deliberadamente robustez (que
funcione razonablemente en VARIAS combinaciones cercanas) por sobre el
punto óptimo puntual, para reducir ese riesgo.

Fix 2026-08-18 (Fase 8, multi-liga): run() estaba hardcodeado a un solo
conjunto de datos (EPL, via load_eval_df() sin parametro). Se parametriza
run(league_key) y se loopea sobre LEAGUES en __main__ -- la pregunta
metodologica que motiva este script (la regla optima de EPL, tuneada en
economic_backtest.py, no tiene por que transferir igual a otra liga con
distinta distribucion de cuotas) solo se puede responder corriendo el
barrido completo por separado en cada una, no asumiendo que el resultado
de EPL aplica en todos lados.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES
from src.evaluation.economic_backtest import _select_bets, _simulate_bankroll, load_eval_df, INITIAL_BANKROLL

MIN_EDGE_GRID = [0.02, 0.05, 0.08, 0.12]
KELLY_FRACTION_GRID = [0.10, 0.25]
MAX_ODDS_GRID = [3.0, 5.0, None]
MIN_STAKE_FRACTION = 0.05  # tope duro de stake maximo, igual al default de economic_backtest.py.


def run(league_key: str) -> None:
    print(f"\n=== {league_key} ===")
    try:
        df_eval = load_eval_df(league_key)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    print(f"Partidos con blend disponible: {len(df_eval)}\n")

    results = []
    for min_edge in MIN_EDGE_GRID:
        for kelly_fraction in KELLY_FRACTION_GRID:
            for max_odds in MAX_ODDS_GRID:
                bets = _select_bets(df_eval, min_edge_threshold=min_edge, max_odds=max_odds)
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

    print(f"=== [{league_key}] Barrido completo de reglas de staking (ordenado por bankroll final, mejor primero) ===")
    print(results_df.round(4).to_string(index=False))

    # Filtro de robustez: solo combinaciones con volumen suficiente para no ser puro ruido de muestra chica.
    MIN_N_BETS_FOR_TRUST = 100
    robust = results_df[results_df["n_bets"] >= MIN_N_BETS_FOR_TRUST].copy()
    print(f"\n=== [{league_key}] Mismo barrido, filtrado a combinaciones con al menos {MIN_N_BETS_FOR_TRUST} "
          f"apuestas (las demas son muestra demasiado chica para confiar) ===")
    print(robust.round(4).to_string(index=False))

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / f"staking_sweep_v4_{league_key}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nGuardado barrido completo -> {out_path}")
    print("\nNo se loggea en el sistema de tracking (son 24 corridas de un mismo analisis, no 24 modelos "
          "distintos) -- una vez elegida la combinacion final para esta liga, esa SI se corre con "
          "economic_backtest.py (actualizando sus constantes MIN_EDGE_THRESHOLD/KELLY_FRACTION/MAX_ODDS, "
          "o extendiendolas a un dict por liga si los optimos difieren) y esa corrida queda trazada normalmente.")


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)