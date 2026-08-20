"""
Fase 10 -- validacion directa de la meta de Tier 1 (80-82% de acierto real)
para NFL, mismo patron que `tier1_probability_validation.py` (futbol):
mide el acierto REAL en los umbrales de probabilidad mas altos que produce
el blend, no solo asume que porque el edge/staking ya funciona (ver
nfl_temporal_stability_check.py) el modelo tambien produce apuestas de alta
confianza.

**Diferencia esperada, real, del propio diseño del mercado de spread**: a
diferencia del moneyline de futbol (donde un favorito aplastante puede
llegar facil a 90%+), el spread esta DISEÑADO por la casa para que cubrir
sea ~50/50 (ver el chequeo de sanidad de backtest_nfl_spread.py: 48.88% de
cobertura real del local, casi exacto). Por construccion del propio
mercado, es estructuralmente mucho mas dificil que aparezcan apuestas de
80%+ de confianza real en cobertura de spread que en moneyline de futbol --
esto no es un fallo del modelo, es la naturaleza del mercado. Se mide
igual, sin asumir el resultado de antemano.

Usa la misma prediccion direccional por partido de nfl_selection_bias_check.py
(el lado que el blend prefiere, sin filtrar por edge de staking).

Uso: python -m src.evaluation.nfl_tier1_probability_validation
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "spread_evaluation_v1.csv"

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre backtest_nfl_spread.py primero.")
        return

    df = pd.read_csv(DATA_PATH)
    non_push = df.loc[~df["is_push"]].copy()

    home_favored = non_push["blend_prob_home_covers"] >= non_push["blend_prob_away_covers"]
    non_push["predicted_prob"] = np.where(
        home_favored, non_push["blend_prob_home_covers"], non_push["blend_prob_away_covers"]
    )
    non_push["actual_won"] = np.where(
        home_favored, non_push["home_covers"] == 1.0, non_push["home_covers"] == 0.0
    )

    print(f"Universo evaluado (sin push, con prediccion del blend): {len(non_push)} partidos.\n")
    print(f"{'umbral':>8} {'n':>6} {'prob_predicha_avg':>18} {'acierto_real':>14}")
    for threshold in THRESHOLDS:
        subset = non_push[non_push["predicted_prob"] >= threshold]
        n = len(subset)
        if n == 0:
            print(f"{threshold:>7.0%} {n:>6} {'--':>18} {'--':>14}")
            continue
        pred_avg = subset["predicted_prob"].mean()
        real_acc = subset["actual_won"].mean()
        print(f"{threshold:>7.0%} {n:>6} {pred_avg:>17.2%} {real_acc:>13.2%}")

    print("\n[LECTURA] Comparar contra la meta de Tier 1 (80-82%, instrucciones originales del proyecto) "
          "y contra el gate real de capital (99% sobre >=2000 apuestas, ver roadmap). El spread esta "
          "diseñado por la casa para ser ~50/50 -- es estructuralmente mas dificil llegar a umbrales altos "
          "de confianza real aca que en el moneyline de futbol, eso no es necesariamente una falla del "
          "modelo, es la naturaleza de este mercado especifico.")


if __name__ == "__main__":
    run()