"""
Fase 10 -- diagnostico de "winner's curse" de seleccion para NFL, mismo
patron que `selection_bias_check.py` (futbol): el resultado economico
diagnostico (`economic_backtest_nfl.py`) salio muy negativo (ROI -50.98%,
drawdown 73.89%, 3,127 de 5,295 partidos seleccionados con solo min_edge=2%
-- un volumen enorme, 59% del universo) y la firma clasica que ya se vio en
futbol era: el modelo esta mal calibrado especificamente en las apuestas
QUE SELECCIONA (no en general), sobreestimando su propia probabilidad.

Metodologia: para CADA partido con odds y sin push, se toma el lado con
mayor `blend_prob_*_covers` (el lado que el modelo prefiere, sin filtrar
por edge todavia) -- esto da una calibracion de "poblacion general"
(TODOS los partidos, una prediccion direccional por partido). Despues se
repite el mismo calculo SOLO sobre los partidos que `economic_backtest_nfl.
_select_bets` efectivamente elige (min_edge=2%) -- esto da la calibracion
de "seleccionados". Si la brecha (prob predicha - acierto real) es mucho
mas negativa en seleccionados que en poblacion general, es la firma de
winner's curse ya vista en futbol: la seleccion de apuestas concentra
sistematicamente los casos donde el modelo esta MAS sobreconfiado, no los
casos donde tiene mas razon.

Reusa `_select_bets` de economic_backtest_nfl.py -- una sola fuente de
verdad de la logica de seleccion, no se duplica.

Uso: python -m src.evaluation.nfl_selection_bias_check
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.evaluation.economic_backtest_nfl import _select_bets, MIN_EDGE

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "spread_evaluation_v1.csv"

BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]


def _directional_calls(df: pd.DataFrame) -> pd.DataFrame:
    """Una prediccion por partido: el lado (home/away) que el blend prefiere,
    y si ese lado gano -- SIN filtrar por edge (esto es 'poblacion
    general', no la seleccion de apuestas real)."""
    df = df.copy()
    home_favored = df["blend_prob_home_covers"] >= df["blend_prob_away_covers"]
    df["predicted_prob"] = np.where(home_favored, df["blend_prob_home_covers"], df["blend_prob_away_covers"])
    df["actual_won"] = np.where(home_favored, df["home_covers"] == 1.0, df["home_covers"] == 0.0)
    return df


def _bucket_summary(df: pd.DataFrame, prob_col: str, won_col: str, label: str) -> pd.DataFrame:
    d = df.copy()
    d["bucket"] = pd.cut(d[prob_col], bins=BUCKETS, right=False)
    summary = d.groupby("bucket", observed=True).agg(
        n=(won_col, "count"),
        prob_predicha_avg=(prob_col, "mean"),
        acierto_real=(won_col, "mean"),
    )
    summary["gap"] = summary["acierto_real"] - summary["prob_predicha_avg"]
    summary.columns = pd.MultiIndex.from_product([[label], summary.columns])
    return summary


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre backtest_nfl_spread.py primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["gameday"] = pd.to_datetime(df["gameday"])

    non_push = df.loc[~df["is_push"]].copy()
    population = _directional_calls(non_push)

    bets = _select_bets(df, MIN_EDGE)
    bets_decided = bets.loc[~bets["is_push"]].copy()
    bets_decided["predicted_prob"] = bets_decided["prob"]
    bets_decided["actual_won"] = bets_decided["won"]

    pop_summary = _bucket_summary(population, "predicted_prob", "actual_won", "poblacion_general")
    sel_summary = _bucket_summary(bets_decided, "predicted_prob", "actual_won", "seleccionados")

    combined = pop_summary.join(sel_summary, how="outer")

    print(f"Poblacion general (una prediccion direccional por partido, sin filtrar por edge): "
          f"{len(population)} partidos.")
    print(f"Seleccionados por economic_backtest_nfl (min_edge={MIN_EDGE:.0%}, sin push): "
          f"{len(bets_decided)} partidos.\n")
    print(combined.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "NaN"))

    pop_gap_avg = (population["actual_won"].astype(float).mean() - population["predicted_prob"].mean())
    sel_gap_avg = (bets_decided["actual_won"].astype(float).mean() - bets_decided["predicted_prob"].mean())
    print(f"\nGap promedio (acierto real - prob predicha) -- poblacion general: {pop_gap_avg:+.4f}")
    print(f"Gap promedio (acierto real - prob predicha) -- seleccionados:      {sel_gap_avg:+.4f}")
    print(f"\nLectura: si el gap de 'seleccionados' es mucho mas negativo que el de 'poblacion general', "
          f"es la misma firma de winner's curse ya diagnosticada en futbol -- la seleccion de apuestas "
          f"concentra los casos donde el modelo esta MAS sobreconfiado, no los casos donde tiene mas razon.")


if __name__ == "__main__":
    run()