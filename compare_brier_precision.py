"""
Diagnostico puntual (no es parte del pipeline permanente, no necesita tracking):
recalcula el Brier score de v2 y v3 con mas decimales que los 4 que se
imprimieron durante los backtests, usando los CSV que ya quedaron guardados
en data/processed/EPL/. No vuelve a entrenar nada -- es instantaneo.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.blending import brier_score_multiclass

MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]
BLEND_COLS = ["blend_prob_home", "blend_prob_draw", "blend_prob_away"]


def report(path, label):
    df = pd.read_csv(path)
    model_probs = df[MODEL_COLS].rename(columns={
        "model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away",
    })
    model_brier = brier_score_multiclass(model_probs, df["FTR"])

    has_blend = df[BLEND_COLS].notna().all(axis=1)
    blend_subset = df.loc[has_blend]
    blend_probs = blend_subset[BLEND_COLS].rename(columns={
        "blend_prob_home": "prob_home", "blend_prob_draw": "prob_draw", "blend_prob_away": "prob_away",
    })
    blend_brier = brier_score_multiclass(blend_probs, blend_subset["FTR"])

    print(f"{label}: model_brier={model_brier:.8f} | blend_brier={blend_brier:.8f} | n_blend={len(blend_subset)}")
    return model_brier, blend_brier


v2_model, v2_blend = report(PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v2.csv", "v2")
v3_model, v3_blend = report(PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v3.csv", "v3")

print(f"\nDiferencia model_brier (v2 - v3): {v2_model - v3_model:+.8f}  (positivo = v3 mejor)")
print(f"Diferencia blend_brier (v2 - v3): {v2_blend - v3_blend:+.8f}  (positivo = v3 mejor)")