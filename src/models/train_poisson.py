"""
Orquestador de Fase 2 (v1): entrena el modelo de Poisson, genera predicciones
para cada partido, las compara con Pinnacle, y calcula el blending Benter Boost.

RECORDATORIO: esta pasada es EN-MUESTRA (in-sample) -- diagnostico de que el
pipeline funciona, NO una medicion real de si el modelo tiene edge. Eso llega
en Fase 3.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.poisson_model import build_long_format, fit_poisson_model, predict_dataframe
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities


def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv")

    print("Entrenando modelo de Poisson sobre 1,900 partidos...")
    long_df = build_long_format(df)
    model = fit_poisson_model(long_df)

    print("Generando predicciones para cada partido (puede tardar ~30s)...")
    model_preds = predict_dataframe(model, df)
    df = pd.concat([df.reset_index(drop=True), model_preds.reset_index(drop=True)], axis=1)

    market_probs = df[["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]].rename(
        columns={"pinnacle_close_prob_home": "prob_home", "pinnacle_close_prob_draw": "prob_draw", "pinnacle_close_prob_away": "prob_away"}
    )
    model_probs = df[["model_prob_home", "model_prob_draw", "model_prob_away"]].rename(
        columns={"model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away"}
    )

    model_brier = brier_score_multiclass(model_probs, df["FTR"])
    market_brier = brier_score_multiclass(market_probs, df["FTR"])
    market_weight = compute_blend_weight(model_brier, market_brier)

    blended = blend_probabilities(model_probs, market_probs, market_weight)
    df["blend_prob_home"] = blended["prob_home"]
    df["blend_prob_draw"] = blended["prob_draw"]
    df["blend_prob_away"] = blended["prob_away"]
    blend_brier = brier_score_multiclass(blended, df["FTR"])

    print("\n=== Resultados EN-MUESTRA (diagnostico, NO backtest real) ===")
    print(f"Brier score modelo propio:        {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.4f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.4f}")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_insample.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    run()