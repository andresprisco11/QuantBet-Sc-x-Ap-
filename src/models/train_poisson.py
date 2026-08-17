"""
Orquestador de Fase 2 (v1): entrena el modelo de Poisson, genera predicciones
para cada partido, las compara con Pinnacle, y calcula el blending Benter Boost.

RECORDATORIO: esta pasada es EN-MUESTRA (in-sample) -- diagnostico de que el
pipeline funciona, NO una medicion real de si el modelo tiene edge. Eso llega
en Fase 3 (backtest.py / backtest_v2.py, walk-forward).

NOTA: el Brier score de mercado/blend excluye partidos sin cuota de cierre de
Pinnacle disponible (mismo tratamiento que en backtest.py -- ver esa nota ahi
para el detalle de por que football-data.co.uk no siempre tiene el archivo de
cierre completo en temporadas muy recientes). El Brier del modelo propio SI
usa todas las filas, porque no depende de Pinnacle en absoluto.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.poisson_model import build_long_format, fit_poisson_model, predict_dataframe
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]


def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv")

    print(f"Entrenando modelo de Poisson sobre {len(df)} partidos...")
    long_df = build_long_format(df)
    model = fit_poisson_model(long_df)

    print("Generando predicciones para cada partido (puede tardar ~30s)...")
    model_preds = predict_dataframe(model, df)
    df = pd.concat([df.reset_index(drop=True), model_preds.reset_index(drop=True)], axis=1)

    # --- Brier score del modelo propio: usa TODAS las filas ---
    # (no depende de Pinnacle, asi que no hay razon para excluir nada aqui)
    model_probs_full = df[MODEL_COLS].rename(columns={
        "model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away",
    })
    model_brier = brier_score_multiclass(model_probs_full, df["FTR"])

    # --- Brier score de mercado / blend: SOLO filas con cuota de cierre de ---
    # Pinnacle completa (evita el mismo bug de NaN que encontramos en Fase 3).
    has_market = df[MARKET_COLS].notna().all(axis=1)
    n_total = len(df)
    n_market = int(has_market.sum())
    n_excluded = n_total - n_market
    if n_excluded:
        print(f"[AVISO] {n_excluded} de {n_total} partidos excluidos del calculo de mercado/blend: "
              f"sin cuota de cierre de Pinnacle disponible en football-data.co.uk.")

    market_subset = df.loc[has_market]
    market_probs = market_subset[MARKET_COLS].rename(columns={
        "pinnacle_close_prob_home": "prob_home", "pinnacle_close_prob_draw": "prob_draw", "pinnacle_close_prob_away": "prob_away",
    })
    model_probs_subset = model_probs_full.loc[has_market]

    market_brier = brier_score_multiclass(market_probs, market_subset["FTR"])
    market_weight = compute_blend_weight(model_brier, market_brier)
    blended = blend_probabilities(model_probs_subset, market_probs, market_weight)
    blend_brier = brier_score_multiclass(blended, market_subset["FTR"])

    df.loc[has_market, "blend_prob_home"] = blended["prob_home"].values
    df.loc[has_market, "blend_prob_draw"] = blended["prob_draw"].values
    df.loc[has_market, "blend_prob_away"] = blended["prob_away"].values

    print("\n=== Resultados EN-MUESTRA (diagnostico, NO backtest real) ===")
    print(f"Partidos totales: {n_total}")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio:        {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.4f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.4f}")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_insample.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="train_poisson.py",
        model_name="poisson",
        model_version="v1",
        data_paths=[PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv"],
        features="goals ~ is_home + C(team) + C(opponent)",
        hyperparameters={},
        metrics={
            "n_total": n_total,
            "n_market": n_market,
            "model_brier": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": blend_brier - market_brier,
        },
        predictions_path=out_path,
        notes="Diagnostico EN-MUESTRA (no walk-forward) -- confirma que el pipeline de entrenamiento/blending "
              "funciona de punta a punta. No es evidencia de edge real (ver backtest.py para eso).",
    )


if __name__ == "__main__":
    run()