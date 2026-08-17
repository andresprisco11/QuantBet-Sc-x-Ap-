"""
Barrido de half_life_days para el modelo v2 (poisson_model_v2.py).

El half-life de 456 dias que usamos en la primera corrida de v2 fue un
valor inicial razonable (~1.5 temporadas), pero nunca se comparo contra
alternativas. Este script corre el mismo walk-forward de backtest_v2.py
para varios valores de half-life y los compara en una sola tabla, para
decidir con datos -- no a ojo -- cual usar de ahora en adelante.

Un half-life mas CORTO le da mas peso a la forma reciente (mas reactivo,
pero con mas ruido). Uno mas LARGO se parece mas al modelo v1 sin
ponderar (mas estable, pero mas lento para reaccionar a cambios reales
de plantel).
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v2 import build_long_format_v2, fit_poisson_model_v2, predict_dataframe_v2
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]

# Candidatos a probar, en dias. Ajusta esta lista si quieres agregar mas.
HALF_LIFE_CANDIDATES = [200, 456, 700]


def run_walkforward(df: pd.DataFrame, half_life_days: float) -> dict:
    """Corre el walk-forward completo para un half_life_days especifico y devuelve las metricas finales."""
    ordered_seasons = SEASONS
    all_oos_records = []

    for i in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:i]
        test_season = ordered_seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if test_df.empty:
            continue

        known_teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
        long_df = build_long_format_v2(train_df, half_life_days=half_life_days)
        model = fit_poisson_model_v2(long_df)
        preds = predict_dataframe_v2(model, test_df, known_teams)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    oos_df = pd.concat(all_oos_records, ignore_index=True)

    model_probs_full = oos_df[MODEL_COLS].rename(columns={
        "model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away",
    })
    model_brier = brier_score_multiclass(model_probs_full, oos_df["FTR"])

    has_market = oos_df[MARKET_COLS].notna().all(axis=1)
    market_subset = oos_df.loc[has_market]
    market_probs = market_subset[MARKET_COLS].rename(columns={
        "pinnacle_close_prob_home": "prob_home", "pinnacle_close_prob_draw": "prob_draw", "pinnacle_close_prob_away": "prob_away",
    })
    model_probs_subset = model_probs_full.loc[has_market]

    market_brier = brier_score_multiclass(market_probs, market_subset["FTR"])
    market_weight = compute_blend_weight(model_brier, market_brier)
    blended = blend_probabilities(model_probs_subset, market_probs, market_weight)
    blend_brier = brier_score_multiclass(blended, market_subset["FTR"])

    return {
        "half_life_days": half_life_days,
        "n_total": len(oos_df),
        "n_market": int(has_market.sum()),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "market_weight": market_weight,
        "blend_brier": blend_brier,
        "gap_vs_mercado": blend_brier - market_brier,  # negativo = el blend le gana al mercado
    }


def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv")
    df["season"] = df["season"].astype(str)
    data_paths = [PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv"]

    results = []
    for hl in HALF_LIFE_CANDIDATES:
        print(f"Probando half_life_days={hl}...")
        result = run_walkforward(df, hl)
        results.append(result)
        print(f"  -> modelo {result['model_brier']:.4f} | mercado {result['market_brier']:.4f} | "
              f"blend {result['blend_brier']:.4f} | gap vs mercado {result['gap_vs_mercado']:+.4f}")

        # Cada candidato de half-life es su propia corrida -- se registra
        # individualmente, no solo el resumen final, para poder reconstruir
        # exactamente que se probo y con que resultado.
        log_run(
            script="tune_half_life.py",
            model_name="poisson",
            model_version="v2",
            data_paths=data_paths,
            features="goals ~ is_home + C(team) + C(opponent) [+ filas sinteticas PROMOTED_TEAM], freq_weights=decaimiento exponencial por recencia",
            hyperparameters={"half_life_days": hl},
            metrics=result,
            predictions_path=None,
            notes="Parte del barrido de half-life (200/456/700). Ver half_life_tuning_results.csv para la comparativa completa.",
        )

    results_df = pd.DataFrame(results)
    print("\n=== Comparacion de half-life ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = results_df.loc[results_df["blend_brier"].idxmin()]
    print(f"\nMejor blend_brier: half_life_days={int(best['half_life_days'])} (blend={best['blend_brier']:.4f}, "
          f"gap vs mercado={best['gap_vs_mercado']:+.4f})")

    out_path = PROCESSED_DATA_DIR / "EPL" / "half_life_tuning_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    run()