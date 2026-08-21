"""
Fase 10 (NBA) -- backtest walk-forward de NBA v5 (Elo + calendario + GSSD
EWM span=15), mismo patron exacto que `backtest_nba_v4.py` pero importando
`nba_margin_model_v5`. Guarda en un path DISTINTO (no pisa las
predicciones de v1-v4) y al final imprime un bloque `[COMPARAR CONTRA v4]`
con los numeros v4 ya confirmados (2026-08-21, corrida real del usuario,
run_id 20260821_034554_d43caa4a) citados inline.

Uso: python -m src.models.backtest_nba_v5
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nba_margin_model_v5 import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH_WITH_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean_with_odds.csv"
DATA_PATH_NO_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"

# Numeros v4 YA CONFIRMADOS (2026-08-21, corrida real del usuario,
# run_id=20260821_034554_d43caa4a, ver roadmap) -- citados aca solo para
# comparar en pantalla, no se recalculan.
V4_BASELINE = {
    "model_brier_all_seasons": 0.208565,
    "n_reliable_market": 6904,
    "model_brier_reliable_subset": 0.215689,
    "market_brier": 0.203437,
    "blend_brier": 0.206765,
    "gap_vs_mercado": 0.003329,
}


def _brier_binary(model_prob_home: pd.Series, ftr: pd.Series) -> float:
    actual = ftr.map({"H": 1.0, "A": 0.0})
    return float(((model_prob_home - actual) ** 2).mean())


def _blend_weight(model_brier: float, market_brier: float) -> float:
    return model_brier / (model_brier + market_brier)


def run() -> None:
    has_odds = DATA_PATH_WITH_ODDS.exists()
    data_path = DATA_PATH_WITH_ODDS if has_odds else DATA_PATH_NO_ODDS
    if not data_path.exists():
        print(f"[SKIP] No existe {data_path}.")
        return

    df = pd.read_csv(data_path)
    if "home_off_ewm" not in df.columns:
        print("[SKIP] Falta home_off_ewm/etc -- corre 'python -m src.processing.add_nba_gssd_features' "
              "(version actualizada con EWM) primero.")
        return

    ordered_seasons = sorted(df["season"].unique())
    all_oos_records = []

    print(f"Temporadas disponibles: {ordered_seasons[0]}-{ordered_seasons[-1]} ({len(ordered_seasons)} temporadas)")
    for i in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:i]
        test_season = ordered_seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if test_df.empty:
            continue

        model, sigma = fit_margin_model(train_df)
        preds = predict_dataframe(model, sigma, test_df)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    oos_df = pd.concat(all_oos_records, ignore_index=True)

    n_no_pred = int(oos_df["model_prob_home"].isna().sum())
    if n_no_pred:
        print(f"\n[INFO] {n_no_pred} partidos OOS sin prediccion (features no calculables) -- "
              f"excluidos de las metricas.")
    oos_df = oos_df[oos_df["model_prob_home"].notna()]
    n_total = len(oos_df)

    model_brier_all = _brier_binary(oos_df["model_prob_home"], oos_df["FTR"])
    print(f"\n=== Resultados FUERA DE MUESTRA -- NBA v5 (Elo + calendario + GSSD EWM span=15) ===")
    print(f"Partidos evaluados OOS: {n_total}")
    print(f"Brier score modelo propio (TODAS las temporadas OOS): {model_brier_all:.6f}")
    print(f"[COMPARAR CONTRA v4] model_brier_all_seasons v4={V4_BASELINE['model_brier_all_seasons']:.6f} "
          f"-> v5={model_brier_all:.6f} ({'MEJORA' if model_brier_all < V4_BASELINE['model_brier_all_seasons'] else 'NO mejora'})")

    metrics = {"n_total": n_total, "model_brier_all_seasons": model_brier_all}

    if has_odds and "market_prob_home" in oos_df.columns:
        reliable = oos_df["market_prob_home"].notna()
        subset = oos_df.loc[reliable]
        n_reliable = int(reliable.sum())

        model_brier = _brier_binary(subset["model_prob_home"], subset["FTR"])
        market_brier = _brier_binary(subset["market_prob_home"], subset["FTR"])
        market_weight = _blend_weight(model_brier, market_brier)
        blended = market_weight * subset["market_prob_home"] + (1.0 - market_weight) * subset["model_prob_home"]
        blend_brier = _brier_binary(blended, subset["FTR"])
        gap = blend_brier - market_brier

        print(f"\n--- Subconjunto con mercado disponible: {n_reliable} partidos ---")
        print(f"Brier score modelo propio:          v4={V4_BASELINE['model_brier_reliable_subset']:.6f} -> v5={model_brier:.6f}")
        print(f"Brier score mercado (no deberia cambiar, mismo mercado): v4={V4_BASELINE['market_brier']:.6f} -> v5={market_brier:.6f}")
        print(f"Peso asignado al mercado:           {market_weight:.1%}")
        print(f"Brier score blend (Benter Boost):   v4={V4_BASELINE['blend_brier']:.6f} -> v5={blend_brier:.6f}")
        print(f"Gap blend vs. mercado:               v4={V4_BASELINE['gap_vs_mercado']:+.6f} -> v5={gap:+.6f} "
              f"({'el blend gana' if gap < 0 else 'el mercado sigue ganando'})")
        print(f"\n[COMPARAR CONTRA v4] gap_vs_mercado v4={V4_BASELINE['gap_vs_mercado']:+.6f} -> v5={gap:+.6f} "
              f"({'MEJORA' if gap < V4_BASELINE['gap_vs_mercado'] else 'NO mejora'})")

        metrics.update({
            "n_reliable_market": n_reliable,
            "model_brier_reliable_subset": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": gap,
        })

    out_path = PROCESSED_DATA_DIR / "NBA" / "model_predictions_oos_walkforward_v5.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nba_v5.py",
        model_name="nba_margin_normal",
        model_version="v5_elo_calendar_gssd_ewm",
        data_paths=[data_path],
        features="point_margin ~ elo_diff + b2b_diff + 3in4_diff + home_is_denver + home_off_ewm + "
                 "home_def_ewm + away_off_ewm + away_def_ewm (GSSD con EWM span=15 en vez de promedio "
                 "movil simple N=10 de v4 -- ver add_nba_gssd_features.py para el barrido de "
                 "ventanas/spans que confirmo que EWM span=15 es superior).",
        hyperparameters={
            "elo_k_factor": 20.0, "elo_home_advantage": 100.0, "elo_season_regression": 0.25,
            "gssd_ewm_span": 15, "gssd_min_periods": 3,
        },
        metrics=metrics,
        predictions_path=out_path,
        notes="v5 de NBA -- reemplaza los 4 regresores GSSD de v4 (promedio movil simple N=10) por su "
              "version de promedio exponencial (EWM span=15), confirmado superior en un barrido "
              "walk-forward completo de 8 configuraciones (N=5/10/15/20 simple, span=5/10/15/20/25/30/40/60 "
              "EWM). Comparado inline contra v4 (ver [COMPARAR CONTRA v4] en el output).",
    )


if __name__ == "__main__":
    run()
