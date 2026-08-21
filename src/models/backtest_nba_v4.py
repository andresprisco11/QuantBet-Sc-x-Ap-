"""
Fase 10 (NBA) -- backtest walk-forward de NBA v4 (Elo + calendario + GSSD
trailing), mismo patron exacto que `backtest_nba_v3.py` pero importando
`nba_margin_model_v4`. Guarda en un path DISTINTO (no pisa las predicciones
de v1/v2/v3) y al final imprime un bloque `[COMPARAR CONTRA v3]` con los
numeros v3 ya confirmados (2026-08-21) citados inline.

Uso: python -m src.models.backtest_nba_v4
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nba_margin_model_v4 import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH_WITH_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean_with_odds.csv"
DATA_PATH_NO_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"

# Numeros v3 YA CONFIRMADOS (2026-08-21, ver roadmap) -- citados aca solo
# para comparar en pantalla, no se recalculan.
V3_BASELINE = {
    "model_brier_all_seasons": 0.208923,
    "n_reliable_market": 6904,
    "model_brier_reliable_subset": 0.216718,
    "market_brier": 0.203437,
    "blend_brier": 0.207101,
    "gap_vs_mercado": 0.003664,
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
    if "home_off_l10" not in df.columns:
        print("[SKIP] Falta home_off_l10/etc -- corre 'python -m src.processing.add_nba_gssd_features' primero.")
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
    print(f"\n=== Resultados FUERA DE MUESTRA -- NBA v4 (Elo + calendario + GSSD trailing) ===")
    print(f"Partidos evaluados OOS: {n_total}")
    print(f"Brier score modelo propio (TODAS las temporadas OOS): {model_brier_all:.6f}")
    print(f"[COMPARAR CONTRA v3] model_brier_all_seasons v3={V3_BASELINE['model_brier_all_seasons']:.6f} "
          f"-> v4={model_brier_all:.6f} ({'MEJORA' if model_brier_all < V3_BASELINE['model_brier_all_seasons'] else 'NO mejora'})")

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
        print(f"Brier score modelo propio:          v3={V3_BASELINE['model_brier_reliable_subset']:.6f} -> v4={model_brier:.6f}")
        print(f"Brier score mercado (no deberia cambiar, mismo mercado): v3={V3_BASELINE['market_brier']:.6f} -> v4={market_brier:.6f}")
        print(f"Peso asignado al mercado:           {market_weight:.1%}")
        print(f"Brier score blend (Benter Boost):   v3={V3_BASELINE['blend_brier']:.6f} -> v4={blend_brier:.6f}")
        print(f"Gap blend vs. mercado:               v3={V3_BASELINE['gap_vs_mercado']:+.6f} -> v4={gap:+.6f} "
              f"({'el blend gana' if gap < 0 else 'el mercado sigue ganando'})")
        print(f"\n[COMPARAR CONTRA v3] gap_vs_mercado v3={V3_BASELINE['gap_vs_mercado']:+.6f} -> v4={gap:+.6f} "
              f"({'MEJORA' if gap < V3_BASELINE['gap_vs_mercado'] else 'NO mejora'})")

        metrics.update({
            "n_reliable_market": n_reliable,
            "model_brier_reliable_subset": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": gap,
        })

    out_path = PROCESSED_DATA_DIR / "NBA" / "model_predictions_oos_walkforward_v4.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nba_v4.py",
        model_name="nba_margin_normal",
        model_version="v4_elo_calendar_gssd",
        data_paths=[data_path],
        features="point_margin ~ elo_diff + b2b_diff + 3in4_diff + home_is_denver + home_off_l10 + "
                 "home_def_l10 + away_off_l10 + away_def_l10 (GSSD trailing, ver add_nba_gssd_features.py, "
                 "adaptado del modelo GSSD de Andrew Mack para ser walk-forward-safe).",
        hyperparameters={
            "elo_k_factor": 20.0, "elo_home_advantage": 100.0, "elo_season_regression": 0.25,
            "gssd_window": 10, "gssd_min_periods": 3,
        },
        metrics=metrics,
        predictions_path=out_path,
        notes="v4 de NBA -- agrega 4 regresores GSSD trailing (ataque/defensa de local/visitante, "
              "ultimos 10 partidos en ese contexto) a elo_diff+b2b_diff+3in4_diff+home_is_denver. "
              "Las 4 variables screeneadas con OLS full-sample antes del walk-forward: todas "
              "significativas (p<0.01) y con el signo teoricamente correcto. Comparado inline contra "
              "v3 (ver [COMPARAR CONTRA v3] en el output).",
    )


if __name__ == "__main__":
    run()
