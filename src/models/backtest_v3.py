"""
Fase 3 v3: mismo walk-forward honesto que backtest_v2.py, pero usando el
modelo v3 (poisson_model_v3.py): recencia + PROMOTED_TEAM (heredado de v2)
MAS el ajuste de correlacion Dixon-Coles para marcadores bajos (0-0, 0-1,
1-0, 1-1).

Objetivo de esta corrida: confirmar si cerrar la ultima limitacion de
arquitectura documentada desde v1 (independencia de goles local/visitante)
termina de cerrar -- o al menos sigue reduciendo -- la brecha que quedo
despues de v2 (blend 0.5696 vs. mercado 0.5608, gap +0.0088) y que el
tuneo de half-life confirmo que no se iba a cerrar solo con mas ajuste de
recencia.

DIFERENCIA CLAVE frente a backtest_v2.py: en cada fold del walk-forward se
estima un rho de Dixon-Coles nuevo, usando SOLO los datos de entrenamiento
de ese fold (nunca los de test) -- mismo principio de no fuga de informacion
del futuro que el resto del proyecto. El rho de cada fold queda registrado
individualmente en el log de tracking, no solo un promedio final.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v2 import build_long_format_v2, fit_poisson_model_v2, DEFAULT_HALF_LIFE_DAYS
from src.models.poisson_model_v3 import estimate_rho, predict_dataframe_v3
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]


def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv")
    df["season"] = df["season"].astype(str)
    ordered_seasons = SEASONS
    all_oos_records = []
    rho_per_fold = {}

    for i in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:i]
        test_season = ordered_seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if test_df.empty:
            print(f"[SKIP] Temporada {test_season}: sin partidos.")
            continue

        known_teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))

        long_df = build_long_format_v2(train_df)
        model = fit_poisson_model_v2(long_df)
        rho = estimate_rho(model, train_df, known_teams)
        rho_per_fold[test_season] = rho

        print(f"Entrenando con {train_seasons} ({len(train_df)} partidos) -> evaluando {test_season} "
              f"({len(test_df)} partidos, rho Dixon-Coles estimado = {rho:+.4f})...")

        preds = predict_dataframe_v3(model, test_df, known_teams, rho)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        fold_df["fold_rho"] = rho
        all_oos_records.append(fold_df)

    oos_df = pd.concat(all_oos_records, ignore_index=True)

    model_probs_full = oos_df[MODEL_COLS].rename(columns={
        "model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away",
    })
    model_brier = brier_score_multiclass(model_probs_full, oos_df["FTR"])

    has_market = oos_df[MARKET_COLS].notna().all(axis=1)
    n_total = len(oos_df)
    n_market = int(has_market.sum())
    n_excluded = n_total - n_market
    if n_excluded:
        print(f"[AVISO] {n_excluded} de {n_total} partidos OOS excluidos del calculo de mercado/blend: "
              f"sin cuota de cierre de Pinnacle disponible en football-data.co.uk.")

    market_subset = oos_df.loc[has_market]
    market_probs = market_subset[MARKET_COLS].rename(columns={
        "pinnacle_close_prob_home": "prob_home", "pinnacle_close_prob_draw": "prob_draw", "pinnacle_close_prob_away": "prob_away",
    })
    model_probs_subset = model_probs_full.loc[has_market]

    market_brier = brier_score_multiclass(market_probs, market_subset["FTR"])
    market_weight = compute_blend_weight(model_brier, market_brier)
    blended = blend_probabilities(model_probs_subset, market_probs, market_weight)
    blend_brier = brier_score_multiclass(blended, market_subset["FTR"])

    oos_df.loc[has_market, "blend_prob_home"] = blended["prob_home"].values
    oos_df.loc[has_market, "blend_prob_draw"] = blended["prob_draw"].values
    oos_df.loc[has_market, "blend_prob_away"] = blended["prob_away"].values

    print("\n=== Resultados FUERA DE MUESTRA v3 (recencia + PROMOTED_TEAM + Dixon-Coles) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"rho de Dixon-Coles por fold: {rho_per_fold}")
    print(f"Brier score modelo propio v3:      {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.4f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.4f}")
    print("\n(Referencia v2 -- mismo conjunto de partidos, comparacion directa valida: "
          "modelo 0.5986, mercado 0.5608, blend 0.5696, gap +0.0088)")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v3.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_v3.py",
        model_name="poisson",
        model_version="v3",
        data_paths=[PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv"],
        features="goals ~ is_home + C(team) + C(opponent) [+ filas sinteticas PROMOTED_TEAM], "
                  "freq_weights=decaimiento exponencial por recencia, "
                  "+ ajuste de correlacion Dixon-Coles (tau) sobre marcadores 0-0/0-1/1-0/1-1",
        hyperparameters={
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "rho_per_fold": rho_per_fold,
            "rho_mean": sum(rho_per_fold.values()) / len(rho_per_fold) if rho_per_fold else None,
        },
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
        notes="Walk-forward v3: recencia + PROMOTED_TEAM (heredado de v2) + correlacion Dixon-Coles "
              "para marcadores bajos. rho estimado por MLE en cada fold, solo sobre datos de entrenamiento.",
    )


if __name__ == "__main__":
    run()