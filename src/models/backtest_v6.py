"""
Fase 3 v6: mismo walk-forward honesto que backtest_v4.py, pero usando el
modelo v6 (poisson_model_v6.py): forma reciente por tiros al arco (heredada
de v4, unica variable que dio resultado positivo hasta ahora) MAS
diferencial neto de corners recientes.

Objetivo de esta corrida: v5 (separar tiros en ataque/defensa) fue
negativo -- la hipotesis alternativa es que la ganancia no viene de
"partir" la misma señal en dos, sino de agregar una señal DISTINTA
encima de v4. Comparacion directa contra v2 y v4: mismo conjunto exacto
de 1,900 partidos.

REQUISITO PREVIO: correr 'python -m src.processing.add_team_form_features'
(version actualizada, con columnas de corners) sobre matches_clean.csv
antes de este script.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v6 import build_long_format_v6, fit_poisson_model_v6, predict_dataframe_v6
from src.models.poisson_model_v2 import DEFAULT_HALF_LIFE_DAYS
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]


def run():
    df = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv")
    df["season"] = df["season"].astype(str)
    ordered_seasons = SEASONS
    all_oos_records = []

    for i in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:i]
        test_season = ordered_seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if test_df.empty:
            print(f"[SKIP] Temporada {test_season}: sin partidos.")
            continue

        known_teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
        print(f"Entrenando con {train_seasons} ({len(train_df)} partidos) -> evaluando {test_season} "
              f"({len(test_df)} partidos, ninguno excluido)...")

        long_df = build_long_format_v6(train_df)
        model = fit_poisson_model_v6(long_df)
        preds = predict_dataframe_v6(model, test_df, known_teams)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
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

    print("\n=== Resultados FUERA DE MUESTRA v6 (recencia + PROMOTED_TEAM + tiros al arco + corners) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio v6:      {model_brier:.6f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.6f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.6f}")
    print("\n(Referencia v4 -- MEJOR MODELO HASTA AHORA, mismo conjunto de partidos: modelo 0.595578, "
          "mercado 0.560801, blend 0.568619, gap +0.007818)")
    print("(Referencia v5 -- descartado, mismo conjunto de partidos: modelo 0.597549, mercado 0.560801, "
          "blend 0.569360, gap +0.008559)")
    print("(Referencia v2 -- mismo conjunto de partidos: modelo 0.598631, mercado 0.560801, "
          "blend 0.569562, gap +0.008761)")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v6.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_v6.py",
        model_name="poisson",
        model_version="v6",
        data_paths=[PROCESSED_DATA_DIR / "EPL" / "matches_clean.csv"],
        features="goals ~ is_home + C(team) + C(opponent) + team_recent_form + team_recent_corner_diff "
                  "[+ filas sinteticas PROMOTED_TEAM], freq_weights=decaimiento exponencial por recencia",
        hyperparameters={
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "recent_form_rolling_window": 5,
            "recent_form_stats": "shots_on_target_diff (v4) + corner_diff (v6, nuevo)",
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
        notes="Walk-forward v6: agrega diferencial neto de corners recientes encima de v4 (no de v5, "
              "descartado). No incluye Dixon-Coles (v3, resultado negativo documentado).",
    )


if __name__ == "__main__":
    run()