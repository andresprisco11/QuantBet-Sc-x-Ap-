"""
Fase 3 v2: mismo walk-forward honesto que backtest.py (v1), pero usando el
modelo v2 (poisson_model_v2.py): ponderacion por recencia + rating
PROMOTED_TEAM para equipos recien ascendidos, en vez de excluirlos.

Objetivo de esta corrida: confirmar si estas dos mejoras cierran (o al menos
reducen) la brecha que encontramos en Fase 3 v1 -- el blend perdiendo contra
el mercado solo (v1: modelo 0.6077, mercado 0.5637, blend 0.5754).

Diferencia clave frente a backtest.py (v1): aqui NINGUN partido se excluye
por equipo recien ascendido -- todos se evaluan. La unica exclusion que se
mantiene es la de partidos sin cuota de cierre de Pinnacle (limitacion de
disponibilidad de datos en football-data.co.uk, no del modelo).

IMPORTANTE al comparar contra v1: el numero total de partidos evaluados va a
ser MAYOR en v2 (porque ya no excluye ascendidos), y esos partidos
adicionales son justamente los mas dificiles de predecir (equipos sin
historial propio). El criterio de exito real no es "el Brier de v2 es menor
que el de v1" (no son exactamente el mismo conjunto de partidos) -- es "el
blend de v2 le gana al mercado DENTRO de su propia corrida".
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v2 import build_long_format_v2, fit_poisson_model_v2, predict_dataframe_v2
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities

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
              f"({len(test_df)} partidos, ninguno excluido -- PROMOTED_TEAM cubre a los ascendidos)...")

        long_df = build_long_format_v2(train_df)
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

    print("\n=== Resultados FUERA DE MUESTRA v2 (recencia + PROMOTED_TEAM) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio v2:      {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.4f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.4f}")
    print("\n(Referencia v1 -- otro conjunto de partidos, no 100% comparable: "
          "modelo 0.6077, mercado 0.5637, blend 0.5754)")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v2.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    run()