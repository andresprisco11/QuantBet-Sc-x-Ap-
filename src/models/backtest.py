"""
Fase 3: Backtesting riguroso con walk-forward validation (ventana expansiva
por temporada).

A diferencia del diagnostico en-muestra de Fase 2 (train_poisson.py), aqui el
modelo NUNCA ve los resultados de la temporada que esta evaluando. Se entrena
con todas las temporadas anteriores y se predice unicamente la temporada
siguiente -- exactamente como se usaria el modelo en produccion real.

LIMITACION CONOCIDA: cada temporada llegan ~3 equipos recien ascendidos sin
historial en las temporadas de entrenamiento. El modelo no puede asignarles
un rating de ataque/defensa sin haber visto ningun partido suyo, asi que esos
partidos se EXCLUYEN de la evaluacion de esa temporada (se reporta cuantos).
Mejora futura: asignarles un rating "recien ascendido" por defecto en vez de
excluirlos.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model import build_long_format, fit_poisson_model, predict_dataframe
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities


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
        before = len(test_df)
        test_df = test_df[test_df["HomeTeam"].isin(known_teams) & test_df["AwayTeam"].isin(known_teams)]
        skipped = before - len(test_df)

        print(f"Entrenando con {train_seasons} ({len(train_df)} partidos) -> evaluando {test_season} ({len(test_df)} partidos)...")
        if skipped:
            print(f"  [AVISO] {skipped} partidos de {test_season} excluidos: equipos recien ascendidos sin historial previo.")

        long_df = build_long_format(train_df)
        model = fit_poisson_model(long_df)

        preds = predict_dataframe(model, test_df)
        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    oos_df = pd.concat(all_oos_records, ignore_index=True)

    market_probs = oos_df[["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]].rename(
        columns={"pinnacle_close_prob_home": "prob_home", "pinnacle_close_prob_draw": "prob_draw", "pinnacle_close_prob_away": "prob_away"}
    )
    model_probs = oos_df[["model_prob_home", "model_prob_draw", "model_prob_away"]].rename(
        columns={"model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away"}
    )

    model_brier = brier_score_multiclass(model_probs, oos_df["FTR"])
    market_brier = brier_score_multiclass(market_probs, oos_df["FTR"])
    market_weight = compute_blend_weight(model_brier, market_brier)

    blended = blend_probabilities(model_probs, market_probs, market_weight)
    oos_df["blend_prob_home"] = blended["prob_home"]
    oos_df["blend_prob_draw"] = blended["prob_draw"]
    oos_df["blend_prob_away"] = blended["prob_away"]
    blend_brier = brier_score_multiclass(blended, oos_df["FTR"])

    print("\n=== Resultados FUERA DE MUESTRA (walk-forward, honesto) ===")
    print(f"Partidos evaluados OOS: {len(oos_df)} (temporadas {ordered_seasons[1:]})")
    print(f"Brier score modelo propio:        {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.4f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.4f}")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    run()