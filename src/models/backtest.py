"""
Fase 3: Backtesting riguroso con walk-forward validation (ventana expansiva
por temporada).

A diferencia del diagnostico en-muestra de Fase 2 (train_poisson.py), aqui el
modelo NUNCA ve los resultados de la temporada que esta evaluando. Se entrena
con todas las temporadas anteriores y se predice unicamente la temporada
siguiente -- exactamente como se usaria el modelo en produccion real.

LIMITACION CONOCIDA #1: cada temporada llegan ~3 equipos recien ascendidos sin
historial en las temporadas de entrenamiento. El modelo no puede asignarles
un rating de ataque/defensa sin haber visto ningun partido suyo, asi que esos
partidos se EXCLUYEN de la evaluacion de esa temporada (se reporta cuantos).
Mejora futura: asignarles un rating "recien ascendido" por defecto en vez de
excluirlos.

LIMITACION CONOCIDA #2: no todas las temporadas tienen el archivo de cuotas
de CIERRE de Pinnacle completo en football-data.co.uk -- las mas recientes
(ej. la que acaba de terminar) suelen tener huecos porque ese archivo se
termina de publicar con retraso respecto al de apertura. Los partidos sin
cuota de cierre de Pinnacle se EXCLUYEN unicamente del calculo de Brier score
de mercado/blend (se reporta cuantos) -- el Brier score del modelo propio
SI los incluye, porque no depende de Pinnacle en absoluto.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model import build_long_format, fit_poisson_model, predict_dataframe
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]
PROB_COLS_RENAME = {"prob_home": "prob_home", "prob_draw": "prob_draw", "prob_away": "prob_away"}


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

    # --- Brier score del modelo propio: usa TODAS las filas OOS ---
    # (no depende de Pinnacle, asi que no hay razon para excluir nada aqui)
    model_probs_full = oos_df[MODEL_COLS].rename(columns={
        "model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away",
    })
    model_brier = brier_score_multiclass(model_probs_full, oos_df["FTR"])

    # --- Brier score de mercado / blend: SOLO filas con cuota de cierre de ---
    # Pinnacle completa (ver LIMITACION CONOCIDA #2 arriba).
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

    print("\n=== Resultados FUERA DE MUESTRA (walk-forward, honesto) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio:        {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.4f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.4f}")

    out_path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    run()