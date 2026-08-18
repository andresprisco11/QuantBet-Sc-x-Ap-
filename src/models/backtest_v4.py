"""
Fase 3 v4: mismo walk-forward honesto que backtest_v2.py, pero usando el
modelo v4 (poisson_model_v4.py): recencia + PROMOTED_TEAM (heredado de v2)
MAS la primera variable real de feature engineering -- forma reciente por
tiros al arco (home_recent_st_diff / away_recent_st_diff).

Objetivo de esta corrida: despues de dos resultados negativos consecutivos
ajustando la ESTRUCTURA del Poisson (half-life, Dixon-Coles), esta es la
primera prueba de si agregar SEÑAL NUEVA (en vez de mas matematica sobre la
misma señal de goles) mueve la aguja. Comparacion directa contra v2 (EPL):
mismo conjunto de 1,900 partidos.

REQUISITO PREVIO: correr 'python -m src.processing.add_team_form_features'
al menos una vez sobre matches_clean.csv antes de este script -- si no,
poisson_model_v4.py va a fallar con un error claro pidiendo que lo corras.

Fix 2026-08-18 (Fase 8, multi-liga): run() estaba hardcodeado a
PROCESSED_DATA_DIR / "EPL" (entrada, salida de predicciones, y notas del
log de tracking) -- a diferencia de poisson_model_v4.py, que ya era
agnostico de liga (recibe DataFrames por parametro, no lee/escribe nada
el mismo). Se parametriza run() por league_key y se loopea sobre LEAGUES
en __main__, mismo patron que clean_data.py / football_data_loader.py /
add_team_form_features.py. Se entrena un modelo INDEPENDIENTE por liga
(decision documentada en el roadmap, Fase 8: "por ahora, separado por
liga" -- pooling entre ligas es metodologia nueva sin probar, mismo
criterio de no agregar sofisticacion sin medirla que ya costo v5/v6).

Se agrega ademas: la comparacion impresa contra v2 al final tenia los
numeros de v2 HARDCODEADOS (0.598631/0.560801/0.569562/+0.008761) -- esos
son especificos del backtest de v2 sobre EPL. Si esto corriera para
La Liga/Serie A/Bundesliga sin cambios, mostraria una comparacion FALSA
(numeros de otra liga) al lado del resultado real de esa liga. Se
condiciona esa impresion a league_key == "EPL" -- para las ligas nuevas
todavia no hay un backtest v2 de referencia con el que comparar (nunca se
corrio v2 aislado para ellas, se arranca directo en v4).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v4 import build_long_format_v4, fit_poisson_model_v4, predict_dataframe_v4
from src.models.poisson_model_v2 import DEFAULT_HALF_LIFE_DAYS
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]

# Referencia v2, EPL unicamente (nunca se corrio v2 aislado para las ligas
# nuevas -- arrancaron directo en v4). Solo se imprime cuando league_key == "EPL".
V2_EPL_REFERENCE = {
    "model_brier": 0.598631, "market_brier": 0.560801, "blend_brier": 0.569562, "gap": 0.008761,
}


def run(league_key: str) -> None:
    data_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not data_path.exists():
        print(f"[SKIP] {league_key}: no existe {data_path} -- corre clean_data.py y "
              f"add_team_form_features.py primero.")
        return

    df = pd.read_csv(data_path)
    df["season"] = df["season"].astype(str)
    ordered_seasons = SEASONS
    all_oos_records = []

    print(f"\n=== {league_key} ===")
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

        long_df = build_long_format_v4(train_df)
        model = fit_poisson_model_v4(long_df)
        preds = predict_dataframe_v4(model, test_df, known_teams)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    if not all_oos_records:
        print(f"[SKIP] {league_key}: no hubo ningun fold OOS evaluable.")
        return

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

    print(f"\n=== Resultados FUERA DE MUESTRA v4 [{league_key}] (recencia + PROMOTED_TEAM + forma reciente por tiros) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio v4:      {model_brier:.6f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.6f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.6f}")

    if league_key == "EPL":
        ref = V2_EPL_REFERENCE
        print(f"\n(Referencia v2 EPL -- mismo conjunto de partidos, comparacion directa valida: "
              f"modelo {ref['model_brier']:.6f}, mercado {ref['market_brier']:.6f}, "
              f"blend {ref['blend_brier']:.6f}, gap +{ref['gap']:.6f})")
    else:
        print(f"\n(Sin referencia v2 para {league_key} -- nunca se corrio v2 aislado para esta liga, "
              f"arranco directo en v4. Comparar contra el propio mercado de esta liga, arriba, no contra EPL.)")

    out_path = PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v4.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"Guardado -> {out_path}")

    log_run(
        script="backtest_v4.py",
        model_name="poisson",
        model_version="v4",
        data_paths=[data_path],
        features=f"[{league_key}] goals ~ is_home + C(team) + C(opponent) + team_recent_form "
                  "[+ filas sinteticas PROMOTED_TEAM], "
                  "freq_weights=decaimiento exponencial por recencia, "
                  "team_recent_form=promedio movil de diferencial de tiros al arco (ultimos 5 partidos, sin fuga)",
        hyperparameters={
            "league_key": league_key,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "recent_form_rolling_window": 5,
            "recent_form_stat": "shots_on_target_diff",
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
        notes=f"[{league_key}] Walk-forward v4: primera variable de feature engineering (forma reciente por "
              "tiros al arco) encima de v2 (recencia + PROMOTED_TEAM). No incluye Dixon-Coles (v3, resultado "
              "negativo documentado en EPL). Modelo entrenado de forma INDEPENDIENTE por liga (Fase 8, "
              "2026-08-18) -- sin pooling entre ligas todavia.",
    )


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)