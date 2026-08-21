"""
Fase 10 (futbol) v8: mismo walk-forward honesto que backtest_v7.py,
parametrizado por liga desde el arranque. Retoma la hipotesis (c) pausada
en Fase 8/Fase 9 -- xG (`poisson_model_v8.py`, construido ENCIMA de v4, no
de v7 -- Elo fue rechazado, ver roadmap Fase 9) como covariable nueva
despues de que v5 (ataque/defensa), v6 (corners) y v7 (Elo) fallaron.

Comparacion directa contra v4, MISMO conjunto de partidos por liga (misma
metodologia ya usada en backtest_v7.py).

LIMITACION REAL, documentada explicitamente (no oculta): la cobertura de xG
es parcial (temporadas >=2022/23 solamente, ver add_team_xg_features.py) --
los folds de test tempranos (2122, 2223 y en parte 2324) entrenan con poco o
ningun xG real en la ventana de forma reciente (prior neutro 0.0 domina), asi
que cualquier mejora real deberia concentrarse en los folds de test mas
recientes (2425, 2526, 2627), donde el training set ya acumulo temporadas
completas con xG real. Este script separa esa lectura explicitamente en vez
de promediar todo junto y esconder la diferencia.

REQUISITO PREVIO: correr 'python -m src.processing.merge_thestatsapi_xg --all',
'python -m src.processing.add_team_form_features' y
'python -m src.processing.add_team_xg_features' sobre matches_clean.csv
antes de este script.

Uso: python -m src.models.backtest_v8
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v8 import build_long_format_v8, fit_poisson_model_v8, predict_dataframe_v8
from src.models.poisson_model_v2 import DEFAULT_HALF_LIFE_DAYS
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]

# Referencia v4 por liga -- mismos valores confirmados que usa backtest_v7.py
# (Fase 8, roadmap). Si no esta disponible se omite la comparacion en vez de
# inventar un numero.
V4_REFERENCE = {
    "EPL": {"model_brier": 0.595578, "market_brier": 0.560801, "blend_brier": 0.568619, "gap": 0.007818},
    "LALIGA": {"model_brier": 0.589535, "market_brier": 0.572426, "blend_brier": 0.576078, "gap": 0.003652},
    "SERIEA": {"model_brier": 0.599352, "market_brier": 0.577595, "blend_brier": 0.584011, "gap": 0.006416},
    "BUNDESLIGA": {"model_brier": 0.607687, "market_brier": 0.576732, "blend_brier": 0.584806, "gap": 0.008074},
}

# Folds de test donde el training set ya acumulo al menos una temporada
# completa con xG real (>=2223) -- lectura "informativa" separada de la
# lectura completa (que incluye folds tempranos dominados por el prior
# neutro 0.0, ver docstring del modulo).
INFORMATIVE_TEST_SEASONS = {"2324", "2425", "2526", "2627"}


def _brier_for_subset(oos_df: pd.DataFrame) -> dict:
    model_probs_full = oos_df[MODEL_COLS].rename(columns={
        "model_prob_home": "prob_home", "model_prob_draw": "prob_draw", "model_prob_away": "prob_away",
    })
    model_brier = brier_score_multiclass(model_probs_full, oos_df["FTR"])

    has_market = oos_df[MARKET_COLS].notna().all(axis=1)
    n_total = len(oos_df)
    n_market = int(has_market.sum())

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
        "n_total": n_total, "n_market": n_market,
        "model_brier": model_brier, "market_brier": market_brier,
        "market_weight": market_weight, "blend_brier": blend_brier,
    }


def run(league_key: str) -> None:
    data_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not data_path.exists():
        print(f"[SKIP] {league_key}: no existe {data_path}.")
        return

    df = pd.read_csv(data_path)
    if "home_recent_xg_diff" not in df.columns:
        print(f"[SKIP] {league_key}: falta home_recent_xg_diff -- corre "
              f"'python -m src.processing.merge_thestatsapi_xg --all' y luego "
              f"'python -m src.processing.add_team_xg_features' primero.")
        return

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
        n_train_with_xg = int((train_df["home_recent_xg_diff"] != 0.0).sum() + (train_df["away_recent_xg_diff"] != 0.0).sum())
        print(f"Entrenando con {train_seasons} ({len(train_df)} partidos, senal xG real en "
              f"{n_train_with_xg} observaciones team-partido) -> evaluando {test_season} ({len(test_df)} partidos)...")

        long_df = build_long_format_v8(train_df)
        model = fit_poisson_model_v8(long_df)
        preds = predict_dataframe_v8(model, test_df, known_teams)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1).copy()
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    if not all_oos_records:
        print(f"[SKIP] {league_key}: no hubo ningun fold OOS evaluable.")
        return

    oos_df = pd.concat(all_oos_records, ignore_index=True)

    full_metrics = _brier_for_subset(oos_df)
    informative_df = oos_df[oos_df["fold_test_season"].isin(INFORMATIVE_TEST_SEASONS)]
    informative_metrics = _brier_for_subset(informative_df) if not informative_df.empty else None

    print(f"\n=== Resultados FUERA DE MUESTRA v8 [{league_key}] (v4 + forma reciente de xG, walk-forward) ===")
    print(f"Partidos evaluados OOS (todos los folds, temporadas {ordered_seasons[1:]}): {full_metrics['n_total']}")
    print(f"Brier score modelo propio v8 (todos los folds):      {full_metrics['model_brier']:.6f}")
    print(f"Brier score mercado (Pinnacle):                      {full_metrics['market_brier']:.6f}")
    print(f"Brier score blend (Benter Boost, todos los folds):   {full_metrics['blend_brier']:.6f}")

    ref = V4_REFERENCE.get(league_key)
    if ref is not None:
        delta_model = full_metrics["model_brier"] - ref["model_brier"]
        delta_blend = full_metrics["blend_brier"] - ref["blend_brier"]
        print(f"\n(Referencia v4 [{league_key}]: modelo {ref['model_brier']:.6f}, mercado {ref['market_brier']:.6f}, "
              f"blend {ref['blend_brier']:.6f}, gap +{ref['gap']:.6f})")
        print(f"Delta v8 vs v4 (todos los folds) -- modelo: {delta_model:+.6f} "
              f"({'MEJORA' if delta_model < 0 else 'empeora'}), blend: {delta_blend:+.6f} "
              f"({'MEJORA' if delta_blend < 0 else 'empeora'})")
    else:
        print(f"\n(Sin referencia v4 guardada para {league_key}.)")

    if informative_metrics is not None:
        print(f"\n--- Subconjunto INFORMATIVO (test={sorted(INFORMATIVE_TEST_SEASONS)}, training ya con >=1 "
              f"temporada real de xG) ---")
        print(f"Partidos evaluados: {informative_metrics['n_total']}")
        print(f"Brier score modelo propio v8:      {informative_metrics['model_brier']:.6f}")
        print(f"Brier score mercado (Pinnacle):     {informative_metrics['market_brier']:.6f}")
        print(f"Brier score blend (Benter Boost):   {informative_metrics['blend_brier']:.6f}")
        print(f"[AVISO] Esta es la lectura mas honesta de si xG aporta senal real -- los folds tempranos "
              f"(arriba) diluyen cualquier efecto porque el training set no tenia xG real todavia.")
    else:
        print(f"\n[AVISO] Sin folds en el subconjunto informativo para {league_key}.")

    out_path = PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v8.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("(NOTA: nombre propio v8, NO sobreescribe matches_clean.csv ni el archivo de v4 -- v4 sigue "
          "siendo la referencia de produccion hasta que se decida adoptar v8 explicitamente.)")

    log_run(
        script="backtest_v8.py",
        model_name="poisson",
        model_version="v8",
        data_paths=[data_path],
        features=f"[{league_key}] goals ~ is_home + C(team) + C(opponent) + team_recent_form + team_xg_form "
                  "[+ filas sinteticas PROMOTED_TEAM], freq_weights=decaimiento exponencial por recencia, "
                  "team_xg_form=diferencial neto de xG de los ultimos 5 partidos (shift(1), prior neutro 0.0 "
                  "donde no hay xG real -- cobertura parcial, ver add_team_xg_features.py)",
        hyperparameters={
            "league_key": league_key,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "recent_form_rolling_window": 5,
            "recent_form_stat": "shots_on_target_diff",
            "xg_form_rolling_window": 5,
            "xg_coverage_seasons": "2223+",
        },
        metrics={
            "n_total": full_metrics["n_total"],
            "model_brier": full_metrics["model_brier"],
            "market_brier": full_metrics["market_brier"],
            "blend_brier": full_metrics["blend_brier"],
            "gap_vs_mercado": full_metrics["blend_brier"] - full_metrics["market_brier"],
            **({f"informative_{k}": v for k, v in informative_metrics.items()} if informative_metrics else {}),
        },
        predictions_path=out_path,
        notes=f"[{league_key}] Walk-forward v8: se agrega forma reciente de xG (TheStatsAPI, cobertura "
              "parcial >=2022/23) encima de v4 -- retoma la hipotesis (c) pausada en Fase 8 despues de que "
              "v5 (ataque/defensa), v6 (corners) y v7 (Elo) fallaran. Construido sobre v4, NO sobre v7. "
              "Ver subconjunto informativo (folds con training ya con xG real) para la lectura mas honesta. "
              "NO reemplaza a v4 como referencia de produccion todavia -- exploratorio.",
    )


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)
