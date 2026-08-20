"""
Fase 3 v7: mismo walk-forward honesto que backtest_v4.py, parametrizado por
liga desde el arranque (a diferencia de v5/v6, que solo se probaron en EPL
-- acá la hipótesis (Elo, ver add_team_elo_features.py) está motivada
específicamente por el problema no resuelto de Serie A/Bundesliga, así que
tiene sentido medirla ahí directamente en vez de filtrar primero por EPL).

Se construye ENCIMA de v4 (recencia + PROMOTED_TEAM + forma reciente por
tiros al arco), agregando el rating Elo del equipo como covariable extra
(`poisson_model_v7.py`). Comparación directa impresa contra v4, MISMO
conjunto de partidos por liga (misma metodología de comparación honesta ya
usada en backtest_v4.py contra v2 en EPL).

REQUISITO PREVIO: correr 'python -m src.processing.add_team_form_features'
Y 'python -m src.processing.add_team_elo_features' sobre matches_clean.csv
antes de este script.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v7 import build_long_format_v7, fit_poisson_model_v7, predict_dataframe_v7
from src.models.poisson_model_v2 import DEFAULT_HALF_LIFE_DAYS
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]

# Referencia v4 por liga -- se completa a mano abajo antes de correr, leyendo el
# ultimo resultado confirmado de backtest_v4.py en el roadmap (Fase 8). Si algun
# valor no esta disponible se omite la comparacion para esa liga en vez de
# imprimir un numero inventado.
V4_REFERENCE = {
    "EPL": {"model_brier": 0.595578, "market_brier": 0.560801, "blend_brier": 0.568619, "gap": 0.007818},
    "LALIGA": {"model_brier": 0.589535, "market_brier": 0.572426, "blend_brier": 0.576078, "gap": 0.003652},
    "SERIEA": {"model_brier": 0.599352, "market_brier": 0.577595, "blend_brier": 0.584011, "gap": 0.006416},
    "BUNDESLIGA": {"model_brier": 0.607687, "market_brier": 0.576732, "blend_brier": 0.584806, "gap": 0.008074},
}


def run(league_key: str) -> None:
    data_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not data_path.exists():
        print(f"[SKIP] {league_key}: no existe {data_path} -- corre clean_data.py, "
              f"add_team_form_features.py y add_team_elo_features.py primero.")
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

        long_df = build_long_format_v7(train_df)
        model = fit_poisson_model_v7(long_df)
        preds = predict_dataframe_v7(model, test_df, known_teams)

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

    print(f"\n=== Resultados FUERA DE MUESTRA v7 [{league_key}] (v4 + Elo walk-forward por equipo) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio v7:      {model_brier:.6f}")
    print(f"Brier score mercado (Pinnacle):    {market_brier:.6f}")
    print(f"Peso asignado al mercado:          {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):  {blend_brier:.6f}")

    ref = V4_REFERENCE.get(league_key)
    if ref is not None:
        delta_model = model_brier - ref["model_brier"]
        delta_blend = blend_brier - ref["blend_brier"]
        print(f"\n(Referencia v4 [{league_key}] -- mismo conjunto de partidos, comparacion directa valida: "
              f"modelo {ref['model_brier']:.6f}, mercado {ref['market_brier']:.6f}, "
              f"blend {ref['blend_brier']:.6f}, gap +{ref['gap']:.6f})")
        print(f"Delta v7 vs v4 -- modelo: {delta_model:+.6f} ({'MEJORA' if delta_model < 0 else 'empeora'}), "
              f"blend: {delta_blend:+.6f} ({'MEJORA' if delta_blend < 0 else 'empeora'})")
    else:
        print(f"\n(Sin referencia v4 guardada para {league_key} en este script -- comparar a mano contra "
              f"el roadmap.)")

    out_path = PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v7.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"Guardado -> {out_path}")
    print("(NOTA: se guarda con nombre v7 propio, NO sobreescribe model_predictions_oos_walkforward_v4.csv "
          "-- v4 sigue siendo la referencia de produccion hasta que se decida adoptar v7 explicitamente.)")

    log_run(
        script="backtest_v7.py",
        model_name="poisson",
        model_version="v7",
        data_paths=[data_path],
        features=f"[{league_key}] goals ~ is_home + C(team) + C(opponent) + team_recent_form + team_elo "
                  "[+ filas sinteticas PROMOTED_TEAM], "
                  "freq_weights=decaimiento exponencial por recencia, "
                  "team_elo=rating Elo walk-forward por equipo (K=32, home_advantage=100, sin tunear, "
                  "ver add_team_elo_features.py)",
        hyperparameters={
            "league_key": league_key,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "recent_form_rolling_window": 5,
            "recent_form_stat": "shots_on_target_diff",
            "elo_k_factor": 32.0,
            "elo_home_advantage": 100.0,
            "elo_initial": 1500.0,
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
        notes=f"[{league_key}] Walk-forward v7: se agrega Elo walk-forward por equipo encima de v4 -- "
              "primera vez que el modelo de futbol pondera la fuerza del rival vencido, misma hipotesis "
              "que ya funciono en tenis esta sesion (2026-08-19/20). Construido sobre v4, no sobre v5/v6 "
              "(ambos resultado negativo documentado). NO reemplaza a v4 como referencia de produccion "
              "todavia -- exploratorio, requiere confirmar con la capa de evaluacion economica completa "
              "antes de adoptar.",
    )


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)