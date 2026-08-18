"""
Fase 8b: walk-forward para MLS, usando el modelo v2 (poisson_model_v2.py:
recencia por time-decay + rating para equipos sin historial), NO v4.

Por que v2 y no v4: v4 (el modelo de referencia en las 4 ligas europeas)
agrega forma reciente calculada a partir de TIROS AL ARCO por equipo
(add_team_form_features.py). MLS no tiene esa columna en absoluto (ver
clean_data.py y mls_loader.py, confirmado con datos reales) -- no es una
cuestion de correr el mismo codigo con un parametro distinto, el insumo
que v4 necesita no existe para esta liga. v2 es el techo real que
permiten estos datos: Poisson + Benter Boost + recencia por goles (no por
tiros) + una categoria sintetica para equipos sin historial de entrenamiento.

Sobre PROMOTED_TEAM en MLS: la funcion identify_promoted_teams_by_season()
de poisson_model_v2.py detecta equipos que aparecen en la temporada N pero
no en la N-1 -- en las ligas europeas eso son ascensos de categoria
inferior; en MLS son equipos de EXPANSION (franquicias nuevas: Austin FC
2021, Charlotte FC 2022, St. Louis City 2023, San Diego FC 2025, etc. --
MLS no tiene ascenso/descenso). El mecanismo es el mismo (equipo sin
historial propio en la ventana de entrenamiento) aunque la razon de
negocio sea distinta -- se reutiliza tal cual, sin modificar
poisson_model_v2.py.

Temporadas: a diferencia de las 4 ligas europeas (que usan SEASONS de
config/settings.py, formato "2324"), MLS trae su propia columna 'season'
con años calendario simples ("2012".."2026", ver mls_loader.py). Se
detectan dinamicamente del propio dataset en vez de hardcodear una lista
nueva en settings.py -- evita otra fuente de desincronizacion si football-
data.co.uk agrega una temporada nueva.

Benchmark de mercado: pinnacle_close_prob_home/draw/away (MLS solo tiene
cierre, no apertura -- ver clean_data.py) -- mismos nombres de columna que
usa backtest_v2.py para las 4 ligas europeas, asi que MARKET_COLS no
necesita ningun cambio.

Salida: data/processed/MLS/model_predictions_oos_walkforward_mls.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.poisson_model_v2 import (
    build_long_format_v2, fit_poisson_model_v2, predict_dataframe_v2, DEFAULT_HALF_LIFE_DAYS,
)
from src.models.blending import brier_score_multiclass, compute_blend_weight, blend_probabilities
from src.tracking.run_logger import log_run

LEAGUE_KEY = "MLS"
MARKET_COLS = ["pinnacle_close_prob_home", "pinnacle_close_prob_draw", "pinnacle_close_prob_away"]
MODEL_COLS = ["model_prob_home", "model_prob_draw", "model_prob_away"]
MIN_TRAIN_MATCHES = 100  # primeras temporadas de MLS pueden tener pocos partidos -- ver aviso abajo


def run() -> None:
    path = PROCESSED_DATA_DIR / LEAGUE_KEY / "matches_clean.csv"
    if not path.exists():
        print(f"[SKIP] No existe {path}. Corre 'python -m src.processing.clean_data' primero.")
        return

    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    ordered_seasons = sorted(df["season"].unique(), key=lambda s: int(s))
    print(f"Temporadas detectadas en el dataset de MLS: {ordered_seasons}")

    all_oos_records = []
    for i in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:i]
        test_season = ordered_seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]

        if test_df.empty:
            print(f"[SKIP] Temporada {test_season}: sin partidos.")
            continue
        if len(train_df) < MIN_TRAIN_MATCHES:
            print(f"[AVISO] Temporada {test_season}: solo {len(train_df)} partidos de entrenamiento "
                  f"acumulados (< {MIN_TRAIN_MATCHES}) -- fold incluido igual, pero tratar sus "
                  f"resultados con mas cautela que folds con mas historial detras.")

        known_teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
        print(f"Entrenando con {train_seasons} ({len(train_df)} partidos) -> evaluando {test_season} "
              f"({len(test_df)} partidos)...")

        long_df = build_long_format_v2(train_df)
        model = fit_poisson_model_v2(long_df)
        preds = predict_dataframe_v2(model, test_df, known_teams)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    if not all_oos_records:
        print("[SKIP] No se pudo evaluar ningun fold (historial insuficiente).")
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
              f"sin cuota de cierre de Pinnacle disponible.")

    market_subset = oos_df.loc[has_market]
    market_probs = market_subset[MARKET_COLS].rename(columns={
        "pinnacle_close_prob_home": "prob_home", "pinnacle_close_prob_draw": "prob_draw",
        "pinnacle_close_prob_away": "prob_away",
    })
    model_probs_subset = model_probs_full.loc[has_market]

    market_brier = brier_score_multiclass(market_probs, market_subset["FTR"])
    market_weight = compute_blend_weight(model_brier, market_brier)
    blended = blend_probabilities(model_probs_subset, market_probs, market_weight)
    blend_brier = brier_score_multiclass(blended, market_subset["FTR"])

    oos_df.loc[has_market, "blend_prob_home"] = blended["prob_home"].values
    oos_df.loc[has_market, "blend_prob_draw"] = blended["prob_draw"].values
    oos_df.loc[has_market, "blend_prob_away"] = blended["prob_away"].values

    print(f"\n=== Resultados FUERA DE MUESTRA MLS (modelo v2: recencia + equipos sin historial) ===")
    print(f"Partidos evaluados OOS: {n_total} (temporadas {ordered_seasons[1:]})")
    print(f"Partidos con cuota de cierre de Pinnacle disponible: {n_market}")
    print(f"Brier score modelo propio (v2):     {model_brier:.4f}")
    print(f"Brier score mercado (Pinnacle cierre): {market_brier:.4f}")
    print(f"Peso asignado al mercado:           {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):   {blend_brier:.4f}")
    print("\nIMPORTANTE: esto NO es directamente comparable con el Brier de v4 en las 4 ligas "
          "europeas -- MLS corre sobre arquitectura v2 (sin features de tiros al arco, que no "
          "existen para esta liga) y el mercado de referencia es la cuota de CIERRE (no hay "
          "apertura), no la de apertura como en las otras 4. Comparar MLS contra MLS en el "
          "tiempo, no contra el numero de EPL/LaLiga/SerieA/Bundesliga.")

    out_path = PROCESSED_DATA_DIR / LEAGUE_KEY / "model_predictions_oos_walkforward_mls.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_mls.py",
        model_name="poisson",
        model_version="v2_mls",
        data_paths=[path],
        features="[MLS] goals ~ is_home + C(team) + C(opponent) [+ filas sinteticas para equipos de "
                  "expansion sin historial], freq_weights=decaimiento exponencial por recencia. "
                  "Arquitectura v2 (sin tiros al arco -- no disponibles para MLS). Mercado = cierre "
                  "de Pinnacle (sin apertura disponible).",
        hyperparameters={"league_key": LEAGUE_KEY, "half_life_days": DEFAULT_HALF_LIFE_DAYS,
                          "min_train_matches": MIN_TRAIN_MATCHES},
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
        notes="[MLS] Fase 8b -- primera corrida walk-forward de MLS, arquitectura v2 (no v4, sin "
              "datos de tiros al arco). Mercado de referencia = cierre de Pinnacle, sin CLV "
              "disponible para esta liga.",
    )


if __name__ == "__main__":
    run()