"""
Fase 10 (NBA) -- backtest walk-forward de NBA v2 (Elo + b2b_diff), mismo
patron exacto que `backtest_nba.py` pero importando `nba_margin_model_v2`
en vez de v1. Guarda en un path DISTINTO (no pisa las predicciones de v1)
y al final imprime un bloque `[COMPARAR CONTRA v1]` con los numeros v1 ya
confirmados (2026-08-21) citados inline, para poder leer la comparacion
sin tener que ir a buscar el log de la corrida anterior.

**Nota sobre la variable v2**: el primer intento fue `rest_diff` continuo
(dias exactos de descanso, mismo camino que NFL) -- se probo con OLS sobre
los 35,546 partidos y NO fue significativo (p=0.335). Se reemplazo por un
indicador BINARIO de back-to-back (`b2b_diff`), que si es significativo
(p<0.0001) -- ver docstring completo en `nba_margin_model_v2.py`. Este
archivo importa la version final (binaria), nunca se corrio walk-forward
con la version continua.

Uso: python -m src.models.backtest_nba_v2
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nba_margin_model_v2 import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH_WITH_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean_with_odds.csv"
DATA_PATH_NO_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"

# Numeros v1 YA CONFIRMADOS (2026-08-21, corrida real del usuario, ver roadmap) --
# citados aca solo para comparar en pantalla, no se recalculan.
V1_BASELINE = {
    "model_brier_all_seasons": 0.209843,
    "n_reliable_market": 6904,
    "model_brier_reliable_subset": 0.218478,
    "market_brier": 0.203437,
    "blend_brier": 0.207550,
    "gap_vs_mercado": 0.004114,
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
    if "home_rest" not in df.columns:
        print("[SKIP] Falta home_rest/away_rest -- corre 'python -m src.processing.add_nba_rest_features' primero "
              "(el binario b2b_diff se deriva de esas columnas dentro de nba_margin_model_v2.py).")
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
        print(f"\n[INFO] {n_no_pred} partidos OOS sin prediccion (b2b_diff no calculable, primer "
              f"partido de una franquicia en ese punto del historico) -- excluidos de las metricas.")
    oos_df = oos_df[oos_df["model_prob_home"].notna()]
    n_total = len(oos_df)

    model_brier_all = _brier_binary(oos_df["model_prob_home"], oos_df["FTR"])
    print(f"\n=== Resultados FUERA DE MUESTRA -- NBA v2 (Elo + b2b_diff) ===")
    print(f"Partidos evaluados OOS: {n_total}")
    print(f"Brier score modelo propio (TODAS las temporadas OOS): {model_brier_all:.6f}")
    print(f"[COMPARAR CONTRA v1] model_brier_all_seasons v1={V1_BASELINE['model_brier_all_seasons']:.6f} "
          f"-> v2={model_brier_all:.6f} ({'MEJORA' if model_brier_all < V1_BASELINE['model_brier_all_seasons'] else 'NO mejora'})")

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
        print(f"Brier score modelo propio:          v1={V1_BASELINE['model_brier_reliable_subset']:.6f} -> v2={model_brier:.6f}")
        print(f"Brier score mercado (no deberia cambiar, mismo mercado): v1={V1_BASELINE['market_brier']:.6f} -> v2={market_brier:.6f}")
        print(f"Peso asignado al mercado:           {market_weight:.1%}")
        print(f"Brier score blend (Benter Boost):   v1={V1_BASELINE['blend_brier']:.6f} -> v2={blend_brier:.6f}")
        print(f"Gap blend vs. mercado:               v1={V1_BASELINE['gap_vs_mercado']:+.6f} -> v2={gap:+.6f} "
              f"({'el blend gana' if gap < 0 else 'el mercado sigue ganando'})")
        print(f"\n[COMPARAR CONTRA v1] gap_vs_mercado v1={V1_BASELINE['gap_vs_mercado']:+.6f} -> v2={gap:+.6f} "
              f"({'MEJORA' if gap < V1_BASELINE['gap_vs_mercado'] else 'NO mejora'})")

        metrics.update({
            "n_reliable_market": n_reliable,
            "model_brier_reliable_subset": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": gap,
        })

    out_path = PROCESSED_DATA_DIR / "NBA" / "model_predictions_oos_walkforward_v2.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nba_v2.py",
        model_name="nba_margin_normal",
        model_version="v2_elo_b2b",
        data_paths=[data_path],
        features="point_margin ~ elo_diff + b2b_diff (home_b2b - away_b2b, binario de back-to-back "
                 "derivado de home_rest==0/away_rest==0). rest_diff CONTINUO probado primero y "
                 "descartado por no significativo (p=0.335, ver docstring de nba_margin_model_v2.py).",
        hyperparameters={
            "elo_k_factor": 20.0, "elo_home_advantage": 100.0, "elo_season_regression": 0.25,
        },
        metrics=metrics,
        predictions_path=out_path,
        notes="v2 de NBA -- agrega b2b_diff (binario, no continuo) a elo_diff como segundo regresor OLS. "
              "A diferencia de NFL (rest_diff continuo funciono ahi), en NBA el efecto de descanso es de "
              "umbral (back-to-back especifico), no lineal en dias -- confirmado con OLS full-sample "
              "(b2b_diff coef=-1.968, t=-15.84, p<0.0001) antes de correr el walk-forward. "
              "Comparado inline contra v1 (ver [COMPARAR CONTRA v1] en el output).",
    )


if __name__ == "__main__":
    run()
