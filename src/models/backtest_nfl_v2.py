"""
Fase 10, v2 -- backtest walk-forward de NFL con el modelo de margen
extendido (nfl_margin_model_v2.py: elo_diff + rest_diff). Metodologia
IDENTICA a backtest_nfl.py (v1) en todo lo demas -- mismo walk-forward por
temporada, misma exclusion explicita de partidos sin mercado confiable
(season<2010), misma formula de blend "Benter Boost" -- para que la
comparacion v1 vs v2 aisle el efecto del feature nuevo (rest_diff) y no
sea un artefacto de metodologia distinta.

UNICA diferencia real de codigo frente a backtest_nfl.py: importa
fit_margin_model/predict_dataframe de nfl_margin_model_v2 en vez de
nfl_margin_model (v1), y guarda en un archivo de salida distinto para no
pisar los resultados de v1 (necesarios para comparar).

Requiere que matches_clean.csv tenga home_elo/away_elo (add_nfl_elo_features.py)
Y home_rest/away_rest (confirmado presente en el CSV real, 2026-08-20 --
ver nfl_margin_model_v2.py).

Uso: python -m src.models.backtest_nfl_v2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nfl_margin_model_v2 import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "matches_clean.csv"


def _brier_binary(model_prob_home: pd.Series, ftr: pd.Series) -> float:
    """Identica a backtest_nfl.py -- empate = 0.5, mismo criterio que el
    resultado 'R' de Elo (ver add_nfl_elo_features.py)."""
    actual = ftr.map({"H": 1.0, "A": 0.0, "T": 0.5})
    return float(((model_prob_home - actual) ** 2).mean())


def _blend_weight(model_brier: float, market_brier: float) -> float:
    """Identica a backtest_nfl.py -- 'Benter Boost' binario, mas peso al
    lado con menor Brier."""
    return model_brier / (model_brier + market_brier)


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre clean_nfl_data.py y add_nfl_elo_features.py primero.")
        return

    df = pd.read_csv(DATA_PATH)
    if "home_elo" not in df.columns:
        print("[SKIP] Falta home_elo/away_elo -- corre 'python -m src.processing.add_nfl_elo_features' primero.")
        return
    if "home_rest" not in df.columns or "away_rest" not in df.columns:
        print("[SKIP] Falta home_rest/away_rest en matches_clean.csv -- confirmar de donde salio este CSV, "
              "estas columnas deberian estar desde clean_nfl_data.py.")
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
    n_total = len(oos_df)

    model_brier_all = _brier_binary(oos_df["model_prob_home"], oos_df["FTR"])
    print(f"\n=== Resultados FUERA DE MUESTRA -- NFL v2, moneyline (elo_diff + rest_diff -> Normal -> P(gana local)) ===")
    print(f"Partidos evaluados OOS (todas las temporadas, {ordered_seasons[1]}-{ordered_seasons[-1]}): {n_total}")
    print(f"Brier score modelo propio (TODAS las temporadas OOS, incluye las sin mercado confiable): {model_brier_all:.6f}")

    reliable = oos_df["reliable_moneyline"] & oos_df["market_prob_home"].notna()
    n_reliable = int(reliable.sum())
    n_excluded = n_total - n_reliable
    print(f"\n[AVISO] {n_excluded} de {n_total} partidos OOS excluidos de la comparacion contra mercado: "
          f"sin moneyline confiable (season<2010) o sin probabilidad de mercado calculable.")

    subset = oos_df.loc[reliable]
    model_brier = _brier_binary(subset["model_prob_home"], subset["FTR"])
    market_brier = _brier_binary(subset["market_prob_home"], subset["FTR"])
    market_weight = _blend_weight(model_brier, market_brier)
    blended = market_weight * subset["market_prob_home"] + (1.0 - market_weight) * subset["model_prob_home"]
    blend_brier = _brier_binary(blended, subset["FTR"])

    print(f"\n--- Subconjunto con mercado confiable (season>=2010): {n_reliable} partidos ---")
    print(f"Brier score modelo propio (v2, con rest_diff): {model_brier:.6f}")
    print(f"Brier score mercado (no-vig, moneyline americano): {market_brier:.6f}")
    print(f"Peso asignado al mercado:           {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):   {blend_brier:.6f}")
    print(f"Gap blend vs. mercado:               {blend_brier - market_brier:+.6f} "
          f"({'el blend gana' if blend_brier < market_brier else 'el mercado sigue ganando'})")

    print(f"\n[COMPARAR CONTRA v1] backtest_nfl.py (v1, solo elo_diff) dio, en la misma corrida original: "
          f"model_brier=0.220293, market_brier=0.210574, market_weight=51.1%, blend_brier=0.213031, "
          f"gap=+0.002457 sobre n=4362. Comparar model_brier y gap de esta corrida contra esos numeros -- "
          f"un model_brier MENOR y/o un gap MAS CHICO (mas cerca de 0, o negativo si el blend empieza a "
          f"ganarle al mercado) confirmaria que rest_diff aporta señal real. Si los numeros son practicamente "
          f"iguales o peores, rest_diff no esta ayudando y no vale la pena adoptarlo.")

    out_path = PROCESSED_DATA_DIR / "NFL" / "model_predictions_oos_walkforward_v2.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nfl_v2.py",
        model_name="nfl_margin_normal",
        model_version="v2",
        data_paths=[DATA_PATH],
        features="point_margin ~ elo_diff (home_elo - away_elo) + rest_diff (home_rest - away_rest), "
                 "sigma = desvio de residuos de training, Elo con ajuste MOV + regresion a la media "
                 "entre temporadas (ver add_nfl_elo_features.py)",
        hyperparameters={
            "elo_k_factor": 20.0,
            "elo_home_advantage": 48.0,
            "elo_season_regression": 1.0 / 3.0,
            "reliable_odds_start_season": 2010,
        },
        metrics={
            "n_total": n_total,
            "n_reliable_market": n_reliable,
            "model_brier_all_seasons": model_brier_all,
            "model_brier_reliable_subset": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": blend_brier - market_brier,
        },
        predictions_path=out_path,
        notes="v2 del modelo de margen de NFL -- agrega rest_diff (dias de descanso) como segundo "
              "regresor junto a elo_diff. Comparar contra v1 (backtest_nfl.py) antes de adoptar.",
    )


if __name__ == "__main__":
    run()
