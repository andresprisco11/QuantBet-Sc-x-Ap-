"""
Fase 8b: diagnostico de calibracion para MLS -- mismo framework que
calibration_analysis.py (unroll partido x resultado posible, calibracion
por decil de probabilidad, por rango de cuota, por lado), pero leyendo
la cuota de CIERRE (PSCH/PSCD/PSCA) en vez de apertura (PSH/PSD/PSA) --
MLS no tiene apertura (ver economic_backtest_mls.py, mismo motivo).

Por que hace falta esto antes de sumar MLS al pooling cross-liga de Tier 1
(tier1_parlay_validation.py, ver roadmap "Proximos pasos"): tanto
tier1_probability_validation.py como selection_bias_check.py dependen del
archivo 'calibration_analysis_v4_long.csv' que este tipo de script genera
-- sin su equivalente para MLS, no se puede correr la misma validacion de
"cuando el modelo dice >=80% de confianza, acierta >=80% de las veces?"
sobre MLS antes de mezclarla con las 4 ligas europeas en un pool de alta
confianza.

Fuente: 'model_predictions_oos_walkforward_mls.csv' (backtest_mls.py) --
no entrena nada nuevo, solo re-analiza.

Salida: data/processed/MLS/calibration_analysis_mls_long.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.tracking.run_logger import log_run

LEAGUE_KEY = "MLS"
N_PROB_BINS = 10
ODDS_BIN_EDGES = [1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
ODDS_BIN_LABELS = ["1.00-1.50", "1.50-2.00", "2.00-3.00", "3.00-5.00", "5.00+"]

# Cierre de Pinnacle (PSCH/PSCD/PSCA) -- MLS no tiene apertura (PSH/PSD/PSA).
SIDES = [
    ("home", "blend_prob_home", "PSCH", "H"),
    ("draw", "blend_prob_draw", "PSCD", "D"),
    ("away", "blend_prob_away", "PSCA", "A"),
]


def _unroll(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for side_name, prob_col, odds_col, ftr_code in SIDES:
            prob = row[prob_col]
            odds = row[odds_col]
            if pd.isna(prob) or pd.isna(odds):
                continue
            records.append({
                "season": row["fold_test_season"],
                "side": side_name,
                "predicted_prob": prob,
                "odds": odds,
                "actual": 1 if row["FTR"] == ftr_code else 0,
            })
    return pd.DataFrame(records)


def _calibration_table(long_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = long_df.groupby(group_col, observed=True).agg(
        n=("actual", "count"),
        predicted_mean=("predicted_prob", "mean"),
        observed_freq=("actual", "mean"),
    )
    table["gap"] = table["observed_freq"] - table["predicted_mean"]
    return table


def _expected_calibration_error(table: pd.DataFrame) -> float:
    total_n = table["n"].sum()
    return float((table["n"] * table["gap"].abs()).sum() / total_n)


def run() -> None:
    print(f"\n=== {LEAGUE_KEY} ===")
    path = PROCESSED_DATA_DIR / LEAGUE_KEY / "model_predictions_oos_walkforward_mls.csv"
    if not path.exists():
        print(f"[SKIP] No existe {path}. Corre 'python -m src.models.backtest_mls' primero.")
        return

    df = pd.read_csv(path)
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df_eval = df.loc[has_blend].copy()

    long_df = _unroll(df_eval)
    print(f"Filas desenrolladas (3 por partido, {len(df_eval)} partidos evaluados): {len(long_df)}")
    base_rates = long_df.groupby("side", observed=True)["actual"].mean().to_dict()
    print(f"Frecuencia base observada por resultado: {base_rates}")

    long_df["prob_decile"] = pd.qcut(long_df["predicted_prob"], q=N_PROB_BINS, duplicates="drop")
    prob_table = _calibration_table(long_df, "prob_decile")
    ece_prob = _expected_calibration_error(prob_table)

    print(f"\n=== [MLS] Calibracion por decil de probabilidad predicha (blend v2) ===")
    print(prob_table.round(4).to_string())
    print(f"\nExpected Calibration Error (ECE), ponderado por decil de probabilidad: {ece_prob:.4f}")

    long_df["odds_bin"] = pd.cut(long_df["odds"], bins=ODDS_BIN_EDGES, labels=ODDS_BIN_LABELS)
    odds_table = _calibration_table(long_df, "odds_bin")
    ece_odds = _expected_calibration_error(odds_table)

    print(f"\n=== [MLS] Calibracion por rango de cuota (CIERRE -- compara directo con "
          f"economic_backtest_mls.py) ===")
    print(odds_table.round(4).to_string())
    print(f"\nExpected Calibration Error (ECE), ponderado por rango de cuota: {ece_odds:.4f}")

    side_table = _calibration_table(long_df, "side")
    print(f"\n=== [MLS] Calibracion por lado (home/draw/away) ===")
    print(side_table.round(4).to_string())

    out_path = PROCESSED_DATA_DIR / LEAGUE_KEY / "calibration_analysis_mls_long.csv"
    long_df.to_csv(out_path, index=False)
    print(f"\nGuardado detalle desenrollado (partido x resultado posible) -> {out_path}")

    log_run(
        script="calibration_analysis_mls.py",
        model_name="poisson",
        model_version="v2_mls",
        data_paths=[path],
        features="[MLS] Diagnostico de calibracion sobre blend_prob_* ya generado por backtest_mls.py -- "
                  "no entrena modelo nuevo. Cuota de referencia = CIERRE (sin apertura disponible).",
        hyperparameters={"league_key": LEAGUE_KEY, "n_prob_bins": N_PROB_BINS, "odds_bin_edges": ODDS_BIN_LABELS},
        metrics={
            "ece_por_decil_probabilidad": ece_prob,
            "ece_por_rango_de_cuota": ece_odds,
        },
        predictions_path=out_path,
        notes="[MLS] Fase 8b -- diagnostico de calibracion, insumo necesario para "
              "tier1_probability_validation_mls.py antes de sumar MLS al pooling cross-liga de Tier 1.",
    )


if __name__ == "__main__":
    run()