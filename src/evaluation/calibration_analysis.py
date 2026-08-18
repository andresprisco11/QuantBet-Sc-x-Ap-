"""
Fase 3.5b: diagnostico de calibracion sobre las probabilidades del blend v4.

MOTIVACION: el backtest economico (economic_backtest.py) encontro CLV
promedio positivo (+0.0062, 60.93% de las apuestas le ganaron a la linea
de cierre) pero ROI fuertemente negativo (-8.94%, drawdown 92.37%),
concentrado casi enteramente en apuestas de cuota alta (3.00+: 63% del
volumen, ROI -18% a -20%). Esa combinacion -- direccion correcta la
mayoria de las veces, resultado economico desastroso -- es la firma
clasica de un modelo con probabilidades MAL CALIBRADAS EN MAGNITUD,
especialmente en la cola (favoritos/perdedores extremos): el modelo
puede "saber" bien hacia que lado se inclina un partido sin saber bien
CUANTO se inclina, y Kelly castiga exactamente ese segundo error, no el
primero.

Este script no es una hipotesis mas -- es la medicion directa de esa
sospecha. Descompone las probabilidades del blend en bins (por
probabilidad predicha y por rango de cuota, este ultimo con los mismos
cortes que economic_backtest.py para comparar directo) y compara la
probabilidad promedio predicha contra la frecuencia real observada en
cada bin. Si el modelo estuviera perfectamente calibrado, ambas
columnas deberian ser practicamente iguales en todos los bins.

METODOLOGIA: "unroll" de cada partido a 3 filas (una por resultado
posible: home/draw/away), cada una con su probabilidad predicha por el
blend y un indicador binario de si ese resultado especifico ocurrio.
Es el approach estandar para reliability diagrams / curvas de
calibracion en pronosticos probabilisticos multiclase.

Fuente de datos: 'model_predictions_oos_walkforward_v4.csv', igual que
economic_backtest.py -- no entrena nada nuevo, solo re-analiza lo que
ya existe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.tracking.run_logger import log_run

N_PROB_BINS = 10
ODDS_BIN_EDGES = [1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
ODDS_BIN_LABELS = ["1.00-1.50", "1.50-2.00", "2.00-3.00", "3.00-5.00", "5.00+"]

SIDES = [
    ("home", "blend_prob_home", "PSH", "H"),
    ("draw", "blend_prob_draw", "PSD", "D"),
    ("away", "blend_prob_away", "PSA", "A"),
]


def _unroll(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (partido, resultado posible) -- 3 filas por partido evaluado."""
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
    """ECE: promedio ponderado por tamano de bin de |predicho - observado|."""
    total_n = table["n"].sum()
    return float((table["n"] * table["gap"].abs()).sum() / total_n)


def run():
    path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.models.backtest_v4' primero -- este script solo "
            f"re-analiza predicciones ya generadas, no entrena nada."
        )
    df = pd.read_csv(path)
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df_eval = df.loc[has_blend].copy()

    long_df = _unroll(df_eval)
    print(f"Filas desenrolladas (3 por partido, {len(df_eval)} partidos evaluados): {len(long_df)}")
    base_rates = long_df.groupby("side", observed=True)["actual"].mean().to_dict()
    print(f"Frecuencia base observada por resultado (referencia, no deberia sumar exactamente 1.0 "
          f"porque cada partido aporta a los 3 lados): {base_rates}")

    # --- calibracion por decil de probabilidad predicha (los 3 resultados juntos) ---
    long_df["prob_decile"] = pd.qcut(long_df["predicted_prob"], q=N_PROB_BINS, duplicates="drop")
    prob_table = _calibration_table(long_df, "prob_decile")
    ece_prob = _expected_calibration_error(prob_table)

    print("\n=== Calibracion por decil de probabilidad predicha (blend v4) ===")
    print(prob_table.round(4).to_string())
    print(f"\nExpected Calibration Error (ECE), ponderado por decil de probabilidad: {ece_prob:.4f}")

    # --- calibracion por rango de cuota (mismos cortes que economic_backtest.py -- comparacion directa) ---
    long_df["odds_bin"] = pd.cut(long_df["odds"], bins=ODDS_BIN_EDGES, labels=ODDS_BIN_LABELS)
    odds_table = _calibration_table(long_df, "odds_bin")
    ece_odds = _expected_calibration_error(odds_table)

    print("\n=== Calibracion por rango de cuota (compara directo con el desglose de economic_backtest.py) ===")
    print(odds_table.round(4).to_string())
    print(f"\nExpected Calibration Error (ECE), ponderado por rango de cuota: {ece_odds:.4f}")

    # --- calibracion por lado (home/draw/away) -- puede haber sesgo estructural por localia ---
    side_table = _calibration_table(long_df, "side")
    print("\n=== Calibracion por lado (home/draw/away) ===")
    print(side_table.round(4).to_string())

    out_path = PROCESSED_DATA_DIR / "EPL" / "calibration_analysis_v4_long.csv"
    long_df.to_csv(out_path, index=False)
    print(f"\nGuardado detalle desenrollado (partido x resultado posible) -> {out_path}")

    log_run(
        script="calibration_analysis.py",
        model_name="poisson",
        model_version="v4",
        data_paths=[path],
        features="Diagnostico de calibracion sobre blend_prob_* ya generado por backtest_v4.py -- no entrena "
                  "modelo nuevo, solo re-analiza.",
        hyperparameters={"n_prob_bins": N_PROB_BINS, "odds_bin_edges": ODDS_BIN_LABELS},
        metrics={
            "ece_por_decil_probabilidad": ece_prob,
            "ece_por_rango_de_cuota": ece_odds,
            "gap_odds_1.00_1.50": float(odds_table.loc["1.00-1.50", "gap"]) if "1.00-1.50" in odds_table.index else None,
            "gap_odds_3.00_5.00": float(odds_table.loc["3.00-5.00", "gap"]) if "3.00-5.00" in odds_table.index else None,
            "gap_odds_5.00_plus": float(odds_table.loc["5.00+", "gap"]) if "5.00+" in odds_table.index else None,
        },
        predictions_path=out_path,
        notes="Diagnostico de calibracion (reliability diagram) para explicar por que economic_backtest.py dio "
              "CLV promedio positivo (+0.0062, 60.93% de apuestas) pero ROI muy negativo (-8.94%, drawdown "
              "92.37%), concentrado en cuotas altas.",
    )


if __name__ == "__main__":
    run()