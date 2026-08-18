"""
Fase 8b: mismo chequeo que tier1_probability_validation.py (cuando el
modelo dice "estoy >=80% seguro", acierta >=80% de las veces de verdad?)
pero para MLS -- necesario antes de decidir si sus picks de alta confianza
entran al pool cross-liga de Tier 1 (ver roadmap, "Proximos pasos").

Fuente: 'calibration_analysis_mls_long.csv' (calibration_analysis_mls.py).
No entrena ni corre nada nuevo.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

LEAGUE_KEY = "MLS"
THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]


def run() -> pd.DataFrame:
    print(f"\n=== {LEAGUE_KEY} ===")
    path = PROCESSED_DATA_DIR / LEAGUE_KEY / "calibration_analysis_mls_long.csv"
    if not path.exists():
        print(f"[SKIP] No existe {path}. Corre 'python calibration_analysis_mls.py' primero.")
        return pd.DataFrame()

    long_df = pd.read_csv(path)
    print(f"Filas disponibles (partido x resultado posible): {len(long_df)}\n")

    rows = []
    for threshold in THRESHOLDS:
        subset = long_df[long_df["predicted_prob"] >= threshold]
        n = len(subset)
        if n == 0:
            rows.append({
                "umbral_confianza": f">={threshold:.0%}", "n_casos": 0,
                "prob_promedio_dicha": float("nan"), "acierto_real": float("nan"),
                "gap": float("nan"), "cuota_promedio": float("nan"),
            })
            continue
        actual = subset["actual"].mean()
        predicted_mean = subset["predicted_prob"].mean()
        avg_odds = subset["odds"].mean()
        rows.append({
            "umbral_confianza": f">={threshold:.0%}",
            "n_casos": n,
            "prob_promedio_dicha": predicted_mean,
            "acierto_real": actual,
            "gap": actual - predicted_mean,
            "cuota_promedio": avg_odds,
        })

    table = pd.DataFrame(rows).set_index("umbral_confianza")
    print(f"=== [MLS] Cuando el modelo dice 'estoy mas seguro que X%', que tan seguido acierta de verdad? ===")
    print(table.round(4).to_string())

    tier1_subset = long_df[long_df["predicted_prob"] >= 0.80]
    if not tier1_subset.empty:
        print(f"\n=== [MLS] Umbral >=80% (meta de Tier 1), desglosado por temporada ===")
        by_season = tier1_subset.groupby("season").agg(
            n_casos=("actual", "count"),
            acierto_real=("actual", "mean"),
            prob_promedio_dicha=("predicted_prob", "mean"),
        )
        print(by_season.round(4).to_string())
    else:
        print(f"\n[AVISO] MLS: cero casos con predicted_prob >= 80% en todo el dataset.")

    table = table.reset_index()
    table["league_key"] = LEAGUE_KEY
    return table


if __name__ == "__main__":
    run()