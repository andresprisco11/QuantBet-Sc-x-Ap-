"""
Validacion directa de la meta de Tier 1: "probabilidad de exito del
modelo superior al 80-82%" (ver instrucciones del proyecto). Hasta ahora
NINGUN script del proyecto midio esto especificamente -- economic_backtest.py
evalua rentabilidad de apuestas con edge sobre el mercado en cualquier
rango de cuota (1.00-3.00 en la version actual), que es una pregunta
DISTINTA a "cuando el modelo dice que esta >80% seguro, acierta >80% de
las veces?".

Este script responde esa pregunta puntual, directo con datos ya
calculados -- no entrena ni corre nada nuevo. Reutiliza
'calibration_analysis_v4_long.csv' (generado por calibration_analysis.py):
para cada partido y cada uno de los 3 resultados posibles ya tiene la
probabilidad que el blend v4 le asigno y si ese resultado ocurrio de
verdad.

Para varios umbrales de confianza (70%, 75%, 80%, 82%, 85%, 90%), filtra
todos los casos donde el modelo dijo "estoy mas seguro que esto", y
reporta la frecuencia real de acierto -- la respuesta honesta a "el
85% de exito es alcanzable con este modelo, hoy?".

IMPORTANTE: esto NO filtra por edge sobre el mercado (a diferencia de
economic_backtest.py) -- mide la confianza del modelo en si misma, que es
literalmente lo que pide la definicion de Tier 1. Tambien reporta la
cuota promedio disponible en cada umbral, para saber si ese segmento
tiene volumen/cuota utilizable en la practica (un favorito al 90% de
probabilidad paga muy poco, y eso importa para el diseno de "bankroll
builder" de Tier 1).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]


def run():
    path = PROCESSED_DATA_DIR / "EPL" / "calibration_analysis_v4_long.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.evaluation.calibration_analysis' primero."
        )
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
    print("=== Cuando el modelo dice 'estoy mas seguro que X%', que tan seguido acierta de verdad? ===")
    print(table.round(4).to_string())
    print("\nLectura: 'acierto_real' es la frecuencia REAL de acierto en cada umbral -- compara eso "
          "directo contra la meta de Tier 1 (80-82%). 'gap' negativo significa que el modelo esta "
          "sobre-confiado incluso en su segmento mas seguro (acierta MENOS de lo que dice). "
          "'cuota_promedio' importa para saber si ese segmento paga lo suficiente para ser un "
          "'bankroll builder' util, no solo si acierta.")

    # Desglose adicional: el umbral mas cercano a la meta real de Tier 1 (80%), por temporada,
    # para ver si es consistente o si depende de un año particular.
    tier1_subset = long_df[long_df["predicted_prob"] >= 0.80]
    if not tier1_subset.empty:
        print("\n=== Umbral >=80% (meta de Tier 1), desglosado por temporada ===")
        by_season = tier1_subset.groupby("season").agg(
            n_casos=("actual", "count"),
            acierto_real=("actual", "mean"),
            prob_promedio_dicha=("predicted_prob", "mean"),
        )
        print(by_season.round(4).to_string())
    else:
        print("\n[AVISO] Cero casos con predicted_prob >= 80% en todo el dataset -- el modelo nunca "
              "llega a ese nivel de confianza en ningun resultado. Esto en si mismo es informacion "
              "clave para Tier 1.")


if __name__ == "__main__":
    run()