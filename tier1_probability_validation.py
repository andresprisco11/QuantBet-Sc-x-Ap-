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

Fix 2026-08-18 (Fase 8, multi-liga): run() estaba hardcodeado a EPL. Se
parametriza por league_key y se loopea sobre LEAGUES en __main__ -- la
pregunta que motiva este script (cuanto volumen real hay en el umbral de
Tier 1) es precisamente la que la expansion de ligas busca resolver, asi
que verlo separado por liga (y comparado) es el punto central de este fix.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]


def run(league_key: str) -> pd.DataFrame:
    print(f"\n=== {league_key} ===")
    path = PROCESSED_DATA_DIR / league_key / "calibration_analysis_v4_long.csv"
    if not path.exists():
        print(f"[SKIP] No existe {path}. Corre 'python -m src.evaluation.calibration_analysis' primero.")
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
    print(f"=== [{league_key}] Cuando el modelo dice 'estoy mas seguro que X%', que tan seguido acierta de verdad? ===")
    print(table.round(4).to_string())

    # Desglose adicional: el umbral mas cercano a la meta real de Tier 1 (80%), por temporada,
    # para ver si es consistente o si depende de un año particular.
    tier1_subset = long_df[long_df["predicted_prob"] >= 0.80]
    if not tier1_subset.empty:
        print(f"\n=== [{league_key}] Umbral >=80% (meta de Tier 1), desglosado por temporada ===")
        by_season = tier1_subset.groupby("season").agg(
            n_casos=("actual", "count"),
            acierto_real=("actual", "mean"),
            prob_promedio_dicha=("predicted_prob", "mean"),
        )
        print(by_season.round(4).to_string())
    else:
        print(f"\n[AVISO] {league_key}: cero casos con predicted_prob >= 80% en todo el dataset -- el modelo "
              f"nunca llega a ese nivel de confianza en ningun resultado en esta liga. Esto en si mismo es "
              f"informacion clave para Tier 1.")

    table = table.reset_index()
    table["league_key"] = league_key
    return table


if __name__ == "__main__":
    all_tables = []
    for league_key in LEAGUES:
        t = run(league_key)
        if not t.empty:
            all_tables.append(t)

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        n_80_total = combined.loc[combined["umbral_confianza"] == ">=80%", "n_casos"].sum()
        print(f"\n=== TOTAL combinado, las {len(all_tables)} ligas, umbral >=80% ===")
        print(f"n_casos sumados entre todas las ligas: {n_80_total} "
              f"(referencia previa, solo EPL: 59 -- ver roadmap Fase 3.5)")
        print(combined[combined["umbral_confianza"] == ">=80%"]
              [["league_key", "n_casos", "prob_promedio_dicha", "acierto_real", "gap"]]
              .round(4).to_string(index=False))