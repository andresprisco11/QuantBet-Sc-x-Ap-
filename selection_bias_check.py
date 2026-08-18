"""
Fase 3.5c: verificacion directa del sesgo de seleccion ("winner's curse")
en el proceso de elegir la apuesta de mayor edge entre 3 resultados por
partido.

MOTIVACION: calibration_analysis.py encontro que la calibracion GENERAL
del blend v4 (los 3 resultados de TODOS los partidos, sin filtrar) es
razonable -- ECE de 1.3-1.4% en EPL -- y en cuotas altas (3.00-5.00,
5.00+) el modelo sobreestima la probabilidad real solo por ~1-1.5 puntos
porcentuales. Eso NO alcanza, por si solo, para explicar un ROI de
-18% a -20% en esos mismos rangos de cuota en el backtest economico.

HIPOTESIS: el proceso de seleccion de apuestas (economic_backtest.py
elige, por partido, el UNICO resultado con mayor edge entre los 3
posibles) es en si mismo una fuente de sesgo -- aunque el modelo este
bien calibrado EN PROMEDIO sobre los 3 resultados, elegir
sistematicamente el maximo de 3 estimaciones ruidosas por partido tiende
a seleccionar los casos donde el ruido empujo la probabilidad estimada
hacia ARRIBA, no necesariamente los casos con señal real -- el mismo
fenomeno que la "maldicion del ganador" (winner's curse) en subastas, o
"regression to the mean" aplicado a elegir el maximo de una muestra
ruidosa. Si esto es lo que esta pasando, el gap de calibracion (predicho
vs observado) deberia ser mucho mas grande especificamente en el
subconjunto de apuestas SELECCIONADAS que en la poblacion general de
resultados posibles.

Este script no recalcula probabilidades ni entrena nada -- solo re-analiza
'economic_backtest_v4_bets.csv' (las apuestas que economic_backtest.py ya
selecciono y simulo, con su columna 'odds_bin' ya calculada) para
comparar DIRECTAMENTE, bin por bin de cuota, el gap de calibracion de la
poblacion general contra el gap del subconjunto efectivamente apostado.

Fix 2026-08-18 (Fase 8, multi-liga): la version anterior tenia
REFERENCE_GENERAL_GAP como un diccionario HARDCODEADO, copiado a mano del
resultado impreso por calibration_analysis.py sobre EPL. Eso era valido
mientras solo existia una liga, pero aplicar esos mismos numeros como
referencia "poblacion general" para Serie A/Bundesliga/La Liga habria
sido lisa y llanamente incorrecto -- cada liga tiene su propia
calibracion de mercado, ya lo confirmo el resultado de backtest_v4.py
multi-liga (Brier de mercado distinto por liga). Se reemplaza el
diccionario fijo por una lectura DINAMICA de
'calibration_analysis_v4_long.csv' de la liga correspondiente (que ya
tiene la columna 'odds_bin' precalculada, mismos cortes que
economic_backtest.py) -- la referencia ahora siempre es la poblacion
general de ESA MISMA liga, nunca un numero prestado de otra.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR


def _general_population_gap(league_key: str) -> pd.DataFrame:
    """Recalcula, para ESTA liga, el gap de calibracion por rango de cuota sobre la
    poblacion general de resultados posibles (los 3 por partido, sin filtrar por si
    fueron seleccionados) -- misma tabla que calibration_analysis.py imprime, leida
    de su output ya guardado en vez de hardcodear numeros de otra liga."""
    path = PROCESSED_DATA_DIR / league_key / "calibration_analysis_v4_long.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.evaluation.calibration_analysis' primero -- "
            f"la referencia de poblacion general se lee de ahi, no se hardcodea."
        )
    long_df = pd.read_csv(path)
    table = long_df.groupby("odds_bin", observed=True).agg(
        predicted_mean=("predicted_prob", "mean"),
        observed_freq=("actual", "mean"),
    )
    table["gap_poblacion_general"] = table["observed_freq"] - table["predicted_mean"]
    return table[["gap_poblacion_general"]]


def run(league_key: str) -> None:
    print(f"\n=== {league_key} ===")
    bets_path = PROCESSED_DATA_DIR / league_key / "economic_backtest_v4_bets.csv"
    if not bets_path.exists():
        print(f"[SKIP] No existe {bets_path}. Corre 'python -m src.evaluation.economic_backtest' primero.")
        return

    try:
        reference = _general_population_gap(league_key)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    bets = pd.read_csv(bets_path)

    table = bets.groupby("odds_bin", observed=True).agg(
        n_apuestas=("won", "count"),
        predicted_mean=("bet_fair_prob", "mean"),
        observed_freq=("won", "mean"),
    )
    table["gap_apuestas_seleccionadas"] = table["observed_freq"] - table["predicted_mean"]
    table = table.join(reference, how="left")
    table["diferencia_de_sesgo"] = table["gap_apuestas_seleccionadas"] - table["gap_poblacion_general"]

    print(f"=== [{league_key}] Comparacion: calibracion de las apuestas SELECCIONADAS vs. poblacion general "
          f"(referencia recalculada de esta misma liga, no hardcodeada) ===")
    print(table.round(4).to_string())
    print("\nLectura: 'gap_apuestas_seleccionadas' mas negativo que 'gap_poblacion_general' en un bin "
          "significa que, especificamente entre las apuestas que el proceso de seleccion eligio en ese "
          "rango de cuota, el modelo sobreestimo la probabilidad real MAS de lo que sobreestima en "
          "promedio sobre todos los resultados posibles de ese mismo rango -- evidencia directa de "
          "'winner's curse': el proceso de elegir el maximo edge entre 3 resultados por partido esta "
          "seleccionando sistematicamente sobreestimaciones de ruido, no señal real.")


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)