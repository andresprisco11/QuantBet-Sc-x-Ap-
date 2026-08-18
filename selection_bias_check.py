"""
Fase 3.5c: verificacion directa del sesgo de seleccion ("winner's curse")
en el proceso de elegir la apuesta de mayor edge entre 3 resultados por
partido.

MOTIVACION: calibration_analysis.py encontro que la calibracion GENERAL
del blend v4 (los 3 resultados de TODOS los partidos, sin filtrar) es
razonable -- ECE de 1.3-1.4%, y en cuotas altas (3.00-5.00, 5.00+) el
modelo sobreestima la probabilidad real solo por ~1-1.5 puntos
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

Este script no recalcula nada -- solo re-analiza
'economic_backtest_v4_bets.csv' (las apuestas que economic_backtest.py ya
selecciono y simulo, con su columna 'odds_bin' ya calculada) para
comparar DIRECTAMENTE, bin por bin de cuota, el gap de calibracion de la
poblacion general (medido por calibration_analysis.py) contra el gap del
subconjunto efectivamente apostado.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

# Gaps de calibration_analysis.py sobre la POBLACION GENERAL (los 3
# resultados posibles de los 1,730 partidos con cuota de cierre, sin
# filtrar por si esa apuesta fue seleccionada) -- copiados aca solo para
# imprimir la comparacion lado a lado, no se recalculan.
REFERENCE_GENERAL_GAP = {
    "1.00-1.50": 0.0375,
    "1.50-2.00": 0.0287,
    "2.00-3.00": 0.0041,
    "3.00-5.00": -0.0097,
    "5.00+": -0.0155,
}


def run():
    path = PROCESSED_DATA_DIR / "EPL" / "economic_backtest_v4_bets.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.evaluation.economic_backtest' primero."
        )
    bets = pd.read_csv(path)

    table = bets.groupby("odds_bin", observed=True).agg(
        n_apuestas=("won", "count"),
        predicted_mean=("bet_fair_prob", "mean"),
        observed_freq=("won", "mean"),
    )
    table["gap_apuestas_seleccionadas"] = table["observed_freq"] - table["predicted_mean"]
    table["gap_poblacion_general"] = [REFERENCE_GENERAL_GAP.get(str(i), float("nan")) for i in table.index]
    table["diferencia_de_sesgo"] = table["gap_apuestas_seleccionadas"] - table["gap_poblacion_general"]

    print("=== Comparacion: calibracion de las apuestas SELECCIONADAS vs. poblacion general ===")
    print(table.round(4).to_string())
    print("\nLectura: 'gap_apuestas_seleccionadas' mas negativo que 'gap_poblacion_general' en un bin "
          "significa que, especificamente entre las apuestas que el proceso de seleccion eligio en ese "
          "rango de cuota, el modelo sobreestimo la probabilidad real MAS de lo que sobreestima en "
          "promedio sobre todos los resultados posibles de ese mismo rango -- evidencia directa de "
          "'winner's curse': el proceso de elegir el maximo edge entre 3 resultados por partido esta "
          "seleccionando sistematicamente sobreestimaciones de ruido, no señal real.")


if __name__ == "__main__":
    run()