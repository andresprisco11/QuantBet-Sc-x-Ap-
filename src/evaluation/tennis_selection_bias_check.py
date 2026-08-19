"""
Escalamiento a Tenis, paso 7 (tras tune_staking_rules_tennis.py, corrido y
CONFIRMADO 2026-08-19 -- ver roadmap). El barrido SÍ encontró volumen
suficiente en ambos tours (edge>=1% fue el único umbral con volumen
reportable: ATP n=153, WTA hasta n=1728 según el resto de la regla) pero
el ROI fue NEGATIVO en las 160 combinaciones probadas en CADA tour -- no
es un problema de umbral, ni siquiera el más permisivo de la grilla dio
positivo. Esto descarta "hay que bajar más el umbral" como próxima
hipótesis (bajarlo más acerca la selección a "apostar a casi todo", que
pierde por el margen del libro por definición).

HIPÓTESIS A PROBAR ACÁ -- la misma lógica que ya diagnosticó la causa
real del problema en Serie A/Bundesliga en fútbol (selection_bias_check.py,
Fase 8): si el ROI es negativo pese a que el modelo "ve" edge positivo, la
causa más probable es WINNER'S CURSE DE SELECCIÓN -- el edge detectado es
en gran parte ruido de estimación (el modelo de habilidad de tenis apenas
mejora sobre el mercado: Brier 0.218-0.219 vs. 0.202, ver
tennis_logistic_model.py), y seleccionar sistemáticamente los partidos con
MAYOR edge aparente selecciona sistemáticamente sobreestimaciones, no
habilidad real detectada. En tenis no hay CLV para confirmar esta firma
como en fútbol (CLV positivo + ROI negativo) -- se prueba directo por
calibración: población completa vs. seleccionado, por rango de cuota.

METODOLOGÍA: para cada partido se calcula el lado (Player1 o Player2) con
mejor edge según blend_prob:
  - "Población completa": SIN umbral de edge -- el lado que el modelo
    favorece en absolutamente todos los partidos con cuota disponible.
  - "Seleccionado": el mismo cálculo, pero filtrado por la única regla que
    tuvo volumen en el barrido (edge>=1%, odds<=6.0).
Se comparan, bucketizados por rango de cuota, la probabilidad promedio
predicha (blend_prob) vs. la tasa de acierto real, en población vs.
seleccionado. Si el seleccionado sobreestima MÁS que la población general
en el mismo rango de cuota (gap más negativo), es la firma de winner's
curse -- misma lectura que selection_bias_check.py en fútbol.

Reutiliza _select_bets()/load_predictions() de economic_backtest_tennis.py
con umbrales extremos (min_edge=-1.0, max_odds=999.0) para generar la
"población completa" sin reimplementar la lógica de elección de lado --
mismo criterio de reutilizar código ya confirmado que el resto del
proyecto.

Salida: data/runs/tennis_selection_bias_check_<tour>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.evaluation.economic_backtest_tennis import _select_bets, load_predictions

TOURS = ["ATP", "WTA"]
SELECTED_MIN_EDGE = 0.01   # unico umbral con volumen reportable en tune_staking_rules_tennis.py
SELECTED_MAX_ODDS = 6.0    # el mas alto de la grilla ya probada

ODDS_BINS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, np.inf]
ODDS_LABELS = ["1.00-1.50", "1.50-2.00", "2.00-2.50", "2.50-3.00", "3.00-4.00", "4.00-6.00", "6.00+"]


def _calibration_by_odds_bucket(bets: pd.DataFrame) -> pd.DataFrame:
    bets = bets.copy()
    bets["odds_bucket"] = pd.cut(bets["odds"], bins=ODDS_BINS, labels=ODDS_LABELS, right=False)
    grouped = bets.groupby("odds_bucket", observed=True).agg(
        n=("won", "size"),
        predicted=("blend_prob", "mean"),
        actual=("won", "mean"),
    ).reset_index()
    grouped["gap"] = grouped["actual"] - grouped["predicted"]
    return grouped


def run(tour: str) -> None:
    print(f"\n=== {tour.upper()} ===")
    try:
        df = load_predictions(tour)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    population = _select_bets(df, min_edge_threshold=-1.0, max_odds=999.0)
    selected = _select_bets(df, min_edge_threshold=SELECTED_MIN_EDGE, max_odds=SELECTED_MAX_ODDS)

    print(f"Población completa (mejor lado de cada partido, sin filtro de edge): {len(population)} partidos")
    print(f"Seleccionado (edge>={SELECTED_MIN_EDGE:.0%}, odds<={SELECTED_MAX_ODDS}): {len(selected)} apuestas")

    pop_calib = _calibration_by_odds_bucket(population)
    sel_calib = _calibration_by_odds_bucket(selected)

    merged = pop_calib.merge(sel_calib, on="odds_bucket", how="outer", suffixes=("_poblacion", "_seleccion"))
    merged["sesgo_seleccion"] = merged["gap_seleccion"] - merged["gap_poblacion"]

    print("\nCalibración por rango de cuota -- población general vs. seleccionado, y el sesgo de selección "
          "(gap_seleccion - gap_poblacion; NEGATIVO y consistente en varios rangos = firma de winner's curse, "
          "misma lectura que selection_bias_check.py en fútbol):")
    pct_cols = ["predicted_poblacion", "actual_poblacion", "gap_poblacion",
                "predicted_seleccion", "actual_seleccion", "gap_seleccion", "sesgo_seleccion"]
    with pd.option_context("display.width", 160):
        print(merged.to_string(index=False, formatters={c: "{:.2%}".format for c in pct_cols}))

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / f"tennis_selection_bias_check_{tour.lower()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    for tour in TOURS:
        run(tour)