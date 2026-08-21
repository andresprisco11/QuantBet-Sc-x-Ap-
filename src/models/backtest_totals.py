"""
Fase 10 (futbol) -- primer modelo de TOTALES (over/under de goles) para las
4 ligas europeas. Con la via de mejorar el MONEYLINE de futbol agotada (v5,
v6, v7 y ahora v8-xG, todos negativos o indistinguibles de ruido -- ver
roadmap), este es terreno nuevo, no una repeticion de la via ya cerrada: el
mercado de totales es un producto DISTINTO del 1X2, con su propia estructura
de +EV, y el proyecto nunca lo probo en futbol (si en NBA, ver
nba_totals_model.py -- ahi la variable es continua, puntaje total; aca es
un conteo Poisson bajo, goles totales, arquitectura distinta a proposito).

CLAVE METODOLOGICA (por que esto NO es un modelo nuevo que entrenar desde
cero): el modelo v4 YA es un Poisson doble -- predice lambda_home y
lambda_away independientes para cada partido (ver poisson_model_v4.py). Si
esos dos Poisson son independientes (el Dixon-Coles de v3, que ajusta
correlacion en el marcador, fue resultado NEGATIVO y no se adopto -- ver
roadmap Fase 2/3), la suma de dos Poisson independientes es OTRO Poisson,
con lambda_total = lambda_home + lambda_away. Esto significa que la
probabilidad de "mas de X goles" para CUALQUIER linea de totales sale
directo de lo que v4 ya calcula, sin entrenar nada nuevo -- se reusa el
modelo ya adoptado como referencia de produccion (v4, NO v8 -- xG fue
rechazado esta misma sesion).

SIN comparacion de mercado real todavia -- misma limitacion honesta que
nba_totals_model.py: football-data.co.uk no trae lineas de totales en las
columnas ya descargadas, y no se agrega esa descarga aca (fuera de alcance
de esta corrida). Se compara contra un baseline INGENUO pero honesto: la
frecuencia empirica de "mas de X goles" en las temporadas de ENTRENAMIENTO
(expandido, sin fuga de la temporada de test) -- equivalente a "apostar
siempre la tasa historica conocida hasta ese momento", sin ningun modelo.

Lineas evaluadas: 1.5, 2.5, 3.5 (las 3 lineas estandar del mercado de
totales de futbol).

Uso: python -m src.models.backtest_totals
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR, SEASONS
from src.models.poisson_model_v4 import build_long_format_v4, fit_poisson_model_v4, predict_dataframe_v4
from src.models.poisson_model_v2 import DEFAULT_HALF_LIFE_DAYS
from src.tracking.run_logger import log_run

TOTAL_LINES = [1.5, 2.5, 3.5]


def _brier_binary(model_prob: pd.Series, actual: pd.Series) -> float:
    return float(((model_prob - actual) ** 2).mean())


def run(league_key: str) -> None:
    data_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not data_path.exists():
        print(f"[SKIP] {league_key}: no existe {data_path}.")
        return

    df = pd.read_csv(data_path).copy()  # evita el PerformanceWarning de fragmentacion al agregar columnas
    if "home_recent_st_diff" not in df.columns:
        print(f"[SKIP] {league_key}: falta home_recent_st_diff -- corre "
              f"'python -m src.processing.add_team_form_features' primero.")
        return

    df["season"] = df["season"].astype(str)
    df["goal_total"] = df["FTHG"] + df["FTAG"]
    ordered_seasons = SEASONS
    all_oos_records = []

    print(f"\n=== {league_key} ===")
    for i in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:i]
        test_season = ordered_seasons[i]
        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]
        if test_df.empty:
            print(f"[SKIP] Temporada {test_season}: sin partidos.")
            continue

        known_teams = set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"]))
        print(f"Entrenando con {train_seasons} ({len(train_df)} partidos) -> evaluando {test_season} "
              f"({len(test_df)} partidos)...")

        # Reusa el modelo v4 YA ADOPTADO -- no se entrena nada nuevo, ver docstring.
        long_df = build_long_format_v4(train_df)
        model = fit_poisson_model_v4(long_df)
        preds = predict_dataframe_v4(model, test_df, known_teams)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1).copy()
        fold_df["lambda_total"] = fold_df["lambda_home"] + fold_df["lambda_away"]
        fold_df["fold_test_season"] = test_season

        # Baseline ingenuo por linea -- frecuencia empirica de "mas de X goles"
        # SOLO en las temporadas de entrenamiento (expandido, sin fuga).
        for line in TOTAL_LINES:
            fold_df[f"baseline_prob_over_{line}"] = float((train_df["goal_total"] > line).mean())
            fold_df[f"model_prob_over_{line}"] = 1.0 - poisson_dist.cdf(int(np.floor(line)), fold_df["lambda_total"])
            fold_df[f"actual_over_{line}"] = (fold_df["goal_total"] > line).astype(int)

        all_oos_records.append(fold_df)

    if not all_oos_records:
        print(f"[SKIP] {league_key}: no hubo ningun fold OOS evaluable.")
        return

    oos_df = pd.concat(all_oos_records, ignore_index=True)
    n_total = len(oos_df)

    print(f"\n=== Resultados FUERA DE MUESTRA -- TOTALES futbol [{league_key}] "
          f"(v4 reusado, Poisson total = lambda_home + lambda_away) ===")
    print(f"Partidos evaluados OOS: {n_total}")

    metrics = {"n_total": n_total}
    for line in TOTAL_LINES:
        brier_model = _brier_binary(oos_df[f"model_prob_over_{line}"], oos_df[f"actual_over_{line}"])
        brier_baseline = _brier_binary(oos_df[f"baseline_prob_over_{line}"], oos_df[f"actual_over_{line}"])
        tasa_real = oos_df[f"actual_over_{line}"].mean()
        mejora_pct = (1 - brier_model / brier_baseline) * 100
        print(f"\n-- Linea {line} goles (tasa real de 'over' en el OOS: {tasa_real:.1%}) --")
        print(f"   Brier score modelo (Poisson v4):            {brier_model:.6f}")
        print(f"   Brier score baseline ingenuo (tasa hist.):  {brier_baseline:.6f}")
        print(f"   Mejora del modelo sobre el baseline: {mejora_pct:+.1f}% "
              f"({'el modelo le gana al baseline' if brier_model < brier_baseline else 'el modelo NO le gana al baseline'})")
        metrics[f"brier_model_over_{line}"] = brier_model
        metrics[f"brier_baseline_over_{line}"] = brier_baseline
        metrics[f"mejora_pct_over_{line}"] = mejora_pct

    print(f"\n[AVISO] SIN comparacion contra mercado real de totales -- football-data.co.uk no trae esas "
          f"lineas en las columnas ya descargadas. Este resultado es SOLO una validacion interna de que "
          f"el modelo captura señal real (le gana a adivinar con la tasa historica), NO una prueba de que "
          f"le gane a un libro real -- misma limitacion ya documentada en nba_totals_model.py.")

    out_path = PROCESSED_DATA_DIR / league_key / "totals_predictions_oos_walkforward_v1.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_totals.py",
        model_name="football_totals_poisson_reuse",
        model_version="v1",
        data_paths=[data_path],
        features=f"[{league_key}] lambda_total = lambda_home + lambda_away (modelo v4 YA adoptado, "
                  "reusado sin reentrenar arquitectura nueva), P(over X) = 1 - PoissonCDF(floor(X), lambda_total). "
                  "Asume independencia home/away (Dixon-Coles descartado en v3, resultado negativo).",
        hyperparameters={
            "league_key": league_key,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "total_lines": TOTAL_LINES,
        },
        metrics=metrics,
        predictions_path=out_path,
        notes=f"[{league_key}] Primer modelo de TOTALES de futbol -- reusa el Poisson doble de v4 (ya "
              "adoptado como referencia de produccion) para derivar P(over/under) via la suma de dos "
              "Poisson independientes, sin entrenar nada nuevo. Terreno nuevo (mercado distinto del 1X2), "
              "no repite la via de moneyline ya agotada (v5/v6/v7/v8 negativos). SIN comparacion de mercado "
              "real todavia (limitacion documentada, no oculta) -- evaluado contra baseline ingenuo "
              "(tasa historica expandida, sin fuga).",
    )


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)
