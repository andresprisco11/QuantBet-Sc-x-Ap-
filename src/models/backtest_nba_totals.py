"""
Fase 10 (NBA) -- backtest walk-forward del primer modelo de TOTALES de NBA
(`nba_totals_model.py`). Mismo patron de walk-forward expansivo (entrena
con temporadas anteriores, evalua la siguiente, sin fuga) que el resto de
los backtests del proyecto.

**SIN comparacion de mercado todavia -- limitacion real, no oculta**: no
hay lineas de totales descargadas (`theoddsapi_historical_loader.py` solo
trajo el mercado `h2h`). Este backtest compara el modelo contra un
baseline INGENUO pero honesto: el promedio de `game_total` de las
temporadas de ENTRENAMIENTO (expandido, nunca incluye la temporada de
test) -- equivalente a "apostar siempre la media historica conocida hasta
ese momento", sin ningun modelo. Si el modelo no le gana a este baseline
ingenuo, no vale la pena publicarlo como señal.

Metrica: RMSE (raiz del error cuadratico medio) sobre `game_total`, no
Brier -- esto es una variable continua (puntos), no una probabilidad
binaria como el moneyline.

Uso: python -m src.models.backtest_nba_totals
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nba_totals_model import fit_totals_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"


def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    return float(np.sqrt(((pred - actual) ** 2).mean()))


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH}.")
        return

    df = pd.read_csv(DATA_PATH)
    if "home_off_ewm" not in df.columns:
        print("[SKIP] Falta home_off_ewm/etc -- corre 'python -m src.processing.add_nba_gssd_features' "
              "(version con EWM) primero.")
        return

    df["game_total"] = df["home_pts"] + df["away_pts"]

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

        model, sigma = fit_totals_model(train_df)
        preds = predict_dataframe(model, sigma, test_df)

        # Baseline ingenuo -- promedio de game_total SOLO de las temporadas
        # de entrenamiento (expandido, sin fuga de la temporada de test).
        baseline_mu = float(train_df["game_total"].mean())

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        fold_df["baseline_mu_total"] = baseline_mu
        all_oos_records.append(fold_df)

    oos_df = pd.concat(all_oos_records, ignore_index=True)

    n_no_pred = int(oos_df["mu_total"].isna().sum())
    if n_no_pred:
        print(f"\n[INFO] {n_no_pred} partidos OOS sin prediccion (features GSSD no calculables) -- "
              f"excluidos de las metricas.")
    oos_df = oos_df[oos_df["mu_total"].notna()]
    n_total = len(oos_df)

    rmse_model = _rmse(oos_df["mu_total"], oos_df["game_total"])
    rmse_baseline = _rmse(oos_df["baseline_mu_total"], oos_df["game_total"])

    print(f"\n=== Resultados FUERA DE MUESTRA -- NBA TOTALES (modelo v1: GSSD + elo_sum + home_is_denver) ===")
    print(f"Partidos evaluados OOS: {n_total}")
    print(f"RMSE modelo propio:                              {rmse_model:.3f} puntos")
    print(f"RMSE baseline ingenuo (promedio historico expandido, sin modelo): {rmse_baseline:.3f} puntos")
    mejora_pct = (1 - rmse_model / rmse_baseline) * 100
    print(f"Mejora del modelo sobre el baseline ingenuo: {mejora_pct:.1f}% "
          f"({'el modelo le gana al baseline' if rmse_model < rmse_baseline else 'el modelo NO le gana al baseline'})")

    print(f"\n[AVISO] SIN comparacion contra mercado real de totales -- theoddsapi_historical_loader.py "
          f"solo descargo el mercado h2h hasta ahora. Este resultado es SOLO una validacion interna de "
          f"que el modelo captura señal real (le gana a adivinar con el promedio historico), NO una "
          f"prueba de que le gane a un libro real. Agregar el mercado de totales requiere volver a "
          f"correr la descarga historica con markets=\"h2h,totals\" (duplica aprox. el costo en creditos "
          f"de esa descarga) -- decision pendiente del usuario antes de gastar mas creditos.")

    metrics = {
        "n_total": n_total,
        "rmse_model": rmse_model,
        "rmse_baseline_naive": rmse_baseline,
        "mejora_pct_vs_baseline": mejora_pct,
    }

    out_path = PROCESSED_DATA_DIR / "NBA" / "totals_predictions_oos_walkforward_v1.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nba_totals.py",
        model_name="nba_totals_normal",
        model_version="v1",
        data_paths=[DATA_PATH],
        features="game_total ~ home_off_ewm + home_def_ewm + away_off_ewm + away_def_ewm + elo_sum "
                 "(home_elo+away_elo) + home_is_denver. Screeneado con OLS full-sample: b2b_sum y "
                 "3in4_sum descartados por no significativos para totales (aunque si importan para "
                 "margen) -- ver docstring de nba_totals_model.py.",
        hyperparameters={"gssd_ewm_span": 15, "gssd_min_periods": 3},
        metrics=metrics,
        predictions_path=out_path,
        notes="Primer modelo de TOTALES de NBA -- Normal(mu,sigma) sobre game_total, misma arquitectura "
              "que el modelo de margen pero SIN comparacion de mercado todavia (no hay lineas de totales "
              "descargadas). Evaluado contra un baseline ingenuo (promedio historico expandido), no "
              "contra un mercado real -- limitacion documentada explicita, no oculta.",
    )


if __name__ == "__main__":
    run()
