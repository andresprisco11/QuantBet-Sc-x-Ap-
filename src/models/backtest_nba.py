"""
Fase 10 (NBA) -- primer backtest walk-forward de NBA, mismo patron honesto
que backtest_nfl.py: entrena con temporadas anteriores, evalua la
siguiente, ventana expansiva, sin fuga. Usa el modelo de margen de puntos
(`nba_margin_model.py`, Normal(mu, sigma) sobre diferencia de Elo).

**Diferencia real frente a NFL, documentada explicitamente, no oculta**:
NFL puede comparar el modelo contra `market_prob_home` porque
`clean_nfl_data.py` ya trae moneyline de mercado en el mismo CSV. NBA
TODAVIA NO -- `games_clean.csv` solo tiene resultados + Elo, ningun dato de
mercado (`theoddsapi_historical_loader.py` esta escrito pero no corrido
todavia, cero creditos gastados en esto). Por eso esta primera corrida de
NBA SOLO reporta el Brier score OOS del modelo propio -- NO hay blend
Benter Boost ni comparacion contra mercado en esta version. Cuando se pegue
el mercado (script de merge futuro, mismo patron que
merge_thestatsapi_xg.py de futbol), este script se extiende -- no se
inventa un numero de mercado para completar la tabla.

**NBA no tiene corte de confiabilidad de temporadas** (a diferencia de NFL,
que excluye season<2010 de moneyline) -- `games_clean.csv` no tiene ninguna
cuota todavia, asi que no aplica esa distincion aca. Cuando el merge de
cuotas exista, ahi sí puede aparecer un corte real segun lo que confirme
`theoddsapi_historical_loader.py` (recordar: cuotas historicas de The Odds
API solo desde 2020-06-06, documentado en ese script).

Uso: python -m src.models.backtest_nba
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nba_margin_model import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"


def _brier_binary(model_prob_home: pd.Series, ftr: pd.Series) -> float:
    """Brier score de 2 vias -- NBA no tiene empate (confirmado, 0 casos
    reales en clean_nba_data.py), asi que a diferencia de NFL no hace falta
    mapear un caso 'T'=0.5."""
    actual = ftr.map({"H": 1.0, "A": 0.0})
    return float(((model_prob_home - actual) ** 2).mean())


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre clean_nba_data.py y add_nba_elo_features.py primero.")
        return

    df = pd.read_csv(DATA_PATH)
    if "home_elo" not in df.columns:
        print("[SKIP] Falta home_elo/away_elo -- corre 'python -m src.processing.add_nba_elo_features' primero.")
        return

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

        model, sigma = fit_margin_model(train_df)
        preds = predict_dataframe(model, sigma, test_df)

        fold_df = pd.concat([test_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        fold_df["fold_test_season"] = test_season
        all_oos_records.append(fold_df)

    oos_df = pd.concat(all_oos_records, ignore_index=True)
    n_total = len(oos_df)

    model_brier_all = _brier_binary(oos_df["model_prob_home"], oos_df["FTR"])
    print(f"\n=== Resultados FUERA DE MUESTRA -- NBA, moneyline (margen de puntos -> Normal -> P(gana local)) ===")
    print(f"Partidos evaluados OOS ({ordered_seasons[1]}-{ordered_seasons[-1]}): {n_total}")
    print(f"Brier score modelo propio: {model_brier_all:.6f}")
    print(f"\n[AVISO] Sin comparacion contra mercado en esta corrida -- games_clean.csv todavia no tiene "
          f"cuotas (theoddsapi_historical_loader.py escrito pero sin correr). Este numero es SOLO el "
          f"modelo propio contra el resultado real, no contra ningun benchmark de mercado todavia.")

    # Chequeo de sanidad de referencia (NO es un target -- un clasificador que solo
    # prediga "gana el local siempre" ya acierta ~58-59% segun el chequeo de sanidad
    # de add_nba_elo_features.py; el Brier de ESE clasificador trivial serviria de
    # piso de comparacion honesto una vez calculado, no se calcula aca para no
    # inflar esta corrida con numeros que no son el foco).
    home_win_rate_oos = (oos_df["FTR"] == "H").mean()
    print(f"\n% de victorias reales del local en el conjunto OOS (referencia, no un target): {home_win_rate_oos:.2%}")

    out_path = PROCESSED_DATA_DIR / "NBA" / "model_predictions_oos_walkforward_v1.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nba.py",
        model_name="nba_margin_normal",
        model_version="v1",
        data_paths=[DATA_PATH],
        features="point_margin ~ elo_diff (home_elo - away_elo), sigma = desvio de residuos de training, "
                 "Elo con ajuste MOV NBA + regresion a la media entre temporadas (ver add_nba_elo_features.py)",
        hyperparameters={
            "elo_k_factor": 20.0,
            "elo_home_advantage": 100.0,
            "elo_season_regression": 0.25,
        },
        metrics={
            "n_total": n_total,
            "model_brier_all_seasons": model_brier_all,
            "home_win_rate_oos": home_win_rate_oos,
        },
        predictions_path=out_path,
        notes="Primer modelo real de NBA -- moneyline derivado de una distribucion de margen de puntos "
              "(Normal(mu,sigma) sobre diferencia de Elo con MOV NBA), mismo diseño que nfl_margin_model.py. "
              "SIN comparacion contra mercado todavia (games_clean.csv no tiene cuotas) -- ese es el "
              "siguiente paso real, no esta corrida.",
    )


if __name__ == "__main__":
    run()
