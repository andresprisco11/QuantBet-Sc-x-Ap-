"""
Fase 10 -- primer backtest walk-forward de NFL, mismo patron honesto que
backtest_v4.py/backtest_v7.py (futbol) y backtest_tennis.py (tenis): entrena
con temporadas anteriores, evalua la siguiente, ventana expansiva, sin
fuga. Usa el modelo de margen de puntos (`nfl_margin_model.py`, Normal(mu,
sigma) sobre diferencia de Elo) para derivar una probabilidad de moneyline
comparable directamente contra `market_prob_home` (no-vig, formula
americana, calculado en clean_nfl_data.py).

**Diferencia real frente a futbol/tenis, documentada explicitamente**: el
mercado de moneyline solo es confiable desde season>=2010
(`reliable_moneyline`, ver clean_nfl_data.py/nfl_data_loader.py) -- el
modelo SI se entrena con todo el historico disponible desde 1999 (el Elo y
la regresion de margen no dependen de tener cuota de mercado), pero la
comparacion de Brier/blend contra el mercado y el calculo de blend weight
SOLO se hacen sobre partidos con `reliable_moneyline=True` y
`market_prob_home` no nulo -- exactamente el mismo criterio de exclusion
explicita (nunca silenciosa) que ya se uso en backtest_v4.py con los
partidos sin cuota de cierre de Pinnacle.

Este script SOLO evalua la parte de MONEYLINE (probabilidad de victoria) --
la evaluacion de cobertura de spread (donde realmente vive el mercado de
NFL) es el siguiente paso, una vez que este baseline este confirmado. No se
mezclan las dos evaluaciones en la misma corrida para poder leer cada
resultado sin ambiguedad, mismo criterio de "una cosa a la vez, confirmar,
seguir" que goberno toda la sesion.

Uso: python -m src.models.backtest_nfl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nfl_margin_model import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "matches_clean.csv"


def _brier_binary(model_prob_home: pd.Series, ftr: pd.Series) -> float:
    """Brier score de 2 vias -- objetivo: 1.0 si gano el local, 0.0 si gano
    el visitante, 0.5 si empataron (mismo criterio que el resultado 'R' de
    Elo para empates, ver add_nfl_elo_features.py -- un empate no es ni
    acierto ni error total del modelo, es el punto medio)."""
    actual = ftr.map({"H": 1.0, "A": 0.0, "T": 0.5})
    return float(((model_prob_home - actual) ** 2).mean())


def _blend_weight(model_brier: float, market_brier: float) -> float:
    """Peso asignado al MERCADO -- mismo criterio 'Benter Boost' que
    compute_blend_weight en blending.py (futbol): mas peso al lado con
    MENOR Brier (mejor calibracion). Se reimplementa aca en vez de
    importar blending.py porque esa version esta escrita para el caso
    multiclase (H/D/A) de futbol/tenis-3-vias -- este es un caso binario,
    formula equivalente pero mas simple."""
    return model_brier / (model_brier + market_brier)


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre clean_nfl_data.py y add_nfl_elo_features.py primero.")
        return

    df = pd.read_csv(DATA_PATH)
    if "home_elo" not in df.columns:
        print("[SKIP] Falta home_elo/away_elo -- corre 'python -m src.processing.add_nfl_elo_features' primero.")
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
    print(f"\n=== Resultados FUERA DE MUESTRA -- NFL, moneyline (margen de puntos -> Normal -> P(gana local)) ===")
    print(f"Partidos evaluados OOS (todas las temporadas, {ordered_seasons[1]}-{ordered_seasons[-1]}): {n_total}")
    print(f"Brier score modelo propio (TODAS las temporadas OOS, incluye las sin mercado confiable): {model_brier_all:.6f}")

    reliable = oos_df["reliable_moneyline"] & oos_df["market_prob_home"].notna()
    n_reliable = int(reliable.sum())
    n_excluded = n_total - n_reliable
    print(f"\n[AVISO] {n_excluded} de {n_total} partidos OOS excluidos de la comparacion contra mercado: "
          f"sin moneyline confiable (season<2010) o sin probabilidad de mercado calculable. "
          f"Ver clean_nfl_data.py/nfl_data_loader.py para el detalle real de cobertura por temporada.")

    subset = oos_df.loc[reliable]
    model_brier = _brier_binary(subset["model_prob_home"], subset["FTR"])
    market_brier = _brier_binary(subset["market_prob_home"], subset["FTR"])
    market_weight = _blend_weight(model_brier, market_brier)
    blended = market_weight * subset["market_prob_home"] + (1.0 - market_weight) * subset["model_prob_home"]
    blend_brier = _brier_binary(blended, subset["FTR"])

    print(f"\n--- Subconjunto con mercado confiable (season>=2010): {n_reliable} partidos ---")
    print(f"Brier score modelo propio:          {model_brier:.6f}")
    print(f"Brier score mercado (no-vig, moneyline americano): {market_brier:.6f}")
    print(f"Peso asignado al mercado:           {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):   {blend_brier:.6f}")
    print(f"Gap blend vs. mercado:               {blend_brier - market_brier:+.6f} "
          f"({'el blend gana' if blend_brier < market_brier else 'el mercado sigue ganando'})")

    out_path = PROCESSED_DATA_DIR / "NFL" / "model_predictions_oos_walkforward_v1.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nfl.py",
        model_name="nfl_margin_normal",
        model_version="v1",
        data_paths=[DATA_PATH],
        features="point_margin ~ elo_diff (home_elo - away_elo), sigma = desvio de residuos de training, "
                 "Elo con ajuste MOV + regresion a la media entre temporadas (ver add_nfl_elo_features.py)",
        hyperparameters={
            "elo_k_factor": 20.0,
            "elo_home_advantage": 48.0,
            "elo_season_regression": 1.0 / 3.0,
            "reliable_odds_start_season": 2010,
        },
        metrics={
            "n_total": n_total,
            "n_reliable_market": n_reliable,
            "model_brier_all_seasons": model_brier_all,
            "model_brier_reliable_subset": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": blend_brier - market_brier,
        },
        predictions_path=out_path,
        notes="Primer modelo real de NFL -- moneyline derivado de una distribucion de margen de puntos "
              "(Normal(mu,sigma) sobre diferencia de Elo con MOV), no un clasificador reciclado de "
              "futbol/tenis. Evaluacion de cobertura de spread (el mercado real de NFL) es el siguiente "
              "paso, no incluido en esta corrida.",
    )


if __name__ == "__main__":
    run()