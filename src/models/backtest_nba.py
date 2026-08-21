"""
Fase 10 (NBA) -- backtest walk-forward de NBA, mismo patron honesto que
backtest_nfl.py: entrena con temporadas anteriores, evalua la siguiente,
ventana expansiva, sin fuga. Usa el modelo de margen de puntos
(`nba_margin_model.py`, Normal(mu, sigma) sobre diferencia de Elo).

**v2 de este script (2026-08-21) -- ya CON comparacion de mercado real**:
la v1 original solo reportaba el Brier del modelo propio porque
`games_clean.csv` no tenia cuotas todavia. Ahora que
`merge_theoddsapi_nba.py` ya pego el consenso no-vig de mercado (confirmado:
95.5% de cobertura sobre el rango con cuotas disponibles, 2020-10-01 en
adelante -- ver roadmap), este script usa `games_clean_with_odds.csv` y
agrega el blend Benter Boost, mismo criterio de ponderacion por error
inverso (`_blend_weight`) ya usado en `backtest_nfl.py` -- reimplementado
aca en vez de importado porque ese es el caso binario simple de NFL, no el
multiclase de `blending.py` (futbol).

**Diferencia real frente a NFL, documentada explicita**: NFL excluye
season<2010 de la comparacion de mercado (moneyline no confiable antes de
esa fecha). NBA no tiene ese problema de COBERTURA-por-temporada -- el
problema real es de RANGO: el proveedor de cuotas (The Odds API) solo tiene
historico desde 2020-06-06, asi que la comparacion de mercado
automaticamente se limita a los partidos con `market_prob_home` no nulo
(2020-21 en adelante) -- no hace falta un corte adicional por temporada,
el propio merge ya deja NaN donde no hay cuota real.

Si `games_clean_with_odds.csv` no existe todavia, cae de vuelta a
`games_clean.csv` y reporta SOLO el Brier del modelo (comportamiento v1),
con aviso explicito -- nunca inventa un numero de mercado.

Uso: python -m src.models.backtest_nba
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.nba_margin_model import fit_margin_model, predict_dataframe
from src.tracking.run_logger import log_run

DATA_PATH_WITH_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean_with_odds.csv"
DATA_PATH_NO_ODDS = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"


def _brier_binary(model_prob_home: pd.Series, ftr: pd.Series) -> float:
    """Brier score de 2 vias -- NBA no tiene empate (confirmado, 0 casos
    reales en clean_nba_data.py), asi que a diferencia de NFL no hace falta
    mapear un caso 'T'=0.5."""
    actual = ftr.map({"H": 1.0, "A": 0.0})
    return float(((model_prob_home - actual) ** 2).mean())


def _blend_weight(model_brier: float, market_brier: float) -> float:
    """Peso asignado al MERCADO -- mismo criterio 'Benter Boost' que
    _blend_weight en backtest_nfl.py: mas peso al lado con MENOR Brier."""
    return model_brier / (model_brier + market_brier)


def run() -> None:
    has_odds = DATA_PATH_WITH_ODDS.exists()
    data_path = DATA_PATH_WITH_ODDS if has_odds else DATA_PATH_NO_ODDS
    if not data_path.exists():
        print(f"[SKIP] No existe {data_path} -- corre clean_nba_data.py y add_nba_elo_features.py primero.")
        return

    df = pd.read_csv(data_path)
    if "home_elo" not in df.columns:
        print("[SKIP] Falta home_elo/away_elo -- corre 'python -m src.processing.add_nba_elo_features' primero.")
        return
    if not has_odds:
        print(f"[AVISO] No existe {DATA_PATH_WITH_ODDS} -- corre 'python -m src.processing.merge_theoddsapi_nba' "
              f"primero para tener comparacion de mercado. Esta corrida SOLO reporta el modelo propio.")

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
    print(f"Brier score modelo propio (TODAS las temporadas OOS, incluye las sin mercado): {model_brier_all:.6f}")

    home_win_rate_oos = (oos_df["FTR"] == "H").mean()
    print(f"% de victorias reales del local en el conjunto OOS (referencia, no un target): {home_win_rate_oos:.2%}")

    metrics = {
        "n_total": n_total,
        "model_brier_all_seasons": model_brier_all,
        "home_win_rate_oos": home_win_rate_oos,
    }

    if has_odds and "market_prob_home" in oos_df.columns:
        reliable = oos_df["market_prob_home"].notna()
        n_reliable = int(reliable.sum())
        n_excluded = n_total - n_reliable
        print(f"\n[AVISO] {n_excluded} de {n_total} partidos OOS excluidos de la comparacion contra mercado: "
              f"sin cuota historica disponible (fuera del rango 2020-10-01 en adelante del proveedor, o "
              f"partido sin match en el merge -- ver merge_theoddsapi_nba.py).")

        subset = oos_df.loc[reliable]
        model_brier = _brier_binary(subset["model_prob_home"], subset["FTR"])
        market_brier = _brier_binary(subset["market_prob_home"], subset["FTR"])
        market_weight = _blend_weight(model_brier, market_brier)
        blended = market_weight * subset["market_prob_home"] + (1.0 - market_weight) * subset["model_prob_home"]
        blend_brier = _brier_binary(blended, subset["FTR"])

        print(f"\n--- Subconjunto con mercado disponible (2020-21 en adelante): {n_reliable} partidos ---")
        print(f"Brier score modelo propio:                    {model_brier:.6f}")
        print(f"Brier score mercado (consenso no-vig, ~10 casas retail): {market_brier:.6f}")
        print(f"Peso asignado al mercado:                     {market_weight:.1%}")
        print(f"Brier score blend (Benter Boost):              {blend_brier:.6f}")
        print(f"Gap blend vs. mercado:                          {blend_brier - market_brier:+.6f} "
              f"({'el blend gana' if blend_brier < market_brier else 'el mercado sigue ganando'})")

        metrics.update({
            "n_reliable_market": n_reliable,
            "model_brier_reliable_subset": model_brier,
            "market_brier": market_brier,
            "market_weight": market_weight,
            "blend_brier": blend_brier,
            "gap_vs_mercado": blend_brier - market_brier,
        })
    else:
        print(f"\n[AVISO] Sin comparacion contra mercado en esta corrida -- ver aviso arriba.")

    out_path = PROCESSED_DATA_DIR / "NBA" / "model_predictions_oos_walkforward_v1.csv"
    oos_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    log_run(
        script="backtest_nba.py",
        model_name="nba_margin_normal",
        model_version="v1" if not has_odds else "v1_con_mercado",
        data_paths=[data_path],
        features="point_margin ~ elo_diff (home_elo - away_elo), sigma = desvio de residuos de training, "
                 "Elo con ajuste MOV NBA + regresion a la media entre temporadas (ver add_nba_elo_features.py)",
        hyperparameters={
            "elo_k_factor": 20.0,
            "elo_home_advantage": 100.0,
            "elo_season_regression": 0.25,
        },
        metrics=metrics,
        predictions_path=out_path,
        notes="Modelo de NBA -- moneyline derivado de una distribucion de margen de puntos "
              "(Normal(mu,sigma) sobre diferencia de Elo con MOV NBA), mismo diseño que nfl_margin_model.py. "
              + ("Con comparacion real de mercado (consenso no-vig multi-libro, The Odds API, 2020-21 en "
                 "adelante) y blend Benter Boost." if has_odds else
                 "SIN comparacion contra mercado (games_clean_with_odds.csv no existe todavia)."),
    )


if __name__ == "__main__":
    run()
