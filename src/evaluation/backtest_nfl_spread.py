"""
Fase 10 -- segunda mitad de la evaluacion de NFL: cobertura de SPREAD, el
mercado real donde vive el dinero en NFL (a diferencia del moneyline, que
en NFL es mas bien secundario -- ver nfl_data_loader.py/nfl_margin_model.py
para la justificacion completa de por que el objeto de prediccion correcto
es una distribucion de margen, no un clasificador binario).

Reusa las predicciones OOS ya calculadas y guardadas por backtest_nfl.py
(`model_predictions_oos_walkforward_v1.csv`) -- NO vuelve a correr el
walk-forward completo, misma logica de reuso que la capa de evaluacion de
futbol/tenis (economic_backtest.py lee el CSV de backtest_v4.py en vez de
reentrenar). `model_prob_home_covers` ya esta calculado ahi (Normal(mu,
sigma) evaluada en spread_line, ver nfl_margin_model.py).

**Punto de disciplina explicito, no asumido**: la cobertura de
`home_spread_odds`/`away_spread_odds` (el PRECIO/vig del spread, distinto
de `spread_line`, que es solo el numero) nunca se confirmo por separado en
el probe original de nfl_data_loader.py -- ese probe solo confirmo
cobertura de spread_line/total_line (el NUMERO), no de los odds del spread
(el PRECIO). Este script imprime esa cobertura real por temporada ANTES de
calcular nada, en vez de asumir que se comporta igual que spread_line.

**Manejo de push**: un partido donde point_margin == spread_line exacto no
es victoria ni derrota para nadie (se devuelve el stake) -- se EXCLUYE
explicitamente del calculo de Brier (necesita un resultado binario
definido), pero se GUARDA en el CSV de salida con una columna `is_push`
explicita -- un push es un resultado real de apuesta (stake devuelto, P&L
cero) que el futuro backtest economico necesita poder simular, no un dato
para tirar. (Bug real, autocorregido antes de construir nada encima: la
primera version de este script excluia los push tambien del CSV guardado,
lo que le habria escondido esos partidos al backtest economico por
completo, no solo al Brier.)

Reusa `_american_to_prob`/`_remove_vig_two_way` de clean_nfl_data.py (misma
formula de odds americanas, una sola fuente de verdad) para calcular la
probabilidad de mercado de cobertura de spread a partir de
home_spread_odds/away_spread_odds.

Uso: python -m src.evaluation.backtest_nfl_spread
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.processing.clean_nfl_data import _american_to_prob, _remove_vig_two_way

PREDICTIONS_PATH = PROCESSED_DATA_DIR / "NFL" / "model_predictions_oos_walkforward_v1.csv"


def _brier_binary(prob_home_covers: pd.Series, actual_home_covers: pd.Series) -> float:
    return float(((prob_home_covers - actual_home_covers) ** 2).mean())


def _blend_weight(model_brier: float, market_brier: float) -> float:
    """Mismo criterio 'Benter Boost' que backtest_nfl.py -- mas peso al lado
    con menor Brier."""
    return model_brier / (model_brier + market_brier)


def run() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"[SKIP] No existe {PREDICTIONS_PATH} -- corre 'python -m src.models.backtest_nfl' primero.")
        return

    df = pd.read_csv(PREDICTIONS_PATH)
    n_total = len(df)

    # Confirmar cobertura real de los ODDS de spread (no solo la linea) antes
    # de asumir nada -- nunca se confirmo esto por separado en el probe original.
    print("Cobertura real de home_spread_odds/away_spread_odds por temporada (partidos OOS):")
    coverage = df.groupby("season").agg(
        n_partidos=("game_id", "count"),
        n_spread_odds=("home_spread_odds", lambda s: s.notna().sum()),
    )
    print(coverage.to_string())

    has_odds = df["home_spread_odds"].notna() & df["away_spread_odds"].notna()
    n_with_odds = int(has_odds.sum())
    print(f"\nTotal con odds de spread disponibles: {n_with_odds} de {n_total} "
          f"({n_with_odds / n_total:.1%}).")

    # Push: el margen cayo exacto sobre la linea -- no es victoria ni derrota
    # para nadie (stake devuelto). Se marca explicitamente, NO se descarta del
    # dataset guardado -- solo se excluye del calculo de Brier (que necesita
    # un resultado binario definido).
    is_push_full = df["point_margin"] == df["spread_line"]
    n_push = int((has_odds & is_push_full).sum())
    print(f"Partidos push (margen == spread_line exacto, dentro del universo con odds): {n_push}.")

    usable = has_odds & df["model_prob_home_covers"].notna()
    scored = df.loc[usable].copy()
    n_usable = len(scored)
    print(f"\nPartidos usables (con odds, con prediccion del modelo, push INCLUIDO): {n_usable}")

    scored["is_push"] = scored["point_margin"] == scored["spread_line"]
    scored["home_covers"] = np.where(
        scored["is_push"], np.nan, (scored["point_margin"] > scored["spread_line"]).astype(float)
    )

    # Probabilidad de mercado (no-vig) se calcula para TODOS los partidos con
    # odds, push incluido -- antes del partido nadie sabe que va a pushear.
    scored["home_spread_prob_raw"] = scored["home_spread_odds"].apply(_american_to_prob)
    scored["away_spread_prob_raw"] = scored["away_spread_odds"].apply(_american_to_prob)
    novig = scored.apply(
        lambda r: _remove_vig_two_way(r["home_spread_prob_raw"], r["away_spread_prob_raw"]), axis=1
    )
    scored["market_prob_home_covers"], scored["market_prob_away_covers"] = zip(*novig)

    # Brier y peso del blend SOLO sobre partidos sin push (necesitan un
    # resultado binario definido) -- pero el blend en si se aplica y se
    # guarda para TODOS los partidos con odds, push incluido, para que el
    # backtest economico tenga probabilidad de apuesta en cada partido real.
    non_push = scored.loc[~scored["is_push"]]
    model_brier = _brier_binary(non_push["model_prob_home_covers"], non_push["home_covers"])
    market_brier = _brier_binary(non_push["market_prob_home_covers"], non_push["home_covers"])
    market_weight = _blend_weight(model_brier, market_brier)

    scored["blend_prob_home_covers"] = (
        market_weight * scored["market_prob_home_covers"] + (1.0 - market_weight) * scored["model_prob_home_covers"]
    )
    scored["blend_prob_away_covers"] = 1.0 - scored["blend_prob_home_covers"]

    blend_brier = _brier_binary(
        scored.loc[~scored["is_push"], "blend_prob_home_covers"], non_push["home_covers"]
    )

    print(f"\n=== Resultados FUERA DE MUESTRA -- NFL, COBERTURA DE SPREAD (el mercado real de NFL) ===")
    print(f"Partidos evaluados en el Brier (sin push): {len(non_push)}")
    print(f"% de veces que cubre el local, sin push (chequeo de sanidad -- deberia rondar 50%, "
          f"el spread esta diseñado para eso): {non_push['home_covers'].mean():.2%}")
    print(f"Brier score modelo propio:               {model_brier:.6f}")
    print(f"Brier score mercado (no-vig, odds de spread americanos): {market_brier:.6f}")
    print(f"Peso asignado al mercado:                {market_weight:.1%}")
    print(f"Brier score blend (Benter Boost):        {blend_brier:.6f}")
    print(f"Gap blend vs. mercado:                    {blend_brier - market_brier:+.6f} "
          f"({'el blend gana' if blend_brier < market_brier else 'el mercado sigue ganando'})")

    out_path = PROCESSED_DATA_DIR / "NFL" / "spread_evaluation_v1.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path} ({n_usable} partidos, incluye {n_push} push marcados con is_push=True)")


if __name__ == "__main__":
    run()