"""
`skill_model_underdog_check.py` (2026-08-20, confirmado por el usuario) dio
un resultado real, pero el mensaje automatico que imprime ese script (
"[HALLAZGO] el modelo SI genera senal de underdog, el blend la esta
tapando") es ENGANOSO tal cual esta escrito -- confunde "hay divergencia"
con "la divergencia es buena senal". Los numeros reales dicen lo contrario:
el acierto real de los casos divergentes CAE a medida que sube la confianza
del modelo (60%: 38.75% real: 70%: 35.56%; 80%: 33.33%; 85%: 30.00%), y en
TODOS los umbrales el acierto real queda POR DEBAJO de lo que el propio
mercado ya pensaba de ese lado (ej. umbral 80%: acierto real 33.33% vs.
prob. de mercado promedio 40.16% -- el modelo estuvo peor que el escepticismo
del mercado, no mejor). Esto no es "senal tapada por el blend" -- es
sobreconfianza sistematica del modelo de habilidad exactamente en la
direccion opuesta a donde el usuario quiere pescar underdogs. El blend, al
pesar el mercado, esta protegiendo correctamente contra esto, no tapando
nada de valor.

Este script responde la pregunta que de verdad importa para producto: ¿esta
"bandera roja" ya recien descubierta contamina alguna de las apuestas que
las reglas de staking YA ADOPTADAS (economic_backtest.py por liga con techo
de edge, economic_backtest_mls.py, y los dos candidatos de tenis confirmados
por estabilidad temporal) estan seleccionando hoy? Si una porcion de las
apuestas ya seleccionadas resulta tener esta bandera (modelo de habilidad
>=60% en un lado que el mercado da <50%), excluirla es una mejora candidata
concreta y de bajo riesgo -- no cambia ninguna regla existente, solo le
agrega un filtro adicional a lo que ya esta en produccion.

No reimplementa la logica de seleccion -- importa _select_bets() /
load_eval_df() de economic_backtest.py (futbol, con MAX_EDGE_CEILING_BY_LEAGUE
ya adoptado), _select_bets_mls() de economic_backtest_mls.py, y
_select_bets()/load_predictions() de economic_backtest_tennis.py (tenis, con
los candidatos ATP/WTA ya confirmados por tennis_temporal_stability_check.py).
Para futbol/MLS, el DataFrame de apuestas ya trae model_prob_*/pinnacle_close_prob_*
de cada lado (row.to_dict() completo) -- se lee directo. Para tenis,
_select_bets() no conserva esas columnas -- se hace merge de vuelta contra
predictions_v1.csv por (Date, Player1, Player2) para recuperarlas.

Umbral de bandera roja: model_prob del lado apostado >=60% Y market_prob de
ese mismo lado <50% -- el umbral mas bajo de skill_model_underdog_check.py,
elegido porque ahi es donde hay mas volumen (n=1,747 en el chequeo anterior)
para medir el efecto con algo de confianza estadistica.

Salida: por mercado, cuantas de las apuestas YA SELECCIONADAS caen en la
bandera roja, y el acierto real de "con bandera" vs. "sin bandera" dentro de
ese mismo conjunto ya seleccionado. Guarda el detalle en
'data/runs/underdog_divergence_exclusion_check.csv'.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR
from src.evaluation.economic_backtest import (
    _select_bets as _select_bets_football, load_eval_df, MAX_EDGE_CEILING_BY_LEAGUE,
    MIN_EDGE_THRESHOLD, MAX_ODDS,
)
from src.evaluation.economic_backtest_mls import _select_bets_mls, load_eval_df_mls
from src.evaluation.economic_backtest_tennis import _select_bets as _select_bets_tennis, load_predictions

RED_FLAG_MODEL_PROB = 0.60
RED_FLAG_MARKET_PROB = 0.50

TENNIS_CANDIDATES = {
    "ATP": {"min_edge": 0.01, "max_odds": 2.5},
    "WTA": {"min_edge": 0.08, "max_odds": 6.0},
}

_FOOTBALL_MODEL_COL = {"home": "model_prob_home", "draw": "model_prob_draw", "away": "model_prob_away"}
_FOOTBALL_MARKET_COL = {
    "home": "pinnacle_close_prob_home", "draw": "pinnacle_close_prob_draw", "away": "pinnacle_close_prob_away",
}


def _flag_football_bets(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets
    bets = bets.copy()
    bets["model_prob_side"] = bets.apply(lambda r: r[_FOOTBALL_MODEL_COL[r["bet_side"]]], axis=1)
    bets["market_prob_side"] = bets.apply(lambda r: r[_FOOTBALL_MARKET_COL[r["bet_side"]]], axis=1)
    bets["red_flag"] = (bets["model_prob_side"] >= RED_FLAG_MODEL_PROB) & (bets["market_prob_side"] < RED_FLAG_MARKET_PROB)
    return bets


def _report(label: str, bets: pd.DataFrame, win_col: str = "won") -> None:
    n = len(bets)
    print(f"\n=== {label}: {n} apuestas ya seleccionadas por la regla de produccion ===")
    if n == 0:
        print("  [SKIP] sin apuestas, nada que evaluar.")
        return
    n_flagged = int(bets["red_flag"].sum())
    if n_flagged == 0:
        print(f"  0 de {n} apuestas caen en la bandera roja (model_prob>={RED_FLAG_MODEL_PROB:.0%} "
              f"y market_prob<{RED_FLAG_MARKET_PROB:.0%} en el lado apostado) -- esta regla de "
              f"produccion ya evita naturalmente esta zona, no hace falta ningun filtro nuevo.")
        return
    flagged = bets[bets["red_flag"]]
    clean = bets[~bets["red_flag"]]
    print(f"  Con bandera roja: n={n_flagged} ({n_flagged/n:.1%} del total), "
          f"acierto real={flagged[win_col].mean():.2%}")
    print(f"  Sin bandera roja: n={len(clean)}, acierto real={clean[win_col].mean():.2%}")


def run() -> pd.DataFrame:
    all_flagged_rows = []

    # --- Futbol (4 ligas), regla de produccion ya adoptada (con techo de edge por liga) ---
    for league_key in LEAGUES:
        try:
            df_eval = load_eval_df(league_key)
        except FileNotFoundError as e:
            print(f"[SKIP] {league_key}: {e}")
            continue
        ceiling = MAX_EDGE_CEILING_BY_LEAGUE.get(league_key)
        bets = _select_bets_football(df_eval, min_edge_threshold=MIN_EDGE_THRESHOLD, max_odds=MAX_ODDS, max_edge_ceiling=ceiling)
        bets = _flag_football_bets(bets)
        _report(league_key, bets)
        if not bets.empty:
            out = bets[["Date", "bet_side", "bet_odds", "model_prob_side", "market_prob_side", "red_flag", "won"]].copy()
            out["market"] = league_key
            all_flagged_rows.append(out)

    # --- MLS, regla de produccion propia ---
    try:
        df_eval_mls = load_eval_df_mls()
        bets_mls = _select_bets_mls(df_eval_mls)
        bets_mls = _flag_football_bets(bets_mls)
        _report("MLS", bets_mls)
        if not bets_mls.empty:
            out = bets_mls[["Date", "bet_side", "bet_odds", "model_prob_side", "market_prob_side", "red_flag", "won"]].copy()
            out["market"] = "MLS"
            all_flagged_rows.append(out)
    except FileNotFoundError as e:
        print(f"[SKIP] MLS: {e}")

    # --- Tenis (ATP/WTA), candidatos ya confirmados por estabilidad temporal ---
    for tour, params in TENNIS_CANDIDATES.items():
        try:
            preds = load_predictions(tour)
        except FileNotFoundError as e:
            print(f"[SKIP] TENNIS_{tour}: {e}")
            continue
        bets_tennis = _select_bets_tennis(preds, min_edge_threshold=params["min_edge"], max_odds=params["max_odds"])
        if bets_tennis.empty:
            _report(f"TENNIS_{tour}", bets_tennis)
            continue
        merge_cols = ["Date", "Player1", "Player2", "model_prob_player1", "market_prob_player1"]
        bets_tennis = bets_tennis.merge(preds[merge_cols], on=["Date", "Player1", "Player2"], how="left")
        p1_side = bets_tennis["side"] == "Player1"
        bets_tennis["model_prob_side"] = bets_tennis["model_prob_player1"].where(p1_side, 1.0 - bets_tennis["model_prob_player1"])
        bets_tennis["market_prob_side"] = bets_tennis["market_prob_player1"].where(p1_side, 1.0 - bets_tennis["market_prob_player1"])
        bets_tennis["red_flag"] = (bets_tennis["model_prob_side"] >= RED_FLAG_MODEL_PROB) & (bets_tennis["market_prob_side"] < RED_FLAG_MARKET_PROB)
        _report(f"TENNIS_{tour}", bets_tennis)
        out = bets_tennis[["Date", "side", "odds", "model_prob_side", "market_prob_side", "red_flag", "won"]].copy()
        out = out.rename(columns={"side": "bet_side", "odds": "bet_odds"})
        out["market"] = f"TENNIS_{tour}"
        all_flagged_rows.append(out)

    if not all_flagged_rows:
        print("\n[AVISO] No hay ninguna apuesta seleccionada en ningun mercado -- nada que guardar.")
        return pd.DataFrame()

    combined = pd.concat(all_flagged_rows, ignore_index=True)
    n_total = len(combined)
    n_flagged_total = int(combined["red_flag"].sum())
    print(f"\n\n=== TOTAL COMBINADO -- apuestas ya seleccionadas por las reglas de produccion, "
          f"los {combined['market'].nunique()} mercados juntos: {n_total} ===")
    if n_flagged_total == 0:
        print(f"  0 de {n_total} caen en la bandera roja -- las reglas actuales ya evitan esta zona "
              f"por completo, no hace falta ningun cambio.")
    else:
        flagged = combined[combined["red_flag"]]
        clean = combined[~combined["red_flag"]]
        print(f"  Con bandera roja: n={n_flagged_total} ({n_flagged_total/n_total:.1%} del total), "
              f"acierto real={flagged['won'].mean():.2%}")
        print(f"  Sin bandera roja: n={len(clean)}, acierto real={clean['won'].mean():.2%}")
        print(f"\n  [LECTURA] Si el acierto 'con bandera' es notablemente menor que 'sin bandera', "
              f"excluir estos casos de la seleccion (filtro adicional, sin tocar ninguna regla ya "
              f"adoptada) es una mejora candidata de bajo riesgo -- se confirmaria formalmente con un "
              f"barrido antes de adoptarla, mismo criterio de todo el proyecto.")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "underdog_divergence_exclusion_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nGuardado detalle combinado -> {out_path}")
    return combined


if __name__ == "__main__":
    run()