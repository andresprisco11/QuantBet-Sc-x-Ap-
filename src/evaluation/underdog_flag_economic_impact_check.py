"""
`underdog_divergence_exclusion_check.py` (2026-08-20, confirmado por el
usuario) encontró que, de las 2,184 apuestas que las reglas de producción ya
seleccionan hoy (4 ligas europeas + MLS + los 2 candidatos de tenis), 363
(16.6%) caen en la bandera roja (modelo de habilidad >=60% seguro de un lado
que el mercado da <50%) -- y ese subconjunto acierta 40.50% real contra
47.34% del resto. PERO ese resultado se mide en ACIERTO (win rate), no en
ROI -- y el desglose por mercado NO es uniforme: La Liga (-16pp, n=35) y MLS
(-2pp, n=219, el mayor volumen) van en la dirección esperada, pero Serie A
se REVIERTE (+15.65pp, n=23) y EPL/Bundesliga muestran gaps chicos (~2pp,
n=57/29) -- exactamente el patrón de "puede ser real en agregado pero
todavía no se sabe si es una meseta o unos pocos casos ruidosos" que el
proyecto siempre exige confirmar con Kelly real antes de adoptar nada (mismo
estándar que ya se aplicó al techo de edge por liga, `edge_ceiling_sweep.py`
-> `economic_backtest.py` con Kelly real, 2026-08-19).

Este script cierra ese paso: mide el impacto ECONÓMICO real (ROI, drawdown,
bankroll final con Kelly fraccional -- el staking que efectivamente usa el
proyecto, no solo win rate) de EXCLUIR la bandera roja de las reglas YA
ADOPTADAS, comparando el backtest CON vs. SIN el filtro, por mercado y en
total. No decide nada todavía -- mide, con el mismo rigor que el resto del
proyecto, si vale la pena adoptarlo formalmente.

No reimplementa Kelly ni la lógica de selección: reutiliza
`_select_bets`/`load_eval_df`/`MAX_EDGE_CEILING_BY_LEAGUE`/`_simulate_bankroll`
de `economic_backtest.py` (fútbol, y MLS que ya importa la misma
`_simulate_bankroll`), `_select_bets_mls`/`load_eval_df_mls` de
`economic_backtest_mls.py`, y `_select_bets`/`load_predictions`/
`_simulate_bankroll` de `economic_backtest_tennis.py` (con alias explícitos
para evitar colisión de nombres entre las dos versiones de cada función).

Salida: por mercado y en TOTAL combinado, ROI/win rate/drawdown máximo/
bankroll final CON vs. SIN la bandera roja excluida. Guarda el detalle en
'data/runs/underdog_flag_economic_impact_check.csv'.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR
from src.evaluation.economic_backtest import (
    _select_bets as _select_bets_football,
    _simulate_bankroll as _simulate_bankroll_football,
    load_eval_df, MAX_EDGE_CEILING_BY_LEAGUE, MIN_EDGE_THRESHOLD, MAX_ODDS,
)
from src.evaluation.economic_backtest_mls import _select_bets_mls, load_eval_df_mls
from src.evaluation.economic_backtest_tennis import (
    _select_bets as _select_bets_tennis,
    _simulate_bankroll as _simulate_bankroll_tennis,
    load_predictions,
)

RED_FLAG_MODEL_PROB = 0.60
RED_FLAG_MARKET_PROB = 0.50
TENNIS_KELLY_FRACTION = 0.10  # mismo valor que tennis_temporal_stability_check.py, candidatos ya confirmados.

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


def _summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"n_bets": 0, "roi": float("nan"), "win_rate": float("nan"),
                "max_drawdown": float("nan"), "final_bankroll": float("nan")}
    staked = bets["stake"].sum()
    profit = bets["profit"].sum()
    return {
        "n_bets": len(bets),
        "roi": profit / staked if staked > 0 else float("nan"),
        "win_rate": bets["won"].mean(),
        "max_drawdown": bets["drawdown"].max(),
        "final_bankroll": bets["bankroll_after"].iloc[-1],
    }


def _print_comparison(label: str, baseline: dict, candidate: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  SIN filtro (regla actual)      : n={baseline['n_bets']}, ROI={baseline['roi']:+.2%}, "
          f"win_rate={baseline['win_rate']:.2%}, drawdown_max={baseline['max_drawdown']:.2%}, "
          f"bankroll_final={baseline['final_bankroll']:.2f}")
    print(f"  CON filtro (excluye bandera roja): n={candidate['n_bets']}, ROI={candidate['roi']:+.2%}, "
          f"win_rate={candidate['win_rate']:.2%}, drawdown_max={candidate['max_drawdown']:.2%}, "
          f"bankroll_final={candidate['final_bankroll']:.2f}")
    if baseline["n_bets"] > 0 and candidate["n_bets"] > 0 and pd.notna(baseline["roi"]) and pd.notna(candidate["roi"]):
        delta = candidate["roi"] - baseline["roi"]
        print(f"  Delta de ROI por excluir la bandera roja: {delta:+.2%}")


def run() -> pd.DataFrame:
    all_rows = []

    for league_key in LEAGUES:
        try:
            df_eval = load_eval_df(league_key)
        except FileNotFoundError as e:
            print(f"[SKIP] {league_key}: {e}")
            continue
        ceiling = MAX_EDGE_CEILING_BY_LEAGUE.get(league_key)
        bets = _select_bets_football(df_eval, min_edge_threshold=MIN_EDGE_THRESHOLD, max_odds=MAX_ODDS, max_edge_ceiling=ceiling)
        bets = _flag_football_bets(bets)
        if bets.empty:
            print(f"\n=== {league_key} === [SKIP] sin apuestas.")
            continue
        baseline_sim = _simulate_bankroll_football(bets)
        candidate_sim = _simulate_bankroll_football(bets[~bets["red_flag"]])
        _print_comparison(league_key, _summarize(baseline_sim), _summarize(candidate_sim))
        out = bets[["Date", "bet_side", "bet_odds", "red_flag", "won"]].copy()
        out["market"] = league_key
        all_rows.append(out)

    try:
        df_eval_mls = load_eval_df_mls()
        bets_mls = _select_bets_mls(df_eval_mls)
        bets_mls = _flag_football_bets(bets_mls)
        if not bets_mls.empty:
            baseline_sim = _simulate_bankroll_football(bets_mls)
            candidate_sim = _simulate_bankroll_football(bets_mls[~bets_mls["red_flag"]])
            _print_comparison("MLS", _summarize(baseline_sim), _summarize(candidate_sim))
            out = bets_mls[["Date", "bet_side", "bet_odds", "red_flag", "won"]].copy()
            out["market"] = "MLS"
            all_rows.append(out)
        else:
            print("\n=== MLS === [SKIP] sin apuestas.")
    except FileNotFoundError as e:
        print(f"[SKIP] MLS: {e}")

    for tour, params in TENNIS_CANDIDATES.items():
        try:
            preds = load_predictions(tour)
        except FileNotFoundError as e:
            print(f"[SKIP] TENNIS_{tour}: {e}")
            continue
        bets_tennis = _select_bets_tennis(preds, min_edge_threshold=params["min_edge"], max_odds=params["max_odds"])
        if bets_tennis.empty:
            print(f"\n=== TENNIS_{tour} === [SKIP] sin apuestas.")
            continue
        merge_cols = ["Date", "Player1", "Player2", "model_prob_player1", "market_prob_player1"]
        bets_tennis = bets_tennis.merge(preds[merge_cols], on=["Date", "Player1", "Player2"], how="left")
        p1_side = bets_tennis["side"] == "Player1"
        bets_tennis["model_prob_side"] = bets_tennis["model_prob_player1"].where(p1_side, 1.0 - bets_tennis["model_prob_player1"])
        bets_tennis["market_prob_side"] = bets_tennis["market_prob_player1"].where(p1_side, 1.0 - bets_tennis["market_prob_player1"])
        bets_tennis["red_flag"] = (bets_tennis["model_prob_side"] >= RED_FLAG_MODEL_PROB) & (bets_tennis["market_prob_side"] < RED_FLAG_MARKET_PROB)

        baseline_sim = _simulate_bankroll_tennis(bets_tennis, kelly_fraction=TENNIS_KELLY_FRACTION)
        candidate_sim = _simulate_bankroll_tennis(bets_tennis[~bets_tennis["red_flag"]], kelly_fraction=TENNIS_KELLY_FRACTION)
        _print_comparison(f"TENNIS_{tour}", _summarize(baseline_sim), _summarize(candidate_sim))
        out = bets_tennis[["Date", "side", "odds", "red_flag", "won"]].copy()
        out = out.rename(columns={"side": "bet_side", "odds": "bet_odds"})
        out["market"] = f"TENNIS_{tour}"
        all_rows.append(out)

    if not all_rows:
        print("\n[AVISO] No hay ninguna apuesta seleccionada en ningun mercado -- nada que guardar.")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "underdog_flag_economic_impact_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nGuardado detalle combinado -> {out_path}")
    print("\n[NOTA] Este script MIDE el impacto economico -- no adopta ningun cambio de regla "
          "automaticamente. Antes de sumar este filtro a economic_backtest.py/economic_backtest_mls.py "
          "en produccion hace falta el mismo estandar que el resto del proyecto: confirmar que el ROI "
          "mejora de forma consistente (no solo en el mercado con mas volumen) y, idealmente, que se "
          "sostiene en un chequeo de estabilidad temporal -- no se adopta de un solo resultado agregado.")
    return combined


if __name__ == "__main__":
    run()