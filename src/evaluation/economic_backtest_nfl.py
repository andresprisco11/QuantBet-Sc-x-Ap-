"""
Fase 10 -- primer backtest economico real de NFL: Kelly fraccional sobre el
mercado de SPREAD (el que realmente importa en NFL, ver
backtest_nfl_spread.py). Mismo patron que economic_backtest.py (futbol) /
economic_backtest_tennis.py / economic_backtest_mls.py: apostar al lado con
mayor edge, staking Kelly fraccional, ejecucion simulada a la cuota real
disponible, medir ROI/drawdown/win rate.

**Diferencia real: ODDS AMERICANAS**, no decimales -- el calculo de "b"
(ganancia neta por unidad de stake) usa `_american_to_b` (nueva, agregada a
clean_nfl_data.py junto a `_american_to_prob` -- no requiere re-correr la
limpieza, es una funcion nueva sin efecto en el CSV ya generado):
  odds > 0: b = odds / 100
  odds < 0: b = 100 / abs(odds)
Edge = blend_prob*(1+b) - 1 -- MISMA definicion de edge que el resto del
proyecto usa con cuotas decimales (equivalente al numerador de Kelly), solo
cambia como se calcula "b".

**Manejo de push, real**: si el partido apostado termina en push
(`is_push=True`, ya viene marcado desde `backtest_nfl_spread.py`), el stake
se devuelve completo -- P&L=0 para esa apuesta, ni gana ni pierde.

**REGLA DE STAKING -- corrida DIAGNOSTICA, NO tuneada todavia**: mismo
criterio que la primerisima corrida de `economic_backtest.py` en futbol
(antes de `tune_staking_rules.py`): `min_edge=2%`, `kelly=25%`, SIN techo
de cuota. A proposito generico -- el proyecto nunca tunea antes de ver un
primer resultado diagnostico, mismo orden ya seguido en futbol/tenis/MLS.

Uso: python -m src.evaluation.economic_backtest_nfl
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.processing.clean_nfl_data import _american_to_b

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "spread_evaluation_v1.csv"

MIN_EDGE = 0.02
KELLY_FRACTION = 0.25
STARTING_BANKROLL = 1000.0


def _select_bets(df: pd.DataFrame, min_edge: float = MIN_EDGE, max_edge: float = None) -> pd.DataFrame:
    """max_edge=None (default) -- sin techo, comportamiento original de la
    corrida diagnostica. Un techo real (max_edge no-None) descarta la
    apuesta si el edge detectado supera ese valor -- mismo remedio que
    edge_ceiling_sweep.py/economic_backtest.py ya adoptaron en futbol
    (La Liga/Serie A/Bundesliga) despues de que edge_magnitude_calibration_
    check.py mostrara que el ROI cae con la magnitud del edge -- exactamente
    el mismo patron que nfl_selection_bias_check.py encontro aca (el
    acierto real cae a medida que la probabilidad/edge declarada sube)."""
    df = df.copy()
    df["b_home"] = df["home_spread_odds"].apply(_american_to_b)
    df["b_away"] = df["away_spread_odds"].apply(_american_to_b)

    df["edge_home"] = df["blend_prob_home_covers"] * (1.0 + df["b_home"]) - 1.0
    df["edge_away"] = df["blend_prob_away_covers"] * (1.0 + df["b_away"]) - 1.0

    bets = []
    for row in df.itertuples(index=False):
        if row.edge_home >= row.edge_away:
            best_side, best_edge, best_b, best_prob = "home", row.edge_home, row.b_home, row.blend_prob_home_covers
        else:
            best_side, best_edge, best_b, best_prob = "away", row.edge_away, row.b_away, row.blend_prob_away_covers

        if best_edge < min_edge:
            continue
        if max_edge is not None and best_edge > max_edge:
            continue

        if row.is_push:
            won = np.nan
        elif best_side == "home":
            won = bool(row.home_covers == 1.0)
        else:
            won = bool(row.home_covers == 0.0)

        bets.append({
            "gameday": row.gameday, "season": row.season, "side": best_side,
            "edge": best_edge, "b": best_b, "prob": best_prob,
            "is_push": row.is_push, "won": won,
        })
    return pd.DataFrame(bets)


def _simulate_bankroll(bets: pd.DataFrame, kelly_fraction: float = KELLY_FRACTION,
                        starting_bankroll: float = STARTING_BANKROLL) -> dict:
    bets = bets.sort_values("gameday").reset_index(drop=True)
    bankroll = starting_bankroll
    peak = starting_bankroll
    max_drawdown = 0.0
    curve = []

    for row in bets.itertuples(index=False):
        kelly_stake_fraction = max(0.0, row.edge / row.b) * kelly_fraction
        stake = bankroll * kelly_stake_fraction

        if row.is_push:
            pnl = 0.0
        elif row.won:
            pnl = stake * row.b
        else:
            pnl = -stake

        bankroll += pnl
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        curve.append(bankroll)

    n_bets = len(bets)
    n_push = int(bets["is_push"].sum())
    n_decided = n_bets - n_push
    n_won = int((bets["won"] == True).sum())
    win_rate = n_won / n_decided if n_decided else float("nan")
    roi = (bankroll - starting_bankroll) / starting_bankroll

    return {
        "n_bets": n_bets, "n_push": n_push, "n_decided": n_decided,
        "win_rate": win_rate, "final_bankroll": bankroll, "roi": roi, "max_drawdown": max_drawdown,
    }


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre 'python -m src.evaluation.backtest_nfl_spread' primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["gameday"] = pd.to_datetime(df["gameday"])

    bets = _select_bets(df, MIN_EDGE)
    print(f"Apuestas seleccionadas (min_edge={MIN_EDGE:.0%}): {len(bets)} de {len(df)} partidos evaluados.")

    if bets.empty:
        print("[AVISO] Ninguna apuesta paso el umbral de edge -- no hay nada que simular.")
        return

    result = _simulate_bankroll(bets, KELLY_FRACTION, STARTING_BANKROLL)

    print(f"\n=== Resultados backtest economico DIAGNOSTICO -- NFL, spread "
          f"(min_edge={MIN_EDGE:.0%}, kelly={KELLY_FRACTION:.0%}, sin techo de cuota) ===")
    print(f"Apuestas totales: {result['n_bets']} (push: {result['n_push']}, decididas: {result['n_decided']})")
    print(f"Win rate (sobre decididas, push excluido del calculo): {result['win_rate']:.2%}")
    print(f"Bankroll: {STARTING_BANKROLL:.2f} -> {result['final_bankroll']:.2f}")
    print(f"ROI: {result['roi']:+.2%}")
    print(f"Drawdown maximo: {result['max_drawdown']:.2%}")

    out_path = PROCESSED_DATA_DIR / "NFL" / "economic_backtest_nfl_bets.csv"
    bets.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\n[NOTA] Corrida DIAGNOSTICA con parametros genericos (min_edge=2%, kelly=25%, sin techo de "
          "cuota) -- mismo punto de partida que tuvo futbol antes de tune_staking_rules.py. No se tunea "
          "todavia. Si esto muestra alguna señal, el proximo paso natural es un barrido de staking como "
          "el que ya se hizo en futbol/tenis/MLS.")


if __name__ == "__main__":
    run()