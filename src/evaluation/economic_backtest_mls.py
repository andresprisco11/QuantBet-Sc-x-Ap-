"""
Fase 8b: backtest economico para MLS -- MISMO framework (Kelly fraccional,
bankroll compuesto cronologico, drawdown) que economic_backtest.py, pero
con una diferencia METODOLOGICA real, no solo de nombres de columna.

Por que un script aparte, no un parametro mas de economic_backtest.py:
en las 4 ligas europeas, el precio de EJECUCION es la cuota de APERTURA de
Pinnacle (PSH/PSD/PSA), y la probabilidad "justa" (blend_prob) incorpora
informacion de la cuota de CIERRE -- eso es explotar que la linea se movio
entre apertura y cierre (la esencia de CLV: comprar antes de que el precio
se corrija). MLS no tiene apertura -- la unica cuota disponible ya ES la
de cierre (ver clean_data.py, mls_loader.py). Eso significa dos cosas
importantes, no cosmeticas:

1. NO hay CLV que calcular para MLS. CLV requiere DOS precios (apertura Y
   cierre) para medir el movimiento entre ellos -- con un solo precio
   disponible, el concepto no aplica. Este script no inventa un proxy ni
   lo deja en NaN silencioso: directamente no calcula ni imprime columnas
   de CLV.
2. La pregunta que este backtest puede responder para MLS es MAS DURA que
   la de las otras 4 ligas: no "aprovechaste una linea temprana barata
   antes de que se corrigiera", sino "le ganaste al precio de cierre
   MISMO, el precio mas eficiente que existe para este partido". blend_prob
   sigue siendo una mezcla model+mercado (Benter Boost), asi que SI puede
   haber edge real (la mitad del peso es el modelo propio, no el mercado)
   -- pero el listón es mas alto que en las otras 4 ligas. No comparar el
   ROI de MLS 1 a 1 contra el de EPL/LaLiga/SerieA/Bundesliga esperando el
   mismo tipo de señal.

Reutiliza _simulate_bankroll de economic_backtest.py sin modificarlo (es
pura mecanica de staking, no depende de CLV). _select_bets_mls es una
version propia -- la de economic_backtest.py esta atada a las columnas de
apertura/cierre de Pinnacle que MLS no tiene.

Los hiperparametros de staking (KELLY_FRACTION/MIN_EDGE_THRESHOLD/MAX_ODDS)
arrancan en los mismos valores tuneados sobre EPL, con el mismo
disclaimer que ya se aplico a Serie A/Bundesliga/La Liga: es un default
global, NO una regla ya validada para MLS -- si hace falta, se puede
extender tune_staking_rules.py a esta liga despues de ver el resultado
base.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.evaluation.economic_backtest import _simulate_bankroll
from src.tracking.run_logger import log_run

LEAGUE_KEY = "MLS"
KELLY_FRACTION = 0.10          # default global tuneado sobre EPL -- sin validar todavia para MLS.
MIN_EDGE_THRESHOLD = 0.08      # idem.
MAX_STAKE_FRACTION = 0.05
MAX_ODDS = 3.0                 # idem.
INITIAL_BANKROLL = 1000.0

# A diferencia de economic_backtest.py (que usa PSH/PSD/PSA de apertura),
# ac se ejecuta a la cuota de CIERRE de Pinnacle -- es la unica que existe.
SIDES = [
    ("home", "PSCH", "blend_prob_home", "H"),
    ("draw", "PSCD", "blend_prob_draw", "D"),
    ("away", "PSCA", "blend_prob_away", "A"),
]


def _select_bets_mls(df: pd.DataFrame, min_edge_threshold: float = MIN_EDGE_THRESHOLD,
                      max_odds: float = MAX_ODDS) -> pd.DataFrame:
    """Igual que _select_bets() de economic_backtest.py en espiritu (mismo criterio
    de edge/Kelly), pero sin CLV -- no hay apertura contra la cual medirlo."""
    records = []
    for _, row in df.iterrows():
        best = None
        for side_name, odds_col, prob_col, ftr_code in SIDES:
            odds = row[odds_col]
            fair_prob = row[prob_col]
            if pd.isna(odds) or pd.isna(fair_prob) or odds <= 1.0:
                continue
            if max_odds is not None and odds > max_odds:
                continue
            edge = fair_prob * odds - 1.0
            if best is None or edge > best["edge"]:
                best = {"side": side_name, "odds": odds, "fair_prob": fair_prob,
                        "edge": edge, "ftr_code": ftr_code}
        if best is not None and best["edge"] > min_edge_threshold:
            kelly_full = (best["fair_prob"] * best["odds"] - 1.0) / (best["odds"] - 1.0)
            record = row.to_dict()
            record.update({
                "bet_side": best["side"],
                "bet_odds": best["odds"],
                "bet_fair_prob": best["fair_prob"],
                "bet_edge": best["edge"],
                "kelly_full": kelly_full,
                "won": row["FTR"] == best["ftr_code"],
            })
            records.append(record)
    return pd.DataFrame(records)


def load_eval_df_mls() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / LEAGUE_KEY / "model_predictions_oos_walkforward_mls.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.models.backtest_mls' primero."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    return df.loc[has_blend].copy()


def _print_breakdown_mls(bets: pd.DataFrame, group_col: str, label: str):
    print(f"\n--- Desglose por {label} ---")
    grouped = bets.groupby(group_col).agg(
        n_apuestas=("won", "count"),
        win_rate=("won", "mean"),
        profit_total=("profit", "sum"),
        stake_total=("stake", "sum"),
    )
    grouped["roi"] = grouped["profit_total"] / grouped["stake_total"]
    print(grouped.round(4).to_string())


def run() -> None:
    print(f"\n=== {LEAGUE_KEY} ===")
    try:
        df_eval = load_eval_df_mls()
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    print(f"Partidos con blend disponible (cuota de cierre presente): {len(df_eval)}")

    bets = _select_bets_mls(df_eval, min_edge_threshold=MIN_EDGE_THRESHOLD, max_odds=MAX_ODDS)
    n_bets = len(bets)
    n_skipped = len(df_eval) - n_bets
    print(f"Partidos con edge > {MIN_EDGE_THRESHOLD:.0%}: {n_bets} apostados, {n_skipped} descartados")

    if n_bets == 0:
        print(f"\n[AVISO] MLS: cero apuestas seleccionadas con el umbral actual.")
        return

    bets = _simulate_bankroll(bets, kelly_fraction=KELLY_FRACTION, max_stake_fraction=MAX_STAKE_FRACTION,
                               initial_bankroll=INITIAL_BANKROLL)

    final_bankroll = bets["bankroll_after"].iloc[-1]
    total_staked = bets["stake"].sum()
    total_profit = bets["profit"].sum()
    roi = total_profit / total_staked if total_staked > 0 else float("nan")
    win_rate = bets["won"].mean()
    max_drawdown = bets["drawdown"].max()

    print(f"\n=== Resultado del backtest economico [MLS] (v2, Kelly fraccional {KELLY_FRACTION:.0%}, "
          f"umbral de edge {MIN_EDGE_THRESHOLD:.0%}, tope de cuota {MAX_ODDS}, ejecucion a CIERRE) ===")
    print(f"Apuestas simuladas:                {n_bets}")
    print(f"Bankroll inicial:                  {INITIAL_BANKROLL:.2f}")
    print(f"Bankroll final:                    {final_bankroll:.2f}  ({final_bankroll / INITIAL_BANKROLL:.3f}x)")
    print(f"Total apostado (suma de stakes):   {total_staked:.2f}")
    print(f"Profit total:                      {total_profit:.2f}")
    print(f"ROI (profit / total apostado):     {roi:.2%}")
    print(f"Win rate:                          {win_rate:.2%}")
    print(f"Drawdown maximo:                   {max_drawdown:.2%}")
    print("(Sin CLV: MLS no tiene cuota de apertura, no hay movimiento de linea que medir.)")

    _print_breakdown_mls(bets, "fold_test_season", "temporada")

    bets["odds_bin"] = pd.cut(
        bets["bet_odds"], bins=[1.0, 1.5, 2.0, 3.0, 5.0, np.inf],
        labels=["1.00-1.50", "1.50-2.00", "2.00-3.00", "3.00-5.00", "5.00+"],
    )
    _print_breakdown_mls(bets, "odds_bin", "rango de cuota apostada")
    _print_breakdown_mls(bets, "bet_side", "lado apostado (H/D/A)")

    out_path = PROCESSED_DATA_DIR / LEAGUE_KEY / "economic_backtest_mls_bets.csv"
    bets.to_csv(out_path, index=False)
    print(f"\nGuardado detalle apuesta por apuesta -> {out_path}")

    log_run(
        script="economic_backtest_mls.py",
        model_name="poisson",
        model_version="v2_mls",
        data_paths=[PROCESSED_DATA_DIR / LEAGUE_KEY / "model_predictions_oos_walkforward_mls.csv"],
        features="[MLS] Analisis economico sobre predicciones de backtest_mls.py (v2) -- no entrena "
                  "modelo nuevo.",
        hyperparameters={
            "league_key": LEAGUE_KEY,
            "kelly_fraction": KELLY_FRACTION,
            "min_edge_threshold": MIN_EDGE_THRESHOLD,
            "max_stake_fraction": MAX_STAKE_FRACTION,
            "max_odds": MAX_ODDS,
            "initial_bankroll": INITIAL_BANKROLL,
            "execution_price": "Pinnacle CIERRE (PSCH/PSCD/PSCA) -- MLS no tiene apertura",
        },
        metrics={
            "n_bets": n_bets,
            "n_skipped": n_skipped,
            "final_bankroll_multiple": final_bankroll / INITIAL_BANKROLL,
            "roi": roi,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
        },
        predictions_path=out_path,
        notes="[MLS] Fase 8b -- backtest economico sin CLV disponible (solo hay cuota de cierre). "
              "Hiperparametros de staking heredados de EPL, sin validar todavia para MLS.",
    )


if __name__ == "__main__":
    run()