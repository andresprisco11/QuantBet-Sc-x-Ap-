"""
Fase 3.5: capa de evaluacion ECONOMICA sobre las predicciones de v4 (el
mejor modelo hasta ahora), en vez de seguir agregando features nuevas al
Poisson por Brier score. Justificacion (ver roadmap, "Principios de
diseno" y "Proximos pasos"): dos intentos seguidos de mejorar v4 (v5, v6)
fallaron, y el Brier score OOS mide calibracion probabilistica, NO
explotabilidad economica -- pueden coexistir un modelo bien calibrado sin
edge real, y un modelo con edge real en un segmento aunque el agregado no
le gane al mercado. Esta capa responde la pregunta que el Brier score no
puede responder: si hubieras apostado con este modelo, historicamente,
te hubiera dado plata.

ACTUALIZACION (post-diagnostico): la primera corrida de este script (con
MIN_EDGE_THRESHOLD=2%, KELLY_FRACTION=25%, sin tope de cuota) dio ROI de
-8.94% y drawdown de 92.37%. calibration_analysis.py y
selection_bias_check.py encontraron la causa raiz: NO es que el modelo
este mal calibrado en general (ECE ~1.3-1.4%, razonable) -- es que el
proceso de elegir el resultado con mayor edge entre los 3 posibles de
cada partido selecciona sistematicamente sobreestimaciones de ruido
("winner's curse"), con una brecha de calibracion mucho mas grande en las
apuestas SELECCIONADAS que en la poblacion general, en los 5 rangos de
cuota medidos. Por eso las funciones de este script ahora aceptan
min_edge_threshold/kelly_fraction/max_odds como PARAMETROS (antes eran
constantes fijas) -- para poder barrerlos en tune_staking_rules.py y
elegir la combinacion con evidencia, no a ojo.

RESULTADO DEL BARRIDO (tune_staking_rules.py, 24 combinaciones, ver
data/runs/staking_sweep_v4.csv para la grilla completa): max_odds=3.0
le gano a 5.0 y a "sin tope" en TODAS las combinaciones sin excepcion
(coincide exactamente con donde selection_bias_check.py encontro el
sesgo de seleccion mas severo). Dentro de max_odds=3.0, kelly_fraction=
0.10 dio ROI positivo en los 4 umbrales de edge probados (zona robusta,
no un pico aislado); kelly_fraction=0.25 solo funciono en el umbral mas
exigente. Se eligio min_edge_threshold=0.08 como punto medio entre
confiabilidad estadistica (256 apuestas) y rendimiento (ROI +2.11%,
drawdown ~19.5%). Estos tres valores son ahora el default de run().

IMPORTANTE (Fase 8, 2026-08-18): esos tres valores se tunearon
EXCLUSIVAMENTE sobre EPL. Al parametrizar este script por liga, se
mantienen como default global -- pero eso es una hipotesis a validar,
NO un hecho: no hay ninguna razon matematica para asumir que la regla
optima de staking de EPL transfiere igual a Serie A, Bundesliga o La
Liga (mercados con distinta distribucion de cuotas, distinta paridad,
distinto Brier de mercado -- ver roadmap, Fase 8). tune_staking_rules.py
tambien se parametrizo por liga (ver ese script) especificamente para
poder chequear esto con evidencia antes de asumirlo.

METODOLOGIA (sin cambios, ver el resto del docstring de la version
anterior para el detalle completo):
1. Fuente: 'model_predictions_oos_walkforward_v4.csv' (backtest_v4.py),
   por liga.
2. Precio de ejecucion: cuotas DECIMALES de apertura de Pinnacle
   (PSH/PSD/PSA).
3. Probabilidad "justa": blend_prob_* (Benter Boost del walk-forward v4).
4. Edge y sizing: formula estandar de Kelly Criterion, con fraccion de
   Kelly conservadora y tope duro de stake maximo por apuesta.
5. Un solo lado por partido: el de mayor edge positivo, si supera
   min_edge_threshold (y si max_odds no es None, solo se consideran
   candidatos con cuota <= max_odds).
6. CLV calculado explicitamente: pinnacle_close_prob_lado -
   pinnacle_open_prob_lado del lado apostado.
7. Bankroll compuesto, cronologico, con drawdown maximo medido sobre la
   curva completa.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR
from src.tracking.run_logger import log_run

KELLY_FRACTION = 0.10          # tuneado sobre EPL (ver docstring) -- default global, a validar por liga.
MIN_EDGE_THRESHOLD = 0.08      # tuneado sobre EPL (ver docstring) -- default global, a validar por liga.
MAX_STAKE_FRACTION = 0.05      # tope duro: nunca mas del 5% del bankroll actual en una sola apuesta.
MAX_ODDS = 3.0                 # tuneado sobre EPL (ver docstring) -- default global, a validar por liga.
INITIAL_BANKROLL = 1000.0      # unidades arbitrarias -- lo que importa es el multiplo final, no la moneda.

SIDES = [
    ("home", "PSH", "blend_prob_home", "pinnacle_open_prob_home", "pinnacle_close_prob_home", "H"),
    ("draw", "PSD", "blend_prob_draw", "pinnacle_open_prob_draw", "pinnacle_close_prob_draw", "D"),
    ("away", "PSA", "blend_prob_away", "pinnacle_open_prob_away", "pinnacle_close_prob_away", "A"),
]


def _select_bets(df: pd.DataFrame, min_edge_threshold: float = MIN_EDGE_THRESHOLD,
                  max_odds: float = MAX_ODDS) -> pd.DataFrame:
    """
    Para cada partido, calcula el edge de los 3 resultados posibles (solo
    considerando los que tengan cuota <= max_odds, si se especifica) y
    selecciona el de mayor edge SI supera min_edge_threshold. Partidos sin
    ningun edge suficiente quedan fuera del backtest economico.
    """
    records = []
    for _, row in df.iterrows():
        best = None
        for side_name, odds_col, prob_col, open_prob_col, close_prob_col, ftr_code in SIDES:
            odds = row[odds_col]
            fair_prob = row[prob_col]
            if pd.isna(odds) or pd.isna(fair_prob) or odds <= 1.0:
                continue
            if max_odds is not None and odds > max_odds:
                continue
            edge = fair_prob * odds - 1.0
            if best is None or edge > best["edge"]:
                best = {
                    "side": side_name,
                    "odds": odds,
                    "fair_prob": fair_prob,
                    "edge": edge,
                    "open_prob": row[open_prob_col],
                    "close_prob": row[close_prob_col],
                    "ftr_code": ftr_code,
                }
        if best is not None and best["edge"] > min_edge_threshold:
            kelly_full = (best["fair_prob"] * best["odds"] - 1.0) / (best["odds"] - 1.0)
            record = row.to_dict()
            record.update({
                "bet_side": best["side"],
                "bet_odds": best["odds"],
                "bet_fair_prob": best["fair_prob"],
                "bet_edge": best["edge"],
                "kelly_full": kelly_full,
                "clv": best["close_prob"] - best["open_prob"],
                "won": row["FTR"] == best["ftr_code"],
            })
            records.append(record)
    return pd.DataFrame(records)


def _simulate_bankroll(bets: pd.DataFrame, kelly_fraction: float = KELLY_FRACTION,
                        max_stake_fraction: float = MAX_STAKE_FRACTION,
                        initial_bankroll: float = INITIAL_BANKROLL) -> pd.DataFrame:
    """
    Recorre las apuestas EN ORDEN CRONOLOGICO aplicando Kelly fraccional
    con tope duro, compone el bankroll apuesta a apuesta, y devuelve el
    dataframe con columnas de stake/bankroll/drawdown agregadas.
    """
    if bets.empty:
        return bets

    bets = bets.sort_values("Date").reset_index(drop=True)
    bankroll = initial_bankroll
    peak = initial_bankroll
    stakes, bankrolls, drawdowns, profits = [], [], [], []

    for _, bet in bets.iterrows():
        kelly_stake_frac = min(max(bet["kelly_full"] * kelly_fraction, 0.0), max_stake_fraction)
        stake = bankroll * kelly_stake_frac
        if bet["won"]:
            profit = stake * (bet["bet_odds"] - 1.0)
        else:
            profit = -stake
        bankroll += profit
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0

        stakes.append(stake)
        bankrolls.append(bankroll)
        drawdowns.append(drawdown)
        profits.append(profit)

    bets = bets.copy()
    bets["stake"] = stakes
    bets["bankroll_after"] = bankrolls
    bets["drawdown"] = drawdowns
    bets["profit"] = profits
    return bets


def load_eval_df(league_key: str) -> pd.DataFrame:
    """Carga model_predictions_oos_walkforward_v4.csv de UNA liga y filtra a partidos con blend disponible.
    Reutilizado por tune_staking_rules.py para no duplicar esta parte."""
    path = PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.models.backtest_v4' primero -- este script solo "
            f"analiza predicciones ya generadas, no entrena nada."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    return df.loc[has_blend].copy()


def _print_breakdown(bets: pd.DataFrame, group_col: str, label: str):
    print(f"\n--- Desglose por {label} ---")
    grouped = bets.groupby(group_col).agg(
        n_apuestas=("won", "count"),
        win_rate=("won", "mean"),
        profit_total=("profit", "sum"),
        stake_total=("stake", "sum"),
        clv_promedio=("clv", "mean"),
        clv_positivo_pct=("clv", lambda s: (s > 0).mean()),
    )
    grouped["roi"] = grouped["profit_total"] / grouped["stake_total"]
    print(grouped.round(4).to_string())


def run(league_key: str) -> None:
    print(f"\n=== {league_key} ===")
    try:
        df_eval = load_eval_df(league_key)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    print(f"Partidos con blend disponible (cuota de cierre presente): {len(df_eval)}")

    bets = _select_bets(df_eval, min_edge_threshold=MIN_EDGE_THRESHOLD, max_odds=MAX_ODDS)
    n_bets = len(bets)
    n_skipped = len(df_eval) - n_bets
    print(f"Partidos con edge > {MIN_EDGE_THRESHOLD:.0%}: {n_bets} apostados, {n_skipped} descartados (sin valor suficiente)")

    if n_bets == 0:
        print(f"\n[AVISO] {league_key}: cero apuestas seleccionadas con el umbral actual -- no hay backtest "
              f"economico que correr. Prueba bajando MIN_EDGE_THRESHOLD si esto es inesperado.")
        return

    bets = _simulate_bankroll(bets, kelly_fraction=KELLY_FRACTION, max_stake_fraction=MAX_STAKE_FRACTION,
                               initial_bankroll=INITIAL_BANKROLL)

    final_bankroll = bets["bankroll_after"].iloc[-1]
    total_staked = bets["stake"].sum()
    total_profit = bets["profit"].sum()
    roi = total_profit / total_staked if total_staked > 0 else float("nan")
    win_rate = bets["won"].mean()
    max_drawdown = bets["drawdown"].max()
    avg_clv = bets["clv"].mean()
    clv_positive_pct = (bets["clv"] > 0).mean()

    print(f"\n=== Resultado del backtest economico [{league_key}] (v4, Kelly fraccional {KELLY_FRACTION:.0%}, "
          f"umbral de edge {MIN_EDGE_THRESHOLD:.0%}, tope de cuota {MAX_ODDS if MAX_ODDS else 'sin tope'}) ===")
    print(f"Apuestas simuladas:                {n_bets}")
    print(f"Bankroll inicial:                  {INITIAL_BANKROLL:.2f}")
    print(f"Bankroll final:                    {final_bankroll:.2f}  ({final_bankroll / INITIAL_BANKROLL:.3f}x)")
    print(f"Total apostado (suma de stakes):   {total_staked:.2f}")
    print(f"Profit total:                      {total_profit:.2f}")
    print(f"ROI (profit / total apostado):     {roi:.2%}")
    print(f"Win rate:                          {win_rate:.2%}")
    print(f"Drawdown maximo:                   {max_drawdown:.2%}")
    print(f"CLV promedio (lado apostado):      {avg_clv:+.4f}")
    print(f"% de apuestas con CLV positivo:    {clv_positive_pct:.2%}")

    _print_breakdown(bets, "fold_test_season", "temporada")

    bets["odds_bin"] = pd.cut(
        bets["bet_odds"], bins=[1.0, 1.5, 2.0, 3.0, 5.0, np.inf],
        labels=["1.00-1.50", "1.50-2.00", "2.00-3.00", "3.00-5.00", "5.00+"],
    )
    _print_breakdown(bets, "odds_bin", "rango de cuota apostada")

    _print_breakdown(bets, "bet_side", "lado apostado (H/D/A)")

    out_path = PROCESSED_DATA_DIR / league_key / "economic_backtest_v4_bets.csv"
    bets.to_csv(out_path, index=False)
    print(f"\nGuardado detalle apuesta por apuesta -> {out_path}")

    log_run(
        script="economic_backtest.py",
        model_name="poisson",
        model_version="v4",
        data_paths=[PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v4.csv"],
        features=f"[{league_key}] Analisis economico sobre predicciones ya generadas por backtest_v4.py -- "
                  "no entrena modelo nuevo.",
        hyperparameters={
            "league_key": league_key,
            "kelly_fraction": KELLY_FRACTION,
            "min_edge_threshold": MIN_EDGE_THRESHOLD,
            "max_stake_fraction": MAX_STAKE_FRACTION,
            "max_odds": MAX_ODDS,
            "initial_bankroll": INITIAL_BANKROLL,
            "execution_price": "Pinnacle apertura (PSH/PSD/PSA)",
        },
        metrics={
            "n_bets": n_bets,
            "n_skipped": n_skipped,
            "final_bankroll_multiple": final_bankroll / INITIAL_BANKROLL,
            "roi": roi,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "avg_clv": avg_clv,
            "clv_positive_pct": clv_positive_pct,
        },
        predictions_path=out_path,
        notes=f"[{league_key}] Corrida del framework multi-metrica (CLV + Kelly fraccional + drawdown + "
              "robustez por temporada/rango de cuota/lado) sobre v4. Hiperparametros de staking tuneados "
              "sobre EPL (Fase 3.5) -- todavia no re-validados especificamente para esta liga si no es EPL.",
    )


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)