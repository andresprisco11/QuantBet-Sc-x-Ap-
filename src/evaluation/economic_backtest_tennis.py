"""
Escalamiento a Tenis, paso 5 (tras tennis_logistic_model.py, corrido y
CONFIRMADO 2026-08-19 -- Brier modelo ATP 0.2189/WTA 0.2178, mercado
0.2017/0.2020, blend practicamente igual al mercado en la mayoria de los
años, gap +0.0008/+0.0011 en agregado -- ver roadmap). Primera medicion
de si ese blend genera ROI real, no solo mejor Brier score. Misma
distincion que ya demostro su valor en fútbol: Brier mide calibracion,
no rentabilidad.

SIN CLV, IGUAL QUE MLS (Fase 8b) -- distinto de las 4 ligas europeas: el
dataset de tenis solo trae UN precio por partido (el mas reciente antes
de que arranque el partido, confirmado en clean_tennis_data.py), no
apertura+cierre. No hay forma de medir "le ganamos al movimiento de la
linea" -- el unico KPI economico disponible es ROI directo contra ese
precio unico. Es un estandar MAS DURO que el de las 4 ligas europeas: ahi
CLV positivo + ROI negativo ya fue una señal util de diagnostico
(winner's curse de seleccion, ver Fase 8) -- esa señal no existe acá.

REGLA DE STAKING DE ESTA PRIMERA CORRIDA: min_edge=8%, kelly_fraction=10%,
max_odds=3.0 -- LOS MISMOS VALORES QUE SE USAN EN FÚTBOL. Esto es
deliberadamente un PUNTO DE PARTIDA, NO una regla tuneada para tenis --
misma disciplina que ya aplico el proyecto en Fase 8 ("no asumir que una
regla tuneada en una liga/deporte transfiere a otro sin probarlo"). Si
esta primera corrida muestra algo prometedor, el paso siguiente es un
barrido de staking especifico de tenis (analogo a tune_staking_rules.py /
tune_staking_rules_mls.py), no adoptar esta regla a ciegas.

MAX_STAKE_FRACTION acá es un valor propio y conservador (5%), NO
importado de economic_backtest.py -- no se asume que el cap usado en
fútbol/MLS aplique igual a un mercado nuevo del que todavía no se conoce
el perfil de riesgo real.

Selección de apuestas: para cada partido, se evalúa el edge de apostarle
a Player1 O a Player2 (blend_prob del lado x cuota real de Pinnacle - 1,
la cuota CON margen, no la no-vig -- el pago real de una apuesta se
calcula con la cuota que de verdad ofrece el libro) y se elige el lado
con mejor edge, igual que _select_bets() en economic_backtest.py de
fútbol. Solo se consideran partidos con cuota de Pinnacle disponible en
AMBOS jugadores (si falta alguna, no se puede evaluar edge con
precisión).

Requiere haber corrido antes:
    python -m src.models.tennis_logistic_model --tours ATP,WTA
    (version que ya incluye Player1_PS_Odds/Player2_PS_Odds en la salida)

Salida: data/runs/economic_backtest_tennis_<tour>_bets.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

MIN_EDGE_THRESHOLD = 0.08
KELLY_FRACTION = 0.10
MAX_ODDS = 3.0
MAX_STAKE_FRACTION = 0.05  # valor propio conservador -- ver docstring, no importado de economic_backtest.py
INITIAL_BANKROLL = 1000.0


def load_predictions(tour: str) -> pd.DataFrame:
    """Carga predictions_v1.csv de un tour -- factorizado para que
    tune_staking_rules_tennis.py pueda reutilizarlo sin reimplementar la
    carga (misma logica de load_eval_df() en economic_backtest.py de futbol,
    que tune_edge_ceiling_by_league.py ya importa directo)."""
    in_path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "predictions_v1.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe {in_path}. Corre 'python -m src.models.tennis_logistic_model --tours {tour}' primero."
        )
    df = pd.read_csv(in_path, parse_dates=["Date"])
    if "Player1_PS_Odds" not in df.columns or "Player2_PS_Odds" not in df.columns:
        raise ValueError(
            f"{in_path} no tiene las columnas Player1_PS_Odds/Player2_PS_Odds -- "
            f"corre de nuevo tennis_logistic_model.py con la version que las incluye en la salida."
        )
    return df


def _select_bets(df: pd.DataFrame, min_edge_threshold: float = MIN_EDGE_THRESHOLD,
                  max_odds: float = MAX_ODDS) -> pd.DataFrame:
    """Parametrizada (min_edge_threshold/max_odds) para que
    tune_staking_rules_tennis.py pueda barrer valores distintos sin
    reimplementar la logica de seleccion -- mismo patron que
    _select_bets(..., max_edge_ceiling=...) en economic_backtest.py de futbol."""
    rows = []
    for row in df.itertuples(index=False):
        odds_p1 = getattr(row, "Player1_PS_Odds", np.nan)
        odds_p2 = getattr(row, "Player2_PS_Odds", np.nan)
        if pd.isna(odds_p1) or pd.isna(odds_p2):
            continue

        blend_p1 = row.blend_prob_player1
        blend_p2 = 1.0 - blend_p1

        edge_p1 = blend_p1 * odds_p1 - 1.0
        edge_p2 = blend_p2 * odds_p2 - 1.0

        if edge_p1 >= edge_p2:
            side, edge, odds, prob, won = "Player1", edge_p1, odds_p1, blend_p1, bool(row.Player1_Won)
        else:
            side, edge, odds, prob, won = "Player2", edge_p2, odds_p2, blend_p2, not bool(row.Player1_Won)

        if edge < min_edge_threshold or odds > max_odds:
            continue

        rows.append({
            "Date": row.Date, "Tournament": row.Tournament, "Surface": row.Surface,
            "Player1": row.Player1, "Player2": row.Player2, "side": side,
            "odds": odds, "blend_prob": prob, "edge": edge, "won": won,
        })

    return pd.DataFrame(rows)


def _simulate_bankroll(bets: pd.DataFrame, kelly_fraction: float = KELLY_FRACTION,
                        max_stake_fraction: float = MAX_STAKE_FRACTION,
                        initial_bankroll: float = INITIAL_BANKROLL) -> pd.DataFrame:
    """Kelly fraccional real -- misma logica que _simulate_bankroll() en
    economic_backtest.py de futbol: apuestas ejecutadas en orden
    cronologico, cada stake calculado sobre el bankroll DESPUES de las
    ganancias/perdidas de apuestas anteriores (interes compuesto real),
    con un cap de stake maximo por apuesta. Parametrizada por la misma
    razon que _select_bets()."""
    bets = bets.sort_values("Date").reset_index(drop=True)
    bankroll = initial_bankroll
    peak = initial_bankroll
    stakes, profits, bankrolls_after, drawdowns = [], [], [], []

    for row in bets.itertuples(index=False):
        b = row.odds - 1.0  # ganancia neta por unidad apostada si gana
        p = row.blend_prob
        q = 1.0 - p
        kelly_full = (b * p - q) / b if b > 0 else 0.0
        kelly_stake_fraction = min(max(0.0, kelly_full) * kelly_fraction, max_stake_fraction)

        stake = bankroll * kelly_stake_fraction
        profit = stake * b if row.won else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0

        stakes.append(stake)
        profits.append(profit)
        bankrolls_after.append(bankroll)
        drawdowns.append(drawdown)

    bets = bets.copy()
    bets["stake"] = stakes
    bets["profit"] = profits
    bets["bankroll_after"] = bankrolls_after
    bets["drawdown"] = drawdowns
    return bets


def run(tour: str) -> None:
    print(f"\n=== {tour.upper()} ===")
    df = load_predictions(tour)

    bets = _select_bets(df)
    n_bets = len(bets)
    print(f"  Partidos evaluados: {len(df)}  |  Apuestas seleccionadas "
          f"(edge>={MIN_EDGE_THRESHOLD:.0%}, odds<={MAX_ODDS}): {n_bets}")

    if n_bets == 0:
        print("  [AVISO] cero apuestas seleccionadas con esta regla -- no se puede simular bankroll.")
        return

    bets = _simulate_bankroll(bets)

    total_staked = bets["stake"].sum()
    total_profit = bets["profit"].sum()
    roi = total_profit / total_staked if total_staked > 0 else float("nan")
    win_rate = bets["won"].mean()
    final_bankroll = bets["bankroll_after"].iloc[-1]
    max_drawdown = bets["drawdown"].max()

    print(f"  ROI: {roi:+.2%}  |  Win rate: {win_rate:.2%}  |  Bankroll final: {final_bankroll:.2f} "
          f"(inicial {INITIAL_BANKROLL:.0f})  |  Drawdown máximo: {max_drawdown:.2%}")
    side_dist = bets["side"].value_counts(normalize=True)
    print(f"  Distribución de lado apostado: Player1 {side_dist.get('Player1', 0):.1%}, "
          f"Player2 {side_dist.get('Player2', 0):.1%}")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / f"economic_backtest_tennis_{tour.lower()}_bets.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(out_path, index=False)
    print(f"  Guardado -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest económico de tenis -- ROI directo (sin CLV) con la regla de staking base de fútbol como punto de partida."
    )
    parser.add_argument("--tours", type=str, default="ATP,WTA", help="Tours a correr, separados por coma (default: ATP,WTA).")
    args = parser.parse_args()

    for tour in args.tours.split(","):
        try:
            run(tour)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {tour}: {e}")