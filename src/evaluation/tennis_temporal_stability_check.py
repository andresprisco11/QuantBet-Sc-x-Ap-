"""
Escalamiento a Tenis, paso 8 (tras confirmar con Elo la primera meseta de
ROI positivo real del proyecto en tenis -- ATP min_edge=1%/max_odds=2.5,
n=341, ROI ~6% consistente en 5 fracciones de Kelly, ver roadmap
2026-08-19). Antes de tratarlo como "candidato serio" hay que hacerle
exactamente el mismo chequeo que ya se le hizo a EPL y a Serie A/Bundesliga
en fútbol (epl_temporal_stability_check.py / temporal_stability_check_all_leagues.py,
Fase 8): un ROI agregado sobre 9 años OOS puede esconder que en realidad
1-2 años buenos tapan el resto -- mismo principio ya documentado en el
roadmap ("un promedio agregado puede esconder una tendencia temporal
real"). Esto se prueba acá, con el mismo estandar de honestidad.

Reutiliza _select_bets()/_simulate_bankroll()/load_predictions() de
economic_backtest_tennis.py -- no se reimplementa la logica de seleccion
ni de staking, mismo criterio del proyecto en cada script de tuning/
diagnostico de esta fase.

Candidatos evaluados (los que salieron del barrido, tune_staking_rules_tennis.py):
- ATP: min_edge=1%, max_odds=2.5, kelly=10% (valor medio de la meseta --
  el barrido mostro ROI ~6% parecido en las 5 fracciones de Kelly
  probadas, 10% es el default que ya usa el resto del proyecto).
- WTA: min_edge=8%, max_odds=6.0, kelly=10% -- ADVERTENCIA EXPLICITA: acá
  n=30 sobre 9 años es ~3 apuestas por año, insuficiente para una lectura
  por-año seria. Se corre igual por completitud, pero el resultado se
  marca como no interpretable temporalmente, no se lee como si lo fuera.

Salida: data/runs/tennis_temporal_stability_check_<tour>.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.evaluation.economic_backtest_tennis import _select_bets, _simulate_bankroll, load_predictions, INITIAL_BANKROLL

CANDIDATES = {
    "ATP": {"min_edge": 0.01, "max_odds": 2.5, "kelly_fraction": 0.10},
    "WTA": {"min_edge": 0.08, "max_odds": 6.0, "kelly_fraction": 0.10},
}

MIN_BETS_PER_YEAR_INTERPRETABLE = 15  # por debajo de esto, un ROI anual individual es ruido puro, se marca


def _year_summary(bets: pd.DataFrame) -> pd.DataFrame:
    bets = bets.copy()
    bets["year"] = bets["Date"].dt.year
    rows = []
    for year, g in bets.groupby("year"):
        total_staked = g["stake"].sum()
        total_profit = g["profit"].sum()
        roi = total_profit / total_staked if total_staked > 0 else float("nan")
        rows.append({
            "year": year, "n_bets": len(g), "roi": roi, "win_rate": g["won"].mean(),
            "interpretable": len(g) >= MIN_BETS_PER_YEAR_INTERPRETABLE,
        })
    return pd.DataFrame(rows).sort_values("year")


def run(tour: str) -> None:
    print(f"\n=== {tour.upper()} ===")
    candidate = CANDIDATES[tour]
    try:
        df = load_predictions(tour)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    bets = _select_bets(df, min_edge_threshold=candidate["min_edge"], max_odds=candidate["max_odds"])
    n_bets = len(bets)
    print(f"Candidato: min_edge={candidate['min_edge']:.0%}, max_odds={candidate['max_odds']}, "
          f"kelly={candidate['kelly_fraction']:.0%}  |  Apuestas seleccionadas: {n_bets}")

    if n_bets == 0:
        print("  [AVISO] cero apuestas -- no se puede evaluar estabilidad temporal.")
        return

    bets = _simulate_bankroll(bets, kelly_fraction=candidate["kelly_fraction"], initial_bankroll=INITIAL_BANKROLL)
    year_df = _year_summary(bets)

    n_years = len(year_df)
    if n_bets / max(n_years, 1) < MIN_BETS_PER_YEAR_INTERPRETABLE:
        print(f"  [AVISO] {n_bets} apuestas repartidas en {n_years} años (~{n_bets/n_years:.0f}/año) -- "
              f"por debajo de {MIN_BETS_PER_YEAR_INTERPRETABLE}/año, el desglose por año de acá abajo es "
              f"referencial, NO una lectura confiable de estabilidad temporal. Se muestra por completitud.")

    print("\nROI y win rate por año:")
    with pd.option_context("display.width", 120):
        print(year_df.to_string(index=False, formatters={
            "roi": "{:+.2%}".format, "win_rate": "{:.2%}".format,
        }))

    n_positive = (year_df["roi"] > 0).sum()
    n_negative = (year_df["roi"] <= 0).sum()
    print(f"\nAños con ROI positivo: {n_positive}/{n_years}  |  Años con ROI negativo o cero: {n_negative}/{n_years}")

    # -- primera mitad vs segunda mitad, mismo chequeo que epl_temporal_stability_check.py en futbol --
    years_sorted = sorted(year_df["year"].unique())
    mid = len(years_sorted) // 2
    first_half_years = set(years_sorted[:mid]) if mid > 0 else set()
    second_half_years = set(years_sorted[mid:])
    bets["year"] = bets["Date"].dt.year
    first_half = bets[bets["year"].isin(first_half_years)]
    second_half = bets[bets["year"].isin(second_half_years)]

    def _half_roi(g):
        staked = g["stake"].sum()
        return (g["profit"].sum() / staked) if staked > 0 else float("nan")

    if first_half_years:
        print(f"\nPrimera mitad ({min(first_half_years)}-{max(first_half_years)}, n={len(first_half)}): "
              f"ROI {_half_roi(first_half):+.2%}")
    print(f"Segunda mitad ({min(second_half_years)}-{max(second_half_years)}, n={len(second_half)}): "
          f"ROI {_half_roi(second_half):+.2%}")

    final_bankroll = bets["bankroll_after"].iloc[-1]
    max_drawdown = bets["drawdown"].max()
    print(f"\nBankroll final: {final_bankroll:.2f} (inicial {INITIAL_BANKROLL:.0f})  |  Drawdown máximo: {max_drawdown:.2%}")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / f"tennis_temporal_stability_check_{tour.lower()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    year_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    for tour in CANDIDATES:
        run(tour)