"""
Fase 10 -- chequeo de estabilidad temporal para la regla de staking de NFL
que salio ganadora en tune_staking_rules_nfl.py: min_edge=8%, max_edge=20%,
kelly=10% -- la version CONSERVADORA del top del barrido (kelly=25% da mas
ROI bruto, +130.76% vs +58.37%, pero con drawdown mucho mas alto, 49.70%
vs 22.30% -- mismo criterio que goberno cada adopcion de staking en este
proyecto: preferir menor drawdown, no mayor ROI bruto).

Mismo patron de disciplina que epl_temporal_stability_check.py/
tennis_temporal_stability_check.py: un ROI positivo en agregado puede
esconder que todo el resultado viene de 1-2 temporadas atipicas -- se
verifica por temporada, primera mitad vs segunda mitad, antes de considerar
esto una señal real.

**Motivo extra de cautela especifico de esta corrida**: el ROI del mejor
candidato del barrido es mucho mas grande que cualquier resultado visto en
todo el proyecto (futbol tope ~4%, tenis ~6% acumulado). Un win rate de
apenas ~52% (cerca del breakeven de -110) generando ese ROI es
matematicamente posible con Kelly compuesto sobre mas de 1,000 apuestas,
pero exige descartar que dependa de un puñado de apuestas de payout enorme
-- se chequea eso tambien aca, no solo la estabilidad por temporada.

Uso: python -m src.evaluation.nfl_temporal_stability_check
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.evaluation.economic_backtest_nfl import _select_bets, _simulate_bankroll, STARTING_BANKROLL

DATA_PATH = PROCESSED_DATA_DIR / "NFL" / "spread_evaluation_v1.csv"

MIN_EDGE = 0.08
MAX_EDGE = 0.20
KELLY_FRACTION = 0.10


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre backtest_nfl_spread.py primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["gameday"] = pd.to_datetime(df["gameday"])

    bets = _select_bets(df, min_edge=MIN_EDGE, max_edge=MAX_EDGE)
    print(f"Regla evaluada: min_edge={MIN_EDGE:.0%}, max_edge={MAX_EDGE:.0%}, kelly={KELLY_FRACTION:.0%}")
    print(f"Apuestas totales: {len(bets)}\n")

    if bets.empty:
        print("[AVISO] Sin apuestas para esta regla -- nada que verificar.")
        return

    seasons = sorted(bets["season"].unique())
    print("Desglose por temporada:")
    rows = []
    for season in seasons:
        season_bets = bets[bets["season"] == season]
        if season_bets.empty:
            continue
        res = _simulate_bankroll(season_bets, kelly_fraction=KELLY_FRACTION, starting_bankroll=STARTING_BANKROLL)
        rows.append({"season": season, "n_bets": res["n_bets"], "win_rate": res["win_rate"],
                      "roi": res["roi"], "max_drawdown": res["max_drawdown"]})
    season_df = pd.DataFrame(rows)
    print(season_df.to_string(index=False, formatters={
        "win_rate": "{:.2%}".format, "roi": "{:+.2%}".format, "max_drawdown": "{:.2%}".format,
    }))

    n_positive = int((season_df["roi"] > 0).sum())
    print(f"\nTemporadas con ROI positivo: {n_positive} de {len(season_df)}.")

    mid = len(seasons) // 2
    first_half_seasons = seasons[:mid]
    second_half_seasons = seasons[mid:]
    first_half_bets = bets[bets["season"].isin(first_half_seasons)]
    second_half_bets = bets[bets["season"].isin(second_half_seasons)]

    if not first_half_bets.empty:
        r1 = _simulate_bankroll(first_half_bets, kelly_fraction=KELLY_FRACTION, starting_bankroll=STARTING_BANKROLL)
        print(f"\nPrimera mitad ({first_half_seasons[0]}-{first_half_seasons[-1]}, n={r1['n_bets']}): "
              f"ROI {r1['roi']:+.2%}, win rate {r1['win_rate']:.2%}, drawdown {r1['max_drawdown']:.2%}")
    if not second_half_bets.empty:
        r2 = _simulate_bankroll(second_half_bets, kelly_fraction=KELLY_FRACTION, starting_bankroll=STARTING_BANKROLL)
        print(f"Segunda mitad ({second_half_seasons[0]}-{second_half_seasons[-1]}, n={r2['n_bets']}): "
              f"ROI {r2['roi']:+.2%}, win rate {r2['win_rate']:.2%}, drawdown {r2['max_drawdown']:.2%}")

    # Chequeo extra: descartar que el resultado dependa de un puñado de
    # apuestas de stake relativo enorme (payout gigante) en vez de señal
    # sostenida across muchas apuestas chicas.
    bets_sorted = bets.copy()
    bets_sorted["approx_stake_fraction"] = (bets_sorted["edge"] / bets_sorted["b"]).clip(lower=0) * KELLY_FRACTION
    top5 = bets_sorted.sort_values("approx_stake_fraction", ascending=False).head(5)
    print("\nTop 5 apuestas por stake relativo (fraccion de Kelly aplicada, antes de compounding) -- "
          "verifica que el resultado no dependa de un puñado de apuestas gigantes:")
    print(top5[["gameday", "season", "side", "edge", "b", "approx_stake_fraction"]].to_string(index=False))


if __name__ == "__main__":
    run()