"""
Fase 8, "Proximos pasos" punto 2, hipotesis (b): el ROI positivo de EPL,
es señal real o ruido de muestra que con mas temporadas terminaria pareciendose
al de los otros 3 mercados europeos (todos negativos o en punto muerto)?

Con solo 6-7 temporadas evaluadas por liga, "+2.11%" o "+4.49%" de ROI
agregado puede estar escondiendo que en realidad 1-2 temporadas atipicas
cargan con todo el resultado positivo, y el resto son planas o negativas
-- exactamente el mismo tipo de chequeo que ya se le hizo a la hipotesis
de paridad competitiva (competitive_balance_analysis.py): no asumir,
descomponer y mirar.

Este script NO entrena ni simula nada nuevo -- reutiliza las predicciones
OOS ya generadas por backtest_v4.py para EPL y aplica la misma logica de
seleccion de apuestas / Kelly fraccional que economic_backtest.py, mirando
dos cortes:
  1. ROI temporada por temporada (el mas fino posible con 6-7 temporadas).
  2. ROI de la primera mitad de temporadas vs. la segunda mitad -- si el
     ROI positivo se sostiene en ambas mitades, es mas dificil de explicar
     como ruido puro; si esta todo concentrado en una mitad, la hipotesis
     de ruido de muestra gana fuerza.

Se corre con las DOS reglas de staking ya documentadas en el roadmap para
EPL (la original min_edge=8%/kelly=10%/max_odds=3.0, y la tuneada por
tune_staking_rules.py: min_edge=12%/kelly=25%/max_odds=3.0), para ver si
la conclusion depende de que regla se use.

IMPORTANTE sobre el nombre del archivo de entrada: se asume, por
convencion con backtest_mls.py -> 'model_predictions_oos_walkforward_mls.csv',
que backtest_v4.py guarda 'model_predictions_oos_walkforward_v4.csv' en
data/processed/EPL/. Si ese no es el nombre real, el script lista los CSV
que SI encuentra en esa carpeta en vez de fallar en silencio -- avisale al
CTO cual es el nombre correcto y se ajusta.

Reutiliza _simulate_bankroll/INITIAL_BANKROLL de economic_backtest.py (pura
mecanica de staking, confirmado que existen porque economic_backtest_mls.py
ya los importa de ahi). La seleccion de apuestas (_select_bets_epl) es
propia de este script -- usa cuota de APERTURA de Pinnacle (PSH/PSD/PSA),
igual que economic_backtest.py hace para las 4 ligas europeas.

Salida: data/runs/epl_temporal_stability_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.evaluation.economic_backtest import _simulate_bankroll, INITIAL_BANKROLL

LEAGUE_KEY = "EPL"
MAX_STAKE_FRACTION = 0.05

STAKING_RULES = [
    {"label": "original (min_edge=8%, kelly=10%, max_odds=3.0)",
     "min_edge_threshold": 0.08, "kelly_fraction": 0.10, "max_odds": 3.0},
    {"label": "tuneada (min_edge=12%, kelly=25%, max_odds=3.0)",
     "min_edge_threshold": 0.12, "kelly_fraction": 0.25, "max_odds": 3.0},
]

# Apertura de Pinnacle -- misma logica que economic_backtest.py para las 4 ligas europeas.
SIDES = [
    ("home", "PSH", "blend_prob_home", "H"),
    ("draw", "PSD", "blend_prob_draw", "D"),
    ("away", "PSA", "blend_prob_away", "A"),
]


def _select_bets_epl(df: pd.DataFrame, min_edge_threshold: float, max_odds: float) -> pd.DataFrame:
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


def _load_predictions() -> pd.DataFrame:
    league_dir = PROCESSED_DATA_DIR / LEAGUE_KEY
    path = league_dir / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        print(f"[ERROR] No existe {path}.")
        if league_dir.exists():
            candidates = sorted(league_dir.glob("*.csv"))
            print(f"CSVs encontrados en {league_dir}:")
            for c in candidates:
                print(f"  - {c.name}")
        else:
            print(f"[ERROR] Ni siquiera existe la carpeta {league_dir}.")
        print("Decile al CTO cual es el nombre real del archivo con las predicciones OOS "
              "de backtest_v4.py para EPL y se ajusta este script.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_blend].copy()
    return df.sort_values("Date").reset_index(drop=True)


def _roi_summary(bets: pd.DataFrame) -> dict:
    total_staked = bets["stake"].sum()
    total_profit = bets["profit"].sum()
    return {
        "n_bets": len(bets),
        "win_rate": bets["won"].mean(),
        "roi": total_profit / total_staked if total_staked > 0 else float("nan"),
    }


def run() -> None:
    print(f"\n=== {LEAGUE_KEY} -- chequeo de estabilidad temporal (hipotesis 'b': ruido de muestra) ===")
    df_eval = _load_predictions()
    if df_eval.empty:
        return

    seasons = sorted(df_eval["fold_test_season"].astype(str).unique(), key=lambda s: s)
    n_seasons = len(seasons)
    half = n_seasons // 2
    first_half_seasons = set(seasons[:half]) if half > 0 else set()
    second_half_seasons = set(seasons[half:])
    print(f"Temporadas OOS evaluadas: {seasons}")
    print(f"Primera mitad: {sorted(first_half_seasons)} | Segunda mitad: {sorted(second_half_seasons)}\n")

    all_rows = []
    for rule in STAKING_RULES:
        print(f"\n--- Regla: {rule['label']} ---")
        bets = _select_bets_epl(df_eval, rule["min_edge_threshold"], rule["max_odds"])
        if bets.empty:
            print("[AVISO] Cero apuestas seleccionadas con esta regla.")
            continue

        bets = bets.sort_values("Date").reset_index(drop=True)
        bets = _simulate_bankroll(bets, kelly_fraction=rule["kelly_fraction"],
                                   max_stake_fraction=MAX_STAKE_FRACTION,
                                   initial_bankroll=INITIAL_BANKROLL)
        bets["fold_test_season"] = bets["fold_test_season"].astype(str)

        print("\nROI por temporada:")
        season_rows = []
        for season in seasons:
            season_bets = bets[bets["fold_test_season"] == season]
            if season_bets.empty:
                print(f"  {season}: sin apuestas")
                continue
            summary = _roi_summary(season_bets)
            print(f"  {season}: n={summary['n_bets']:4d}  win_rate={summary['win_rate']:.2%}  "
                  f"ROI={summary['roi']:+.2%}")
            season_rows.append({"regla": rule["label"], "temporada": season, **summary})
        all_rows.extend(season_rows)

        first_half_bets = bets[bets["fold_test_season"].isin(first_half_seasons)]
        second_half_bets = bets[bets["fold_test_season"].isin(second_half_seasons)]
        print("\nROI agregado por mitad:")
        if not first_half_bets.empty:
            s1 = _roi_summary(first_half_bets)
            print(f"  Primera mitad {sorted(first_half_seasons)}: n={s1['n_bets']:4d}  "
                  f"win_rate={s1['win_rate']:.2%}  ROI={s1['roi']:+.2%}")
            all_rows.append({"regla": rule["label"], "temporada": "PRIMERA_MITAD", **s1})
        if not second_half_bets.empty:
            s2 = _roi_summary(second_half_bets)
            print(f"  Segunda mitad {sorted(second_half_seasons)}: n={s2['n_bets']:4d}  "
                  f"win_rate={s2['win_rate']:.2%}  ROI={s2['roi']:+.2%}")
            all_rows.append({"regla": rule["label"], "temporada": "SEGUNDA_MITAD", **s2})

    if not all_rows:
        print("\n[AVISO] Ninguna regla produjo apuestas -- nada que guardar.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "epl_temporal_stability_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio, no un modelo nuevo) "
          "-- mismo criterio que competitive_balance_analysis.py.")


if __name__ == "__main__":
    run()