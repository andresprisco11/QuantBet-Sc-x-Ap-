"""
Fase 8, "Proximos pasos" punto 2, hipotesis (b), extendida a las 4 ligas
europeas -- version generalizada de epl_temporal_stability_check.py.

Motivo: el chequeo de EPL sola (epl_temporal_stability_check.py) dio un
resultado ambiguo -- ni confirma ni refuta que el ROI positivo sea ruido de
muestra (3 de 5 temporadas positivas, las 2 mitades cronologicas positivas
bajo 2 reglas de staking distintas, pero con varianza enorme temporada a
temporada dado el n chico por temporada). La pregunta natural que se abre:
Serie A y Bundesliga (agregado negativo, -7.81%/-10.91% con la regla
original) tambien tienen temporadas buenas escondidas dentro de un
promedio malo? O son negativas de punta a punta, sin excepcion?

Si el patron de EPL (mezcla de temporadas buenas y malas, promedio positivo)
se repite en las otras 3 pero con promedio negativo, la lectura mas
razonable es "las 4 ligas tienen ruido temporada a temporada similar, EPL
solo tuvo mas suerte en el promedio de las temporadas disponibles" --
apoya (b). Si en cambio Serie A/Bundesliga son negativas en CASI TODAS sus
temporadas sin excepcion (a diferencia de EPL), la lectura es que hay algo
estructuralmente distinto en esas ligas, no solo menos suerte -- debilita
(b) y refuerza que valga la pena seguir con (a)/(c).

Corre la MISMA regla de staking (la original: min_edge=8%, kelly=10%,
max_odds=3.0) en las 4 ligas, para que la comparacion sea de manzanas con
manzanas -- no se reconstruyen los parametros ganadores especificos de
Serie A/Bundesliga del barrido de 24 combinaciones (no quedaron registrados
en el roadmap, solo el ROI resultante), asi que se usa el mismo punto de
referencia ya usado en la tabla original de Fase 8.

Reutiliza _simulate_bankroll/INITIAL_BANKROLL de economic_backtest.py, y la
misma logica de seleccion de apuestas (apertura de Pinnacle, PSH/PSD/PSA)
que epl_temporal_stability_check.py, generalizada por liga.

Salida: data/runs/temporal_stability_check_all_leagues.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR
from src.evaluation.economic_backtest import _simulate_bankroll, INITIAL_BANKROLL

MAX_STAKE_FRACTION = 0.05
ORIGINAL_RULE = {"min_edge_threshold": 0.08, "kelly_fraction": 0.10, "max_odds": 3.0}

SIDES = [
    ("home", "PSH", "blend_prob_home", "H"),
    ("draw", "PSD", "blend_prob_draw", "D"),
    ("away", "PSA", "blend_prob_away", "A"),
]


def _select_bets(df: pd.DataFrame, min_edge_threshold: float, max_odds: float) -> pd.DataFrame:
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
                "bet_side": best["side"], "bet_odds": best["odds"],
                "bet_fair_prob": best["fair_prob"], "bet_edge": best["edge"],
                "kelly_full": kelly_full, "won": row["FTR"] == best["ftr_code"],
            })
            records.append(record)
    return pd.DataFrame(records)


def _load_predictions(league_key: str) -> pd.DataFrame:
    league_dir = PROCESSED_DATA_DIR / league_key
    path = league_dir / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path}.")
        if league_dir.exists():
            candidates = sorted(league_dir.glob("*.csv"))
            print(f"  CSVs encontrados en {league_dir}: {[c.name for c in candidates]}")
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


def analyze_league(league_key: str) -> list:
    print(f"\n=== {league_key} ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []

    bets = _select_bets(df_eval, ORIGINAL_RULE["min_edge_threshold"], ORIGINAL_RULE["max_odds"])
    if bets.empty:
        print("[AVISO] Cero apuestas seleccionadas.")
        return []

    bets = bets.sort_values("Date").reset_index(drop=True)
    bets = _simulate_bankroll(bets, kelly_fraction=ORIGINAL_RULE["kelly_fraction"],
                               max_stake_fraction=MAX_STAKE_FRACTION,
                               initial_bankroll=INITIAL_BANKROLL)
    bets["fold_test_season"] = bets["fold_test_season"].astype(str)

    seasons = sorted(bets["fold_test_season"].unique())
    n_seasons = len(seasons)
    half = n_seasons // 2
    first_half_seasons = set(seasons[:half]) if half > 0 else set()
    second_half_seasons = set(seasons[half:])

    rows = []
    n_positive_seasons = 0
    print("ROI por temporada:")
    for season in seasons:
        season_bets = bets[bets["fold_test_season"] == season]
        if season_bets.empty:
            continue
        summary = _roi_summary(season_bets)
        if summary["roi"] > 0:
            n_positive_seasons += 1
        print(f"  {season}: n={summary['n_bets']:4d}  win_rate={summary['win_rate']:.2%}  "
              f"ROI={summary['roi']:+.2%}")
        rows.append({"league_key": league_key, "temporada": season, **summary})

    print(f"Temporadas con ROI positivo: {n_positive_seasons} de {len(seasons)}")

    if first_half_seasons:
        s1 = _roi_summary(bets[bets["fold_test_season"].isin(first_half_seasons)])
        print(f"Primera mitad {sorted(first_half_seasons)}: ROI={s1['roi']:+.2%} (n={s1['n_bets']})")
        rows.append({"league_key": league_key, "temporada": "PRIMERA_MITAD", **s1})
    if second_half_seasons:
        s2 = _roi_summary(bets[bets["fold_test_season"].isin(second_half_seasons)])
        print(f"Segunda mitad {sorted(second_half_seasons)}: ROI={s2['roi']:+.2%} (n={s2['n_bets']})")
        rows.append({"league_key": league_key, "temporada": "SEGUNDA_MITAD", **s2})

    total_summary = _roi_summary(bets)
    print(f"Agregado completo: ROI={total_summary['roi']:+.2%} (n={total_summary['n_bets']})")
    rows.append({"league_key": league_key, "temporada": "TOTAL", **total_summary})

    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "temporal_stability_check_all_leagues.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")

    print("\n=== Resumen: temporadas con ROI positivo por liga (regla original) ===")
    for league_key in LEAGUES.keys():
        league_rows = [r for r in all_rows if r["league_key"] == league_key
                       and r["temporada"] not in ("PRIMERA_MITAD", "SEGUNDA_MITAD", "TOTAL")]
        if not league_rows:
            continue
        n_pos = sum(1 for r in league_rows if r["roi"] > 0)
        print(f"  {league_key}: {n_pos} de {len(league_rows)} temporadas positivas")

    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que epl_temporal_stability_check.py.")


if __name__ == "__main__":
    run()