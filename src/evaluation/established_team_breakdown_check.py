"""
Fase 8, siguiente paso tras team_tenure_calibration_check.py -- la hipotesis
de tenure de equipo (2026-08-18) quedo REFUTADA por el propio dato, y de
forma mas reveladora de lo esperado:

- En Serie A y Bundesliga, el grupo "tenure_bajo" (equipos con <=1
  temporada previa en el dataset) es una porcion CHICA y DECRECIENTE del
  volumen reciente (Serie A: 37.5% del volumen en 'Resto' -> apenas 5.3% en
  'Recientes'; Bundesliga: 46.0% -> 5.6%) -- lo opuesto de "mas exposicion a
  equipos nuevos con el tiempo".
- Peor para la hipotesis: en los pocos casos tenure_bajo que SI hay en
  'Recientes', el win rate es RELATIVAMENTE BUENO (Serie A 50.00% n=6,
  Bundesliga 40.00% n=5 -- muestras chiquitas, no concluyentes solas).
- La señal real, con muestra grande y confiable, esta en el otro grupo:
  **tenure_alto (equipos con historial largo en el dataset) es el que se
  derrumbo** -- Serie A 40.00% (Resto, n=80) -> 33.64% (Recientes, n=107);
  Bundesliga 55.74% (Resto, n=61, ROI +0.154) -> 34.12% (Recientes, n=85,
  ROI -0.228). Ahi vive casi todo el volumen y casi toda la perdida.

Conclusion: el problema NO es que el modelo tenga poca informacion sobre
equipos nuevos -- es que dejo de predecir bien a equipos que YA conoce
hace tiempo, en Serie A/Bundesliga, en las ultimas 3 temporadas. Esto abre
una pregunta distinta: ese deterioro en tenure_alto, es AMPLIO (afecta por
igual a muchos equipos establecidos -- coherente con una degradacion de
modelo/features generalizada en esas 2 ligas) o esta CONCENTRADO en un
puñado de equipos puntuales (coherente con una historia idiosincratica de
2-3 clubes -- cambio de entrenador, crisis institucional, etc. -- que no es
un problema del modelo en general sino de esos casos)?

Esa distincion importa para decidir el siguiente paso: si es amplio, hay
que revisar la arquitectura/features del modelo para esas 2 ligas (posible
reconsideracion de Sportmonks, hipotesis (c) ya anotada en el roadmap); si
esta concentrado, el fix es mucho mas quirurgico (excluir o marcar esos
equipos puntuales) y no dice nada sobre el modelo en general.

Este script SOLO mide dispersion -- no explica el "por que" de ningun club
en particular (eso requeriria contexto futbolistico que no esta en los
datos). Para las apuestas ya seleccionadas y clasificadas como tenure_alto
(misma logica de team_tenure_calibration_check.py), desglosa win rate y
volumen POR EQUIPO, separando 'Recientes' (2324-2526) de 'Resto', en las 4
ligas europeas.

Salida: data/runs/established_team_breakdown_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ORIGINAL_RULE = {"min_edge_threshold": 0.08, "max_odds": 3.0}
RECENT_SEASONS = {"2324", "2425", "2526"}
LOW_TENURE_MAX_PRIOR_SEASONS = 1
MIN_TEAM_BETS_TO_SHOW = 3  # no mostrar equipos con muestra irrelevante

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
            record = {
                "Date": row["Date"], "fold_test_season": str(row["fold_test_season"]),
                "bet_side": best["side"], "bet_odds": best["odds"],
                "bet_edge": best["edge"], "won": row["FTR"] == best["ftr_code"],
            }
            if "HomeTeam" in df.columns:
                record["HomeTeam"] = row.get("HomeTeam")
            if "AwayTeam" in df.columns:
                record["AwayTeam"] = row.get("AwayTeam")
            records.append(record)
    return pd.DataFrame(records)


def _load_predictions(league_key: str) -> pd.DataFrame:
    league_dir = PROCESSED_DATA_DIR / league_key
    path = league_dir / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path}.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_blend].copy()
    return df.sort_values("Date").reset_index(drop=True)


def _team_prior_seasons_map(league_key: str) -> dict:
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not path.exists():
        print(f"[AVISO] {league_key}: no existe {path}, no se puede calcular tenure de equipo.")
        return {}

    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    seasons_sorted = sorted(df["season"].unique())

    team_seasons_seen = {}
    prior_count = {}
    for season in seasons_sorted:
        season_df = df[df["season"] == season]
        teams_this_season = set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"]))
        for team in teams_this_season:
            prior_count[(team, season)] = len(team_seasons_seen.get(team, set()))
        for team in teams_this_season:
            team_seasons_seen.setdefault(team, set()).add(season)

    return prior_count


def analyze_league(league_key: str) -> list:
    print(f"\n=== {league_key} (solo apuestas tenure_alto) ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []
    if "HomeTeam" not in df_eval.columns or "AwayTeam" not in df_eval.columns:
        print("[AVISO] No hay HomeTeam/AwayTeam -- se omite esta liga.")
        return []

    bets = _select_bets(df_eval, ORIGINAL_RULE["min_edge_threshold"], ORIGINAL_RULE["max_odds"])
    if bets.empty:
        print("[AVISO] Cero apuestas seleccionadas.")
        return []

    prior_map = _team_prior_seasons_map(league_key)
    if not prior_map:
        return []

    bets["team_bet_on"] = bets.apply(lambda r: r["HomeTeam"] if r["bet_side"] == "home" else r["AwayTeam"], axis=1)
    bets["prior_seasons"] = bets.apply(
        lambda r: prior_map.get((r["team_bet_on"], r["fold_test_season"])), axis=1)
    bets = bets.dropna(subset=["prior_seasons"]).copy()
    bets["prior_seasons"] = bets["prior_seasons"].astype(int)
    bets = bets[bets["prior_seasons"] > LOW_TENURE_MAX_PRIOR_SEASONS].copy()  # solo tenure_alto
    bets["es_reciente"] = bets["fold_test_season"].isin(RECENT_SEASONS)

    if bets.empty:
        print("[AVISO] Cero apuestas tenure_alto.")
        return []

    rows = []
    for periodo_label, periodo_mask in [("Recientes", bets["es_reciente"]), ("Resto", ~bets["es_reciente"])]:
        subset = bets[periodo_mask]
        if subset.empty:
            continue
        by_team = subset.groupby("team_bet_on").agg(
            n_bets=("won", "size"), win_rate=("won", "mean"))
        by_team = by_team[by_team["n_bets"] >= MIN_TEAM_BETS_TO_SHOW].sort_values("n_bets", ascending=False)

        n_teams_total = subset["team_bet_on"].nunique()
        overall_win_rate = subset["won"].mean()
        print(f"\n  -- {periodo_label} -- {n_teams_total} equipos distintos, "
              f"win_rate global tenure_alto={overall_win_rate:.2%}, n={len(subset)}")
        if by_team.empty:
            print(f"    (ningun equipo individual con >= {MIN_TEAM_BETS_TO_SHOW} apuestas)")
        else:
            print(f"    {'equipo':20s} {'n':>4s} {'win_rate':>9s}")
            for team, row in by_team.iterrows():
                marker = " <-- por debajo del global" if row["win_rate"] < overall_win_rate - 0.10 else ""
                print(f"    {team[:20]:20s} {int(row['n_bets']):4d} {row['win_rate']:9.2%}{marker}")
            n_below = (by_team["win_rate"] < overall_win_rate - 0.10).sum()
            pct_volumen_below = by_team.loc[by_team["win_rate"] < overall_win_rate - 0.10, "n_bets"].sum() / len(subset)
            print(f"    Equipos con win_rate >10pp por debajo del global: {n_below} de {len(by_team)} "
                  f"({pct_volumen_below:.1%} del volumen de este periodo)")

        for team, row in by_team.iterrows():
            rows.append({"league_key": league_key, "periodo": periodo_label, "team": team,
                         "n_bets": int(row["n_bets"]), "win_rate": row["win_rate"]})
        rows.append({"league_key": league_key, "periodo": periodo_label, "team": "__GLOBAL__",
                     "n_bets": len(subset), "win_rate": overall_win_rate})

    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "established_team_breakdown_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura para SERIEA/BUNDESLIGA en 'Recientes': si son POCOS equipos (2-4) los que "
          "concentran la mayoria del volumen con win_rate muy por debajo del global, el problema es "
          "idiosincratico de esos clubes puntuales, no del modelo en general. Si son MUCHOS equipos "
          "distintos, cada uno con pocas apuestas pero todos rindiendo mal, es un problema sistemico "
          "de calibracion del modelo en esas ligas -- candidato real para revisar arquitectura/"
          "features (hipotesis (c), Sportmonks) en vez de seguir buscando un patron de seleccion.")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que team_tenure_calibration_check.py.")


if __name__ == "__main__":
    run()