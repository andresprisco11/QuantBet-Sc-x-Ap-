"""
Fase 8, siguiente paso tras selection_composition_temporal_check.py.

Ese script (2026-08-18) midio 3 dimensiones de composicion (lado H/D/A,
rango de cuota, concentracion por equipo) comparando temporadas recientes
(2324-2526) contra el resto, en las 4 ligas europeas. Resultado real:

- Lado H/D/A: pct_home/draw/away CASI IDENTICO entre "recientes" y "resto"
  en SERIEA (41.4%->40.7% away) y BUNDESLIGA (69.9%->68.9% home) -- descarta
  que el modelo se haya inclinado hacia un lado nuevo.
- Cuota promedio: tambien casi estable en SERIEA (2.312->2.334) y solo
  sube un poco en BUNDESLIGA (2.247->2.375) -- señal debil, no explica un
  colapso de 13-14 puntos de win rate.
- La señal real y grande: **win_rate se derrumba** a cuota/lado
  practicamente identicos -- SERIEA 48.44%->34.51% (-13.9pp), BUNDESLIGA
  47.79%->34.44% (-13.3pp) -- mientras que en EPL/LALIGA (control) el
  win_rate se mantiene estable o incluso MEJORA (EPL 40.51%->44.90%).
  Esto es una señal de que la PROBABILIDAD ESTIMADA para esas apuestas dejo
  de corresponder a la frecuencia real de acierto, en las mismas ligas/
  temporadas donde el CLV se mantiene positivo -- no es que el modelo
  eligio apuestas mas arriesgadas, es que las apuestas que elige (mismo
  perfil de cuota y lado que antes) empezaron a fallar mas.
- El top 5 de equipos mas apostados cambio de composicion, y en SERIEA la
  concentracion en el top 5 subio fuerte (64.1%->78.8%). Dato notable a
  simple vista: **Parma** aparece en el top 5 de "recientes" en Serie A
  (23 apuestas) sin aparecer en el top 5 de "resto" -- consistente con ser
  un equipo RECIEN ASCENDIDO (poca historia en el dataset de entrenamiento).
  Mismo patron en BUNDESLIGA: **Heidenheim** aparece en el top 5 de
  "recientes" (15 apuestas) sin aparecer en el de "resto" -- tambien un
  equipo de ascenso reciente (2023).

Hipotesis nueva, concreta y testeable (no especulacion sobre features que
no estan confirmadas -- xG, tiros, etc. -- sino sobre lo que el propio
walk-forward expone: cuanta historia tiene cada equipo en el momento de la
prediccion): el modelo selecciona sistematicamente apuestas sobre equipos
con POCA historia en la ventana de entrenamiento (recien ascendidos, o
equipos con pocas temporadas previas en el dataset), y esas apuestas fallan
mas seguido que las de equipos con historia larga -- consistente con que el
modelo (Poisson + recencia + categoria sintetica para equipos sin
historial, ver Fase 2) tiene mas incertidumbre real ahi de la que su propia
probabilidad estimada refleja, y esa incertidumbre se traduce en mas
volumen de apuestas de alto edge APARENTE (justamente porque el modelo esta
menos seguro de si mismo en esos casos, no porque haya mas edge real).

Este script prueba esa hipotesis directamente: para cada apuesta ya
seleccionada (misma regla: min_edge=8%, max_odds=3.0), calcula cuantas
TEMPORADAS PREVIAS distintas jugo en la liga el equipo por el que se
apuesta (home si bet_side=='home', away si bet_side=='away'; no hay draws
porque pct_draw=0% en la seleccion de max-edge), usando el historial
COMPLETO de matches_clean.csv (no solo el set OOS) hasta la temporada
anterior a la apostada. Clasifica en "tenure_bajo" (<=1 temporada previa,
equipo nuevo o casi) vs. "tenure_alto" (>=2 temporadas previas), y compara
win_rate/ROI entre los dos grupos, separando ademas "recientes" (2324-2526)
de "resto" para ver si el efecto es mas fuerte ahora que antes.

Corre sobre las 4 ligas europeas para tener EPL/La Liga como control.

Salida: data/runs/team_tenure_calibration_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ORIGINAL_RULE = {"min_edge_threshold": 0.08, "max_odds": 3.0}
RECENT_SEASONS = {"2324", "2425", "2526"}
LOW_TENURE_MAX_PRIOR_SEASONS = 1  # <=1 temporada previa = tenure_bajo

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
    """(team, season) -> numero de temporadas DISTINTAS previas en las que ese
    equipo aparecio (local o visitante) en matches_clean.csv de esta liga."""
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not path.exists():
        print(f"[AVISO] {league_key}: no existe {path}, no se puede calcular tenure de equipo.")
        return {}

    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    seasons_sorted = sorted(df["season"].unique())

    team_seasons_seen = {}  # team -> set of seasons already seen (strictly before current)
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
    print(f"\n=== {league_key} ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []

    if "HomeTeam" not in df_eval.columns or "AwayTeam" not in df_eval.columns:
        print("[AVISO] No hay HomeTeam/AwayTeam en las predicciones OOS -- no se puede calcular tenure.")
        return []

    bets = _select_bets(df_eval, ORIGINAL_RULE["min_edge_threshold"], ORIGINAL_RULE["max_odds"])
    if bets.empty:
        print("[AVISO] Cero apuestas seleccionadas.")
        return []

    prior_map = _team_prior_seasons_map(league_key)
    if not prior_map:
        return []

    def _tenure(row):
        team = row["HomeTeam"] if row["bet_side"] == "home" else row["AwayTeam"]
        return prior_map.get((team, row["fold_test_season"]))

    bets["team_bet_on"] = bets.apply(lambda r: r["HomeTeam"] if r["bet_side"] == "home" else r["AwayTeam"], axis=1)
    bets["prior_seasons"] = bets.apply(_tenure, axis=1)
    bets = bets.dropna(subset=["prior_seasons"]).copy()
    bets["prior_seasons"] = bets["prior_seasons"].astype(int)
    bets["tenure_group"] = bets["prior_seasons"].apply(
        lambda n: "tenure_bajo" if n <= LOW_TENURE_MAX_PRIOR_SEASONS else "tenure_alto")
    bets["es_reciente"] = bets["fold_test_season"].isin(RECENT_SEASONS)

    def _summary(subset):
        n = len(subset)
        if n == 0:
            return {"n_bets": 0, "win_rate": float("nan"), "flat_roi_per_bet": float("nan")}
        win_rate = subset["won"].mean()
        flat_profit = ((subset["bet_odds"] - 1.0) * subset["won"] - (1 - subset["won"])).sum()
        return {"n_bets": n, "win_rate": win_rate, "flat_roi_per_bet": flat_profit / n}

    rows = []
    print(f"{'periodo':12s} {'tenure':12s} {'n':>5s} {'win_rate':>9s} {'flat_ROI':>9s}")
    for periodo_label, periodo_mask in [("Recientes", bets["es_reciente"]), ("Resto", ~bets["es_reciente"])]:
        for tenure_group in ["tenure_bajo", "tenure_alto"]:
            subset = bets[periodo_mask & (bets["tenure_group"] == tenure_group)]
            s = _summary(subset)
            print(f"{periodo_label:12s} {tenure_group:12s} {s['n_bets']:5d} "
                  f"{s['win_rate']:9.2%} {s['flat_roi_per_bet']:+9.3f}")
            rows.append({"league_key": league_key, "periodo": periodo_label,
                         "tenure_group": tenure_group, **s})

    # pct de volumen que es tenure_bajo, por periodo -- si sube en "Recientes"
    # respecto a "Resto", confirma que el modelo esta apostando MAS sobre
    # equipos de poca historia ahora que antes.
    for periodo_label, periodo_mask in [("Recientes", bets["es_reciente"]), ("Resto", ~bets["es_reciente"])]:
        subset = bets[periodo_mask]
        if subset.empty:
            continue
        pct_bajo = (subset["tenure_group"] == "tenure_bajo").mean()
        print(f"  {periodo_label}: {pct_bajo:.1%} del volumen es tenure_bajo (n={len(subset)})")
        rows.append({"league_key": league_key, "periodo": periodo_label,
                      "tenure_group": "PCT_TENURE_BAJO", "n_bets": len(subset),
                      "win_rate": pct_bajo, "flat_roi_per_bet": float("nan")})

    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "team_tenure_calibration_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: si en SERIEA/BUNDESLIGA el win_rate de 'tenure_bajo' es notablemente peor que "
          "'tenure_alto' (y esa brecha es mayor o solo aparece en 'Recientes'), mientras que en "
          "EPL/LALIGA la brecha es chica o pareja entre periodos, confirma que el modelo esta "
          "sobre-confiando en equipos de poca historia en esas 2 ligas, cada vez mas -- y el fix "
          "concreto es penalizar o filtrar el edge estimado para equipos con pocas temporadas "
          "previas en el dataset, no un ajuste de staking general.")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que selection_composition_temporal_check.py.")


if __name__ == "__main__":
    run()