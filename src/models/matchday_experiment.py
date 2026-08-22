"""
Experimento SIN DINERO -- version 3: cuotas EN VIVO de Pinnacle via The Odds
API, ya no capturas de pantalla pegadas a mano (ver roadmap, "Arranca el
punto 2 de la ruta", 2026-08-22).

CAMBIO REAL respecto a v2 (no cosmetico): v2 usaba FanDuel como proxy del
mercado porque no habia cuotas de Pinnacle en vivo -- avisaba explicitamente
que el peso del blend estaba tuneado contra Pinnacle pero se ejecutaba
contra otro libro. Ese aviso YA NO APLICA: se confirmo con una llamada real
(2026-08-22, ver roadmap) que Pinnacle SI esta disponible para las 4 ligas
en The Odds API. Este script pide Pinnacle directo, la misma fuente contra
la que se calibro V4_BRIER["market_brier"] en economic_backtest.py.

SEGUNDO CAMBIO REAL: ya no hay una lista MATCHES editada a mano cada semana.
El script pide TODOS los partidos que The Odds API tenga cargados para cada
liga (tipicamente la proxima jornada completa) y los procesa automatico.

MAPEO DE NOMBRES -- el punto mas fragil de este cambio, tratado con cuidado:
los nombres de equipo que devuelve The Odds API (ej. "Brighton and Hove
Albion", "Atlético Madrid", "VfB Stuttgart") NO son los mismos que usa
football-data.co.uk (ej. "Brighton", "Ath Madrid", "Stuttgart"). Se resuelve
en 3 pasos, en orden, y si ninguno funciona el partido se DESCARTA con un
aviso explicito en vez de adivinar:
  1. Alias manual conocido (TEAM_ALIASES) -- convenciones publicas y bien
     documentadas de football-data.co.uk, pero escritas de memoria, no
     verificadas contra un fixture real de la temporada 2026-27 todavia.
  2. Match exacto contra el set real de equipos del dataset (known_teams).
  3. Match difuso (difflib) como ultimo recurso, con un umbral alto para no
     confundir equipos distintos.
Cada corrida imprime la tabla de mapeo completa (API -> football-data) ANTES
de predecir nada, para poder auditar/corregir alias antes de confiar en el
resultado -- misma disciplina que el resto del proyecto: no asumir, mostrar
la respuesta real y dejar corregir.

METODOLOGIA (sin cambios respecto a v2):
- v4 entrenado con TODO el historico real disponible (no walk-forward).
- Forma reciente = promedio de tiros al arco (diferencial neto) de los
  ultimos 5 partidos jugados reales de cada equipo, sin shift (el proximo
  partido todavia no paso, no hay fuga posible).
- Blend Benter Boost (compute_blend_weight/blend_probabilities) + Kelly
  fraccional + edge threshold/techo por liga + gate de Tier 1: EXACTAMENTE
  la misma logica de economic_backtest.py, sin reinventar nada.
- Correct score: Poisson home/away independiente (Dixon-Coles descartado,
  resultado negativo, ver roadmap) -- celdas chicas (0-0/1-1) menos
  confiables que el 1X2.

AVISO DE FRESCURA DE FORMA (generalizado, ya no hardcodeado a un equipo):
si el ultimo partido real de un equipo en el dataset es de hace mas de
STALE_FORM_DAYS dias, se marca explicitamente -- esto cubre casos como
Ipswich la semana pasada (descendio, sin partidos recientes) sin tener que
acordarse de agregar el nombre a mano cada vez.

Uso: python -m src.models.matchday_experiment
"""
import difflib
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.poisson_model import MAX_GOALS, scoreline_matrix, outcome_probs_from_matrix
from src.models.poisson_model_v4 import build_long_format_v4, fit_poisson_model_v4, PROMOTED_LABEL
from src.models.blending import compute_blend_weight, blend_probabilities
from src.tracking.run_logger import RUNS_DIR
from src.ingestion.theoddsapi_live_odds_loader import fetch_upcoming_odds

# Carpeta de registro PERMANENTE (versionada, a diferencia de data/processed/
# que esta en .gitignore) -- "fuente de informacion" de los experimentos
# matchday de la temporada 2026-27.
EXPERIMENTS_DIR = RUNS_DIR / "experiments_2026_27"

# Brier score CONFIRMADO de v4 (walk-forward OOS real, ver roadmap Fase 8 /
# V4_REFERENCE en backtest_v7.py y backtest_v8.py) -- el peso del blend se
# deriva de esto, NO se inventa un numero nuevo aca. market_brier fue medido
# historicamente contra Pinnacle -- ahora el mercado real de este script
# TAMBIEN es Pinnacle, ya no hay descalce de libro.
V4_BRIER = {
    "EPL": {"model_brier": 0.595578, "market_brier": 0.560801},
    "LALIGA": {"model_brier": 0.589535, "market_brier": 0.572426},
    "SERIEA": {"model_brier": 0.599352, "market_brier": 0.577595},
    "BUNDESLIGA": {"model_brier": 0.607687, "market_brier": 0.576732},
}

# Mismas reglas de staking que economic_backtest.py (Fase 3.5 / Fase 8,
# tuneadas sobre datos reales) -- se REUSAN tal cual.
KELLY_FRACTION = 0.10
MIN_EDGE_THRESHOLD = 0.08
MAX_STAKE_FRACTION = 0.05
MAX_ODDS_DECIMAL = 3.0
MAX_EDGE_CEILING_BY_LEAGUE = {"EPL": None, "LALIGA": 0.25, "SERIEA": 0.25, "BUNDESLIGA": 0.30}

TIER1_THRESHOLD = 0.80  # gate real del proyecto (mandato original) para una sola pierna.
STALE_FORM_DAYS = 45  # si el ultimo partido real de un equipo es mas viejo que esto, se avisa.

# --- Alias API (The Odds API) -> football-data.co.uk ---------------------
# Escritos de memoria a partir de las convenciones publicas conocidas de
# football-data.co.uk -- NO verificados contra un fixture real de la
# temporada 2026-27 todavia. Cualquier equipo que no aparezca aca se
# resuelve por match exacto o difuso contra known_teams (ver resolve_team);
# lo que no se resuelva de ninguna forma se imprime como UNMATCHED, nunca
# se adivina en silencio.
TEAM_ALIASES = {
    # EPL
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves", "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham", "Newcastle United": "Newcastle",
    "Leicester City": "Leicester", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
    "West Bromwich Albion": "West Brom", "Queens Park Rangers": "QPR",
    "AFC Bournemouth": "Bournemouth", "Norwich City": "Norwich", "Cardiff City": "Cardiff",
    "Hull City": "Hull",
    # LaLiga
    "Athletic Club": "Ath Bilbao", "Atlético Madrid": "Ath Madrid", "Atletico Madrid": "Ath Madrid",
    "Real Sociedad": "Sociedad", "Real Betis": "Betis", "Rayo Vallecano": "Vallecano",
    "Celta Vigo": "Celta", "RCD Espanyol": "Espanol", "Espanyol": "Espanol",
    "Deportivo Alavés": "Alaves", "Alavés": "Alaves", "Real Valladolid": "Valladolid",
    "CA Osasuna": "Osasuna", "Girona FC": "Girona", "RCD Mallorca": "Mallorca",
    "UD Las Palmas": "Las Palmas", "Cadiz CF": "Cadiz",
    # Serie A
    "Inter Milan": "Inter", "AC Milan": "Milan", "AS Roma": "Roma", "SS Lazio": "Lazio",
    "Hellas Verona": "Verona", "Cagliari Calcio": "Cagliari", "Genoa CFC": "Genoa",
    "Torino FC": "Torino", "US Lecce": "Lecce", "Udinese Calcio": "Udinese",
    "Parma Calcio 1913": "Parma", "Como 1907": "Como", "Venezia FC": "Venezia",
    "SSC Napoli": "Napoli", "Atalanta BC": "Atalanta", "Bologna FC 1909": "Bologna",
    "ACF Fiorentina": "Fiorentina", "US Sassuolo": "Sassuolo", "Empoli FC": "Empoli",
    # Bundesliga
    "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt", "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg", "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach", "1. FC Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg", "1. FSV Mainz 05": "Mainz", "Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim", "1899 Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg", "1. FC Köln": "FC Koln", "1. FC Koln": "FC Koln",
    "FC St. Pauli": "St Pauli", "St. Pauli": "St Pauli", "VfL Bochum": "Bochum",
    "1. FC Heidenheim": "Heidenheim", "Hamburger SV": "Hamburg", "Hertha BSC": "Hertha",
    "Fortuna Düsseldorf": "Fortuna Dusseldorf",
}


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.lower().strip()


def resolve_team(api_name: str, known_teams: set) -> str | None:
    """Resuelve un nombre de equipo de la API contra el set real de equipos
    del dataset, en 3 pasos (alias -> exacto -> difuso). Devuelve None si no
    hay match confiable -- nunca adivina en silencio."""
    if api_name in TEAM_ALIASES and TEAM_ALIASES[api_name] in known_teams:
        return TEAM_ALIASES[api_name]
    if api_name in known_teams:
        return api_name
    norm_targets = {_normalize(t): t for t in known_teams}
    if _normalize(api_name) in norm_targets:
        return norm_targets[_normalize(api_name)]
    close = difflib.get_close_matches(_normalize(api_name), list(norm_targets.keys()), n=1, cutoff=0.75)
    if close:
        return norm_targets[close[0]]
    return None


def current_recent_form(df: pd.DataFrame, window: int = 5):
    """Forma reciente ACTUAL de cada equipo (sin shift) + fecha de su ultimo
    partido real -- esta segunda parte es lo que permite el aviso de
    frescura generalizado (STALE_FORM_DAYS), sin hardcodear nombres."""
    home = pd.DataFrame({
        "Date": df["Date"], "team": df["HomeTeam"], "stat_for": df["HST"], "stat_against": df["AST"],
    })
    away = pd.DataFrame({
        "Date": df["Date"], "team": df["AwayTeam"], "stat_for": df["AST"], "stat_against": df["HST"],
    })
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["stat_diff"] = long_df["stat_for"] - long_df["stat_against"]
    long_df = long_df.sort_values(["team", "Date"])
    form = long_df.groupby("team")["stat_diff"].apply(lambda s: s.tail(window).mean()).to_dict()
    last_date = long_df.groupby("team")["Date"].max().to_dict()
    return form, last_date


def load_and_train(league_key: str):
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["season"] = df["season"].astype(str)
    known_teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    form, last_date = current_recent_form(df)
    long_df = build_long_format_v4(df)
    model = fit_poisson_model_v4(long_df)
    return model, known_teams, form, last_date


def predict(model, known_teams, form, home_team, away_team):
    home_label = home_team if home_team in known_teams else PROMOTED_LABEL
    away_label = away_team if away_team in known_teams else PROMOTED_LABEL
    home_form = form.get(home_team, 0.0)
    away_form = form.get(away_team, 0.0)
    if pd.isna(home_form):
        home_form = 0.0
    if pd.isna(away_form):
        away_form = 0.0

    home_row = pd.DataFrame({"team": [home_label], "opponent": [away_label], "is_home": [1], "team_recent_form": [home_form]})
    away_row = pd.DataFrame({"team": [away_label], "opponent": [home_label], "is_home": [0], "team_recent_form": [away_form]})
    lambda_home = model.predict(home_row).iloc[0]
    lambda_away = model.predict(away_row).iloc[0]
    matrix = scoreline_matrix(lambda_home, lambda_away, max_goals=MAX_GOALS)
    prob_home, prob_draw, prob_away = outcome_probs_from_matrix(matrix)
    return lambda_home, lambda_away, prob_home, prob_draw, prob_away, matrix, home_label, away_label


def pinnacle_fixtures(league_key: str) -> pd.DataFrame:
    """Trae las cuotas en vivo de la liga y las reduce a UNA fila por
    partido con las 3 cuotas decimales de Pinnacle -- descarta cualquier
    partido donde Pinnacle no cotizo los 3 resultados (h2h incompleto)."""
    raw = fetch_upcoming_odds(league_key)
    if raw.empty:
        return pd.DataFrame()
    pin = raw[(raw["bookmaker"] == "pinnacle") & (raw["market"] == "h2h")].copy()
    if pin.empty:
        print(f"[AVISO] {league_key}: Pinnacle no trajo cuotas h2h en esta corrida (raro, dado el probe "
              f"de hoy -- revisar manualmente si persiste).")
        return pd.DataFrame()

    rows = []
    for event_id, grp in pin.groupby("event_id"):
        home_team = grp["home_team"].iloc[0]
        away_team = grp["away_team"].iloc[0]
        commence_time = grp["commence_time"].iloc[0]
        prices = dict(zip(grp["outcome_name"], grp["outcome_price_decimal"]))
        if home_team not in prices or away_team not in prices or "Draw" not in prices:
            continue  # h2h incompleto para este partido -- se descarta, no se adivina el faltante
        rows.append({
            "event_id": event_id, "home_team": home_team, "away_team": away_team,
            "commence_time": commence_time,
            "decimal_home": prices[home_team], "decimal_draw": prices["Draw"], "decimal_away": prices[away_team],
        })
    return pd.DataFrame(rows)


def main():
    results = []
    now_utc = datetime.now(timezone.utc)
    run_date_str = now_utc.strftime("%Y-%m-%d")

    for league_key in V4_BRIER:
        print(f"\n{'#'*70}\n# {league_key}\n{'#'*70}")
        fixtures = pinnacle_fixtures(league_key)
        if fixtures.empty:
            print(f"[SKIP] {league_key}: sin partidos con Pinnacle h2h completo en esta corrida.")
            continue

        print(f"Entrenando v4 con todo el historico real disponible ({league_key})...")
        model, known_teams, form, last_date = load_and_train(league_key)

        print(f"\n--- Mapeo de nombres de equipo (API -> football-data.co.uk) ---")
        mapped_rows = []
        for _, row in fixtures.iterrows():
            home_fd = resolve_team(row["home_team"], known_teams)
            away_fd = resolve_team(row["away_team"], known_teams)
            status_home = home_fd if home_fd else "*** SIN RESOLVER ***"
            status_away = away_fd if away_fd else "*** SIN RESOLVER ***"
            print(f"  {row['home_team']:30s} -> {status_home:20s} | {row['away_team']:30s} -> {status_away}")
            mapped_rows.append({**row.to_dict(), "home_fd": home_fd, "away_fd": away_fd})

        for row in mapped_rows:
            if row["home_fd"] is None or row["away_fd"] is None:
                print(f"[DESCARTADO -- nombre sin resolver] {row['home_team']} vs {row['away_team']}: "
                      f"agregar a TEAM_ALIASES manualmente y volver a correr.")
                continue

            home_fd, away_fd = row["home_fd"], row["away_fd"]
            lh, la, ph, pd_, pa, matrix, home_label, away_label = predict(model, known_teams, form, home_fd, away_fd)

            decimal_home, decimal_draw, decimal_away = row["decimal_home"], row["decimal_draw"], row["decimal_away"]
            p_home_mkt = 1.0 / decimal_home
            p_draw_mkt = 1.0 / decimal_draw
            p_away_mkt = 1.0 / decimal_away
            overround = p_home_mkt + p_draw_mkt + p_away_mkt
            p_home_mkt_fair = p_home_mkt / overround
            p_draw_mkt_fair = p_draw_mkt / overround
            p_away_mkt_fair = p_away_mkt / overround

            # --- Blend Benter Boost: ahora contra Pinnacle real, la misma
            # fuente que calibro V4_BRIER["market_brier"] -- ya no hay
            # descalce de libro (ver docstring). ---
            brier = V4_BRIER[league_key]
            market_weight = compute_blend_weight(brier["model_brier"], brier["market_brier"])
            model_probs_df = pd.DataFrame({"prob_home": [ph], "prob_draw": [pd_], "prob_away": [pa]})
            market_probs_df = pd.DataFrame({"prob_home": [p_home_mkt_fair], "prob_draw": [p_draw_mkt_fair], "prob_away": [p_away_mkt_fair]})
            blended = blend_probabilities(model_probs_df, market_probs_df, market_weight)
            bh, bd, ba = blended["prob_home"].iloc[0], blended["prob_draw"].iloc[0], blended["prob_away"].iloc[0]

            promoted_flag = " [PROMOTED_TEAM -- sin historial real]" if home_label == PROMOTED_LABEL or away_label == PROMOTED_LABEL else ""
            stale_flag = ""
            for team_fd, side in [(home_fd, "local"), (away_fd, "visita")]:
                ld = last_date.get(team_fd)
                if ld is not None and not pd.isna(ld):
                    gap_days = (now_utc.replace(tzinfo=None) - ld).days
                    if gap_days > STALE_FORM_DAYS:
                        stale_flag += f" [AVISO: forma de {team_fd} ({side}) desactualizada, ultimo partido real hace {gap_days}d]"

            print(f"\n=== {home_fd} vs {away_fd} ({league_key}) -- {row['commence_time']}{promoted_flag}{stale_flag} ===")
            print(f"lambda_home={lh:.2f} lambda_away={la:.2f}  (peso del mercado en el blend: {market_weight:.1%})")
            print(f"{'':12s} {'MODELO':>10s} {'PINNACLE (sin vig)':>18s} {'BLEND (Benter Boost)':>22s}")
            print(f"{'Local':12s} {ph:>10.1%} {p_home_mkt_fair:>18.1%} {bh:>22.1%}")
            print(f"{'Empate':12s} {pd_:>10.1%} {p_draw_mkt_fair:>18.1%} {bd:>22.1%}")
            print(f"{'Visita':12s} {pa:>10.1%} {p_away_mkt_fair:>18.1%} {ba:>22.1%}")
            print(f"(overround del libro: {overround:.1%})")

            sides = [
                ("Local", bh, decimal_home), ("Empate", bd, decimal_draw), ("Visita", ba, decimal_away),
            ]
            max_edge_ceiling = MAX_EDGE_CEILING_BY_LEAGUE[league_key]
            best_bet = None
            for side_label, fair_prob, decimal_odds in sides:
                if decimal_odds > MAX_ODDS_DECIMAL:
                    continue
                edge = fair_prob * decimal_odds - 1.0
                if best_bet is None or edge > best_bet["edge"]:
                    best_bet = {"side": side_label, "fair_prob": fair_prob, "decimal_odds": decimal_odds, "edge": edge}

            if best_bet is None:
                print(f"Ningun lado tiene cuota <= {MAX_ODDS_DECIMAL} -- sin apuesta evaluable.")
                best_bet = {"side": "N/A", "fair_prob": 0.0, "decimal_odds": 0.0, "edge": -1.0}

            print(f"Mejor edge disponible (blend vs. cuota Pinnacle, cuota<={MAX_ODDS_DECIMAL}): "
                  f"{best_bet['side']} {best_bet['edge']:+.1%} (prob. blend {best_bet['fair_prob']:.1%}, "
                  f"cuota decimal {best_bet['decimal_odds']:.2f})")

            qualifies = best_bet["edge"] > MIN_EDGE_THRESHOLD
            if max_edge_ceiling is not None and best_bet["edge"] > max_edge_ceiling:
                qualifies = False
                print(f"[DESCARTADO] edge {best_bet['edge']:.1%} supera el techo de {league_key} "
                      f"(<={max_edge_ceiling:.0%}) -- fuera de la zona validada.")
            if qualifies:
                kelly_full = (best_bet["fair_prob"] * best_bet["decimal_odds"] - 1.0) / (best_bet["decimal_odds"] - 1.0)
                kelly_stake_frac = min(max(kelly_full * KELLY_FRACTION, 0.0), MAX_STAKE_FRACTION)
                print(f"[CALIFICA -- edge>{MIN_EDGE_THRESHOLD:.0%}] Kelly fraccional ({KELLY_FRACTION:.0%} de Kelly "
                      f"completo, tope {MAX_STAKE_FRACTION:.0%}): stake sugerido = {kelly_stake_frac:.2%} del bankroll")
            else:
                print(f"No califica para apuesta bajo las reglas de economic_backtest.py "
                      f"(min_edge={MIN_EDGE_THRESHOLD:.0%}, max_odds={MAX_ODDS_DECIMAL}, "
                      f"techo={'sin techo' if max_edge_ceiling is None else f'<={max_edge_ceiling:.0%}'}).")

            best_single_prob = max(bh, bd, ba)
            tier1 = best_single_prob >= TIER1_THRESHOLD
            print(f"Tier 1 (>={TIER1_THRESHOLD:.0%} en una sola pierna): "
                  f"{'SI -- ' + str(round(best_single_prob*100,1)) + '%' if tier1 else f'NO ({best_single_prob:.1%}, no alcanza)'}")

            top_scores = []
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    top_scores.append(((i, j), matrix[i, j]))
            top_scores.sort(key=lambda x: -x[1])
            print("Top 5 marcadores exactos (modelo, Poisson independiente -- ver aviso de Dixon-Coles):")
            for (i, j), p in top_scores[:5]:
                print(f"   {i}-{j}: {p:.1%}")

            results.append({
                "league": league_key, "home": home_fd, "away": away_fd,
                "commence_time": row["commence_time"],
                "lambda_home": lh, "lambda_away": la,
                "model_prob_home": ph, "model_prob_draw": pd_, "model_prob_away": pa,
                "pinnacle_prob_home_fair": p_home_mkt_fair, "pinnacle_prob_draw_fair": p_draw_mkt_fair,
                "pinnacle_prob_away_fair": p_away_mkt_fair, "overround": overround,
                "market_weight_blend": market_weight,
                "blend_prob_home": bh, "blend_prob_draw": bd, "blend_prob_away": ba,
                "best_bet_side": best_bet["side"], "best_bet_edge": best_bet["edge"],
                "best_bet_qualifies": qualifies,
                "tier1_reached": tier1, "tier1_best_prob": best_single_prob,
                "top_score_1": f"{top_scores[0][0][0]}-{top_scores[0][0][1]}", "top_score_1_prob": top_scores[0][1],
                "promoted_team_involved": bool(promoted_flag),
                "stale_form_flag": bool(stale_flag),
            })

    if not results:
        print("\n[AVISO] Ningun partido evaluado en esta corrida -- ver avisos arriba.")
        return

    out_df = pd.DataFrame(results)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"matchday_experiment_{run_date_str}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado (carpeta versionada, se commitea) -> {out_path} ({len(out_df)} partidos evaluados)")


if __name__ == "__main__":
    main()
