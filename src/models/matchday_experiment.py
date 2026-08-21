"""
Experimento SIN DINERO -- 8 partidos reales de manana (5 EPL + 3 LaLiga),
usando v4 (la referencia real de produccion, ya adoptada) contra las cuotas
reales que el usuario pego de su casa de apuestas.

VERSION 2 (2026-08-21): la primera version de este script solo comparaba
probabilidad del modelo vs. probabilidad implicita del mercado -- un
diagnostico de cara-validez, NO el framework real del proyecto. El usuario
lo noto: faltaba el blend Benter Boost, el sizing con Kelly fraccional, el
umbral/techo de edge por liga (economic_backtest.py, ya validado con datos
reales) y el chequeo contra el gate real de Tier 1 (>=80-82%). Esta version
aplica esas 4 piezas, reusando exactamente la misma logica que
economic_backtest.py -- no una version simplificada aparte.

AVISO METODOLOGICO IMPORTANTE que no tenia la v1 tampoco: el peso del blend
(compute_blend_weight) se tuneo historicamente contra PINNACLE (el libro
sharp de referencia del proyecto), no contra FanDuel. Ac'a no hay cuotas de
Pinnacle para estos partidos especificos (no se descargaron en vivo), asi
que se usa FanDuel como proxy del "mercado" tanto para el blend como para
el precio de ejecucion del edge -- una aproximacion razonable para un
experimento sin dinero, pero FanDuel tiene su propio margen/comportamiento,
distinto al de Pinnacle, y el resultado no es identico a lo que
economic_backtest.py mediria con datos reales de Pinnacle.

METODOLOGIA:
- Se entrena v4 con TODO el historico real disponible (no walk-forward --
  aca queremos el mejor modelo posible HOY, no una evaluacion OOS).
- La "forma reciente" (recent_st_diff) de cada equipo para el proximo
  partido se calcula como el promedio de tiros al arco (diferencial neto)
  de sus ultimos 5 partidos jugados REALES -- sin shift(1) esta vez, porque
  el proximo partido todavia no se jugo (no hay fuga posible: es el futuro).
- CAVEATS REALES que hay que leer antes de confiar en esto:
  1) Hull no tiene NINGUN historial en el dataset (2021-2026) -- recien
     ascendido, se le da el tratamiento PROMOTED_TEAM (rating neutro), el
     mismo diseño que v4 ya usa para cualquier equipo nuevo.
  2) Ipswich SI tiene historial, pero su ultimo partido real en el dataset
     es de mayo 2025 (descendieron, no jugaron en la EPL 2025-26) -- su
     "forma reciente" esta desactualizada mas de un año, tratar esa
     prediccion con mas cautela que el resto.
  3. El resto de los equipos de EPL tiene forma actualizada hasta el final
     de la temporada pasada (24 de mayo 2026) -- el modelo NO tiene datos
     de pretemporada/fichajes, solo resultados.
  4. Sevilla y Espanyol YA jugaron su primer partido real de la 2026-27 (15
     y 16 de agosto), asi que su forma reciente incluye ese partido. El
     resto de LaLiga todavia esta con forma de la temporada pasada.
- CORRECT SCORE: v4 asume Poisson home/away INDEPENDIENTES (Dixon-Coles,
  que corrige la correlacion en marcadores bajos como 0-0/1-1, fue
  resultado NEGATIVO y no se adopto -- ver roadmap Fase 2/3). Esto quiere
  decir que las celdas de marcador exacto MAS chicas (0-0, 1-0, 0-1, 1-1)
  probablemente esten peor calibradas que el 1X2 -- se muestran igual, con
  este aviso explicito, no como un numero confiable al mismo nivel que el
  moneyline.

NO es un script de la pipeline permanente (no tiene walk-forward, no logea
en experiment_log.jsonl) -- es una herramienta de matchday, pensada para
editarse a mano cada semana: cambiar EPL_MAP/LALIGA_MAP si aparecen equipos
nuevos, y la lista MATCHES con los partidos y cuotas del dia (formato
americano, tal cual los pega la casa de apuestas). Se puede correr para
cualquier liga ya soportada (agregar SERIEA_MAP/BUNDESLIGA_MAP con el mismo
criterio si hace falta).

Uso: python -m src.models.matchday_experiment
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.poisson_model import MAX_GOALS, scoreline_matrix, outcome_probs_from_matrix
from src.models.poisson_model_v4 import build_long_format_v4, fit_poisson_model_v4, PROMOTED_LABEL
from src.models.blending import compute_blend_weight, blend_probabilities
from src.tracking.run_logger import RUNS_DIR

# Carpeta de registro PERMANENTE (versionada, a diferencia de data/processed/
# que esta en .gitignore) -- "nuestra fuente de informacion" de los
# experimentos matchday de la temporada 2026-27, pedido explicito del
# usuario 2026-08-21: cada corrida de este script se guarda aca, no se pisa.
EXPERIMENTS_DIR = RUNS_DIR / "experiments_2026_27"

# Fecha de los partidos de ESTA corrida -- editar junto con MATCHES cada
# semana, se usa para el nombre del archivo de salida.
MATCHDAY_DATE = "2026-08-22"

# Brier score CONFIRMADO de v4 (walk-forward OOS real, ver roadmap Fase 8 /
# V4_REFERENCE en backtest_v7.py y backtest_v8.py) -- el peso del blend se
# deriva de esto, NO se inventa un numero nuevo aca.
V4_BRIER = {
    "EPL": {"model_brier": 0.595578, "market_brier": 0.560801},
    "LALIGA": {"model_brier": 0.589535, "market_brier": 0.572426},
    "SERIEA": {"model_brier": 0.599352, "market_brier": 0.577595},
    "BUNDESLIGA": {"model_brier": 0.607687, "market_brier": 0.576732},
}

# Mismas reglas de staking que economic_backtest.py (Fase 3.5 / Fase 8,
# tuneadas sobre EPL con datos reales) -- se REUSAN tal cual, no se
# reinventan para este script.
KELLY_FRACTION = 0.10
MIN_EDGE_THRESHOLD = 0.08
MAX_STAKE_FRACTION = 0.05
MAX_ODDS_DECIMAL = 3.0
MAX_EDGE_CEILING_BY_LEAGUE = {"EPL": None, "LALIGA": 0.25, "SERIEA": 0.25, "BUNDESLIGA": 0.30}

TIER1_THRESHOLD = 0.80  # gate real del proyecto (mandato original) para una sola pierna.

EPL_MAP = {
    "Hull": "Hull", "Man United": "Man United", "Everton": "Everton",
    "Crystal Palace": "Crystal Palace", "Ipswich": "Ipswich", "Sunderland": "Sunderland",
    "Nottm Forest": "Nott'm Forest", "Leeds": "Leeds", "Brentford": "Brentford", "Tottenham": "Tottenham",
}
LALIGA_MAP = {
    "Athletic Club": "Ath Bilbao", "Sevilla": "Sevilla", "Valencia": "Valencia",
    "Celta Vigo": "Celta", "Espanyol": "Espanol", "Real Madrid": "Real Madrid",
}

MATCHES = [
    ("EPL", "Hull", "Man United", {"home": 800, "draw": 410, "away": -290}),
    ("EPL", "Everton", "Crystal Palace", {"home": 120, "draw": 240, "away": 230}),
    ("EPL", "Ipswich", "Sunderland", {"home": 180, "draw": 220, "away": 160}),
    ("EPL", "Nottm Forest", "Leeds", {"home": 130, "draw": 230, "away": 210}),
    ("EPL", "Brentford", "Tottenham", {"home": 130, "draw": 270, "away": 185}),
    ("LALIGA", "Athletic Club", "Sevilla", {"home": -140, "draw": 260, "away": 440}),
    ("LALIGA", "Valencia", "Celta Vigo", {"home": 135, "draw": 220, "away": 220}),
    ("LALIGA", "Espanyol", "Real Madrid", {"home": 650, "draw": 370, "away": -240}),
]


def american_to_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return odds / 100.0 + 1.0
    else:
        return 100.0 / (-odds) + 1.0


def current_recent_form(df: pd.DataFrame, window: int = 5) -> dict:
    """Forma reciente ACTUAL de cada equipo (sin shift -- incluye su ultimo
    partido jugado, lista para usarse en el PROXIMO partido, que todavia no
    paso)."""
    home = pd.DataFrame({
        "Date": df["Date"], "team": df["HomeTeam"], "stat_for": df["HST"], "stat_against": df["AST"],
    })
    away = pd.DataFrame({
        "Date": df["Date"], "team": df["AwayTeam"], "stat_for": df["AST"], "stat_against": df["HST"],
    })
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["stat_diff"] = long_df["stat_for"] - long_df["stat_against"]
    long_df = long_df.sort_values(["team", "Date"])
    last_n = long_df.groupby("team")["stat_diff"].apply(lambda s: s.tail(window).mean())
    return last_n.to_dict()


def load_and_train(league_key: str):
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["season"] = df["season"].astype(str)
    known_teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    form = current_recent_form(df)
    long_df = build_long_format_v4(df)
    model = fit_poisson_model_v4(long_df)
    return model, known_teams, form


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


def main():
    print("Entrenando v4 con todo el historico real disponible (EPL, LaLiga)...")
    epl_model, epl_teams, epl_form = load_and_train("EPL")
    laliga_model, laliga_teams, laliga_form = load_and_train("LALIGA")

    results = []
    for league_key, home_app, away_app, odds in MATCHES:
        if league_key == "EPL":
            model, known_teams, form = epl_model, epl_teams, epl_form
            home_fd, away_fd = EPL_MAP[home_app], EPL_MAP[away_app]
        else:
            model, known_teams, form = laliga_model, laliga_teams, laliga_form
            home_fd, away_fd = LALIGA_MAP[home_app], LALIGA_MAP[away_app]

        lh, la, ph, pd_, pa, matrix, home_label, away_label = predict(model, known_teams, form, home_fd, away_fd)

        p_home_mkt = american_to_prob(odds["home"])
        p_draw_mkt = american_to_prob(odds["draw"])
        p_away_mkt = american_to_prob(odds["away"])
        overround = p_home_mkt + p_draw_mkt + p_away_mkt
        p_home_mkt_fair = p_home_mkt / overround
        p_draw_mkt_fair = p_draw_mkt / overround
        p_away_mkt_fair = p_away_mkt / overround

        # --- Blend Benter Boost: mismo peso que economic_backtest.py deriva
        # de model_brier/market_brier, aca aplicado a la probabilidad sin
        # vig de FanDuel (ver aviso metodologico del docstring -- no es
        # Pinnacle, el peso se tuneo contra Pinnacle historicamente). ---
        brier = V4_BRIER[league_key]
        market_weight = compute_blend_weight(brier["model_brier"], brier["market_brier"])
        model_probs_df = pd.DataFrame({"prob_home": [ph], "prob_draw": [pd_], "prob_away": [pa]})
        market_probs_df = pd.DataFrame({"prob_home": [p_home_mkt_fair], "prob_draw": [p_draw_mkt_fair], "prob_away": [p_away_mkt_fair]})
        blended = blend_probabilities(model_probs_df, market_probs_df, market_weight)
        bh, bd, ba = blended["prob_home"].iloc[0], blended["prob_draw"].iloc[0], blended["prob_away"].iloc[0]

        promoted_flag = " [PROMOTED_TEAM -- sin historial real]" if home_label == PROMOTED_LABEL or away_label == PROMOTED_LABEL else ""
        stale_flag = ""
        if home_fd in form and pd.isna(form.get(home_fd)) is False and home_app == "Ipswich":
            stale_flag = " [AVISO: forma de Ipswich desactualizada, ultimo partido real hace >1 año]"

        print(f"\n=== {home_app} vs {away_app} ({league_key}){promoted_flag}{stale_flag} ===")
        print(f"lambda_home={lh:.2f} lambda_away={la:.2f}  (peso del mercado en el blend: {market_weight:.1%})")
        print(f"{'':12s} {'MODELO':>10s} {'MERCADO (sin vig)':>18s} {'BLEND (Benter Boost)':>22s}")
        print(f"{'Local':12s} {ph:>10.1%} {p_home_mkt_fair:>18.1%} {bh:>22.1%}")
        print(f"{'Empate':12s} {pd_:>10.1%} {p_draw_mkt_fair:>18.1%} {bd:>22.1%}")
        print(f"{'Visita':12s} {pa:>10.1%} {p_away_mkt_fair:>18.1%} {ba:>22.1%}")
        print(f"(overround del libro: {overround:.1%})")

        # --- Edge/Kelly/Tier 1 con la probabilidad BLEND (la "fair prob"
        # real del proyecto, no la del modelo solo) contra la cuota decimal
        # de FanDuel -- misma formula que economic_backtest.py._select_bets. ---
        sides = [
            ("Local", bh, odds["home"]), ("Empate", bd, odds["draw"]), ("Visita", ba, odds["away"]),
        ]
        max_edge_ceiling = MAX_EDGE_CEILING_BY_LEAGUE[league_key]
        best_bet = None
        for side_label, fair_prob, american_odds in sides:
            decimal_odds = american_to_decimal(american_odds)
            edge = fair_prob * decimal_odds - 1.0
            if decimal_odds > MAX_ODDS_DECIMAL:
                continue
            if best_bet is None or edge > best_bet["edge"]:
                best_bet = {"side": side_label, "fair_prob": fair_prob, "decimal_odds": decimal_odds, "edge": edge}

        print(f"Mejor edge disponible (blend vs. cuota FanDuel, cuota<=|{MAX_ODDS_DECIMAL}|): "
              f"{best_bet['side']} {best_bet['edge']:+.1%} (prob. blend {best_bet['fair_prob']:.1%}, "
              f"cuota decimal {best_bet['decimal_odds']:.2f})")

        qualifies = best_bet["edge"] > MIN_EDGE_THRESHOLD
        if max_edge_ceiling is not None and best_bet["edge"] > max_edge_ceiling:
            qualifies = False
            print(f"[DESCARTADO] edge {best_bet['edge']:.1%} supera el techo de {league_key} "
                  f"(<={max_edge_ceiling:.0%}) -- fuera de la zona validada, se descarta igual que "
                  f"economic_backtest.py lo haria.")
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
        print(f"Tier 1 (>={TIER1_THRESHOLD:.0%} en una sola pierna, mandato original del proyecto): "
              f"{'SI -- ' + str(round(best_single_prob*100,1)) + '%' if tier1 else f'NO ({best_single_prob:.1%}, no alcanza)'}")

        # Top 5 marcadores exactos segun el modelo
        top_scores = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                top_scores.append(((i, j), matrix[i, j]))
        top_scores.sort(key=lambda x: -x[1])
        print("Top 5 marcadores exactos (modelo, Poisson independiente -- ver aviso de Dixon-Coles):")
        for (i, j), p in top_scores[:5]:
            print(f"   {i}-{j}: {p:.1%}")

        results.append({
            "league": league_key, "home": home_app, "away": away_app,
            "lambda_home": lh, "lambda_away": la,
            "model_prob_home": ph, "model_prob_draw": pd_, "model_prob_away": pa,
            "market_prob_home_fair": p_home_mkt_fair, "market_prob_draw_fair": p_draw_mkt_fair,
            "market_prob_away_fair": p_away_mkt_fair, "overround": overround,
            "market_weight_blend": market_weight,
            "blend_prob_home": bh, "blend_prob_draw": bd, "blend_prob_away": ba,
            "best_bet_side": best_bet["side"], "best_bet_edge": best_bet["edge"],
            "best_bet_qualifies": qualifies,
            "tier1_reached": tier1, "tier1_best_prob": best_single_prob,
            "top_score_1": f"{top_scores[0][0][0]}-{top_scores[0][0][1]}", "top_score_1_prob": top_scores[0][1],
            "promoted_team_involved": bool(promoted_flag),
        })

    out_df = pd.DataFrame(results)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"matchday_experiment_{MATCHDAY_DATE}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado (carpeta versionada, se commitea) -> {out_path}")


if __name__ == "__main__":
    main()
