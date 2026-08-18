"""
Fase 8, "Proximos pasos" punto 2 -- pregunta abierta nueva (la mas afilada
del proyecto en este momento): clv_temporal_trend_check.py ya confirmo que
el deterioro de ROI en Serie A/Bundesliga (2324->2425->2526, empeorando
casi monotonicamente) NO es porque el mercado se volvio mas dificil de
vencer -- el CLV se mantuvo positivo y estable en esas mismas temporadas.
Eso deja una sola explicacion posible: la COMPOSICION de lo que el modelo
selecciona como "mejor apuesta" cambio con el tiempo, y ese cambio es lo
que esta perdiendo plata pese a seguir comprando barato.

Este script no asume cual es el cambio -- lo mide, temporada por temporada,
en 3 dimensiones directamente observables sobre las apuestas ya seleccionadas
(misma regla de seleccion que los scripts anteriores: min_edge=8%, max_odds=3.0,
apertura de Pinnacle):

  1. Lado apostado (home/draw/away) -- % de cada uno por temporada. Si el
     modelo empezo a inclinarse hacia un lado sistematicamente mas dificil
     de acertar (draws, o away en ligas top-heavy), eso es composicion, no
     mercado.
  2. Rango de cuota de las apuestas seleccionadas -- si el modelo empezo a
     seleccionar cuotas mas altas (mas varianza, mas tipico de winner's
     curse) en las temporadas recientes, la degradacion es coherente con
     "el modelo se volvio mas agresivo/ruidoso en su seleccion", no con
     "el mercado cambio".
  3. Concentracion por equipo -- si un puñado de equipos especificos
     (posiblemente con datos ruidosos o cambios de plantilla no capturados
     por las features) empiezan a dominar las apuestas seleccionadas en las
     temporadas recientes, eso apunta a un problema de datos/features para
     esos equipos en particular, no un fenomeno de liga completa.

Corre sobre las 4 ligas europeas (no solo Serie A/Bundesliga) para tener
EPL/La Liga como control -- si estas 3 dimensiones se mantienen estables en
EPL/La Liga mientras cambian en Serie A/Bundesliga, eso confirma que el
cambio de composicion es especifico de esas 2 ligas y no un artefacto del
pipeline entero.

Reutiliza la misma logica de seleccion de apuestas que
temporal_stability_check_all_leagues.py / clv_temporal_trend_check.py
(propia de este script, no una funcion privada no confirmada de
selection_bias_check.py) para que la comparacion temporada por temporada
sea sobre el MISMO conjunto de apuestas ya analizado en los 2 chequeos
anteriores.

HomeTeam/AwayTeam: se asume que estas columnas existen en el CSV de
predicciones OOS (vienen de matches_clean.csv via el pipeline de features).
Si no existen, el script lo avisa y simplemente omite el desglose por
equipo -- no falla en silencio ni inventa datos.

Salida: data/runs/selection_composition_temporal_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ORIGINAL_RULE = {"min_edge_threshold": 0.08, "max_odds": 3.0}

# Las 3 temporadas recientes vs. el resto -- mismo corte que ya reveló el
# deterioro monotónico en temporal_stability_check_all_leagues.py.
RECENT_SEASONS = {"2324", "2425", "2526"}

ODDS_BUCKETS = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0001)]
ODDS_BUCKET_LABELS = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0"]

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


def _odds_bucket_pct(odds: pd.Series) -> dict:
    out = {}
    for (lo, hi), label in zip(ODDS_BUCKETS, ODDS_BUCKET_LABELS):
        out[f"pct_odds_{label}"] = ((odds >= lo) & (odds < hi)).mean()
    return out


def _team_concentration(bets: pd.DataFrame) -> tuple:
    """Devuelve (top5_equipos_str, pct_de_apuestas_en_top5) si hay columnas de equipo."""
    if "HomeTeam" not in bets.columns or "AwayTeam" not in bets.columns:
        return ("(sin datos de equipo)", float("nan"))
    teams = pd.concat([bets["HomeTeam"], bets["AwayTeam"]])
    counts = teams.value_counts()
    top5 = counts.head(5)
    pct_top5 = top5.sum() / len(bets) if len(bets) > 0 else float("nan")
    top5_str = ", ".join(f"{team}({n})" for team, n in top5.items())
    return (top5_str, pct_top5)


def analyze_league(league_key: str) -> list:
    print(f"\n=== {league_key} ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []

    if "HomeTeam" not in df_eval.columns or "AwayTeam" not in df_eval.columns:
        print("[AVISO] No hay columnas HomeTeam/AwayTeam en el CSV de predicciones -- "
              "se omite el desglose por equipo para esta liga.")

    bets = _select_bets(df_eval, ORIGINAL_RULE["min_edge_threshold"], ORIGINAL_RULE["max_odds"])
    if bets.empty:
        print("[AVISO] Cero apuestas seleccionadas.")
        return []

    seasons = sorted(bets["fold_test_season"].unique())
    rows = []
    print(f"{'Temporada':10s} {'n':>5s} {'%Home':>7s} {'%Draw':>7s} {'%Away':>7s} "
          f"{'cuota_avg':>10s} {'cuota_med':>10s} {'win_rate':>9s} {'flat_ROI':>9s}")
    for season in seasons:
        sb = bets[bets["fold_test_season"] == season]
        n = len(sb)
        pct_home = (sb["bet_side"] == "home").mean()
        pct_draw = (sb["bet_side"] == "draw").mean()
        pct_away = (sb["bet_side"] == "away").mean()
        avg_odds = sb["bet_odds"].mean()
        med_odds = sb["bet_odds"].median()
        win_rate = sb["won"].mean()
        flat_profit = ((sb["bet_odds"] - 1.0) * sb["won"] - (1 - sb["won"])).sum()
        flat_roi = flat_profit / n if n > 0 else float("nan")
        buckets = _odds_bucket_pct(sb["bet_odds"])
        top5_str, pct_top5 = _team_concentration(sb)

        print(f"{season:10s} {n:5d} {pct_home:7.1%} {pct_draw:7.1%} {pct_away:7.1%} "
              f"{avg_odds:10.3f} {med_odds:10.3f} {win_rate:9.2%} {flat_roi:+9.3f}")

        row = {
            "league_key": league_key, "temporada": season, "n_bets": n,
            "pct_home": pct_home, "pct_draw": pct_draw, "pct_away": pct_away,
            "avg_odds": avg_odds, "median_odds": med_odds, "win_rate": win_rate,
            "flat_roi_per_bet": flat_roi, "pct_top5_equipos": pct_top5,
            "top5_equipos": top5_str, "es_temporada_reciente": season in RECENT_SEASONS,
        }
        row.update(buckets)
        rows.append(row)

    # Comparacion explicita: promedio de temporadas recientes (2324-2526) vs. el resto.
    recent = bets[bets["fold_test_season"].isin(RECENT_SEASONS)]
    older = bets[~bets["fold_test_season"].isin(RECENT_SEASONS)]
    print("\n  -- Recientes (2324-2526) vs. resto --")
    for label, subset in [("Recientes", recent), ("Resto", older)]:
        if subset.empty:
            print(f"  {label}: sin datos")
            continue
        n = len(subset)
        pct_home = (subset["bet_side"] == "home").mean()
        pct_draw = (subset["bet_side"] == "draw").mean()
        pct_away = (subset["bet_side"] == "away").mean()
        avg_odds = subset["bet_odds"].mean()
        win_rate = subset["won"].mean()
        top5_str, pct_top5 = _team_concentration(subset)
        print(f"  {label:10s} n={n:4d}  %H/%D/%A={pct_home:.1%}/{pct_draw:.1%}/{pct_away:.1%}  "
              f"cuota_avg={avg_odds:.3f}  win_rate={win_rate:.2%}  %en_top5_equipos={pct_top5:.1%}")
        print(f"    Top 5 equipos mas apostados: {top5_str}")

    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "selection_composition_temporal_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: comparar la fila 'Recientes' vs 'Resto' de SERIEA y BUNDESLIGA contra la misma "
          "comparacion en EPL/LALIGA. Si %draw, cuota_avg o concentracion en pocos equipos cambia "
          "fuerte en Serie A/Bundesliga pero se mantiene estable en EPL/La Liga, eso identifica "
          "cual dimension especifica de la seleccion es la que se degrado -- y es el candidato "
          "directo para el proximo filtro o ajuste de features.")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que clv_temporal_trend_check.py.")


if __name__ == "__main__":
    run()