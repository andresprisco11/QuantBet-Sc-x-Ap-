"""
Fase 8, "Proximos pasos" punto 2, hipotesis (e): el deterioro real (no
ruido) que aparecio en Serie A/Bundesliga en epl_temporal_stability_check.py
/ temporal_stability_check_all_leagues.py (3 temporadas seguidas negativas
y empeorando, 2324->2425->2526), es porque el MERCADO se volvio mas
eficiente con el tiempo, o porque la SELECCION de apuestas empeoro con el
tiempo sin que el mercado cambiara?

La forma de distinguirlo con lo que ya tenemos: mirar el CLV (Closing Line
Value) temporada por temporada, no solo el ROI.
- Si el CLV promedio TAMBIEN cae en las mismas temporadas donde cae el ROI
  -> el modelo esta perdiendo su ventaja de timing contra el mercado, el
  mercado se volvio mas dificil de vencer con el tiempo (mas sharp).
- Si el CLV se mantiene positivo y estable pero el ROI se derrumba igual
  -> la direccion del modelo sigue siendo correcta (compra barato antes de
  que la linea se mueva a su favor), el problema esta en la SELECCION de
  apuestas (que resultado elegir de los 3), no en el mercado -- mismo
  patron de "winner's curse" que selection_bias_check.py ya diagnostico en
  agregado, pero ahora viendo si empeora con el tiempo.

Corre sobre las 4 ligas europeas (no solo Serie A/Bundesliga) para tener
EPL/La Liga como comparacion -- si el CLV de EPL/La Liga es estable en el
tiempo y el de Serie A/Bundesliga no, eso refuerza que hay algo especifico
de esas dos ligas, no un fenomeno de todo el proyecto.

CLV se calcula directo de las cuotas crudas (no se asume una funcion
privada de economic_backtest.py que no esta confirmada en este entorno):
para el lado apostado, CLV = cuota_apertura_Pinnacle / cuota_cierre_Pinnacle - 1.
Positivo significa que la cuota de cierre bajo respecto a la de apertura
(el mercado se movio a favor de la apuesta despues de que se ejecuto) --
definicion estandar de CLV positivo.

Reutiliza la misma logica de seleccion de apuestas (regla original:
min_edge=8%, kelly=10%, max_odds=3.0) que epl_temporal_stability_check.py /
temporal_stability_check_all_leagues.py, para que la comparacion temporada
por temporada sea sobre el MISMO conjunto de apuestas ya analizado.

Salida: data/runs/clv_temporal_trend_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ORIGINAL_RULE = {"min_edge_threshold": 0.08, "max_odds": 3.0}

# side_name, cuota de apertura (ejecucion), prob justa del blend, codigo FTR, cuota de cierre
SIDES = [
    ("home", "PSH", "blend_prob_home", "H", "PSCH"),
    ("draw", "PSD", "blend_prob_draw", "D", "PSCD"),
    ("away", "PSA", "blend_prob_away", "A", "PSCA"),
]


def _select_bets_with_clv(df: pd.DataFrame, min_edge_threshold: float, max_odds: float) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        best = None
        for side_name, open_col, prob_col, ftr_code, close_col in SIDES:
            odds_open = row[open_col]
            fair_prob = row[prob_col]
            if pd.isna(odds_open) or pd.isna(fair_prob) or odds_open <= 1.0:
                continue
            if max_odds is not None and odds_open > max_odds:
                continue
            edge = fair_prob * odds_open - 1.0
            if best is None or edge > best["edge"]:
                best = {"side": side_name, "odds_open": odds_open, "edge": edge,
                        "ftr_code": ftr_code, "close_col": close_col}
        if best is not None and best["edge"] > min_edge_threshold:
            odds_close = row.get(best["close_col"])
            clv = (best["odds_open"] / odds_close - 1.0) if pd.notna(odds_close) and odds_close > 1.0 else float("nan")
            records.append({
                "Date": row["Date"], "fold_test_season": row["fold_test_season"],
                "bet_side": best["side"], "bet_odds_open": best["odds_open"],
                "bet_odds_close": odds_close, "clv": clv,
                "won": row["FTR"] == best["ftr_code"],
            })
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


def analyze_league(league_key: str) -> list:
    print(f"\n=== {league_key} ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []

    bets = _select_bets_with_clv(df_eval, ORIGINAL_RULE["min_edge_threshold"], ORIGINAL_RULE["max_odds"])
    if bets.empty:
        print("[AVISO] Cero apuestas seleccionadas.")
        return []

    if bets["clv"].isna().all():
        print("[AVISO] No hay cuotas de cierre disponibles (PSCH/PSCD/PSCA) para calcular CLV en esta liga.")
        return []

    bets["fold_test_season"] = bets["fold_test_season"].astype(str)
    seasons = sorted(bets["fold_test_season"].unique())

    rows = []
    print("CLV promedio y % de apuestas con CLV positivo, por temporada:")
    for season in seasons:
        season_bets = bets[bets["fold_test_season"] == season].dropna(subset=["clv"])
        if season_bets.empty:
            print(f"  {season}: sin CLV disponible")
            continue
        avg_clv = season_bets["clv"].mean()
        pct_positive = (season_bets["clv"] > 0).mean()
        n = len(season_bets)
        print(f"  {season}: n={n:4d}  CLV_promedio={avg_clv:+.4f}  %CLV_positivo={pct_positive:.2%}")
        rows.append({"league_key": league_key, "temporada": season, "n_bets": n,
                      "clv_promedio": avg_clv, "pct_clv_positivo": pct_positive})

    total = bets.dropna(subset=["clv"])
    print(f"Agregado completo: n={len(total)}  CLV_promedio={total['clv'].mean():+.4f}  "
          f"%CLV_positivo={(total['clv'] > 0).mean():.2%}")
    rows.append({"league_key": league_key, "temporada": "TOTAL", "n_bets": len(total),
                  "clv_promedio": total["clv"].mean(), "pct_clv_positivo": (total["clv"] > 0).mean()})

    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "clv_temporal_trend_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: comparar esta tendencia de CLV por temporada, liga por liga, contra el ROI por "
          "temporada ya obtenido en temporal_stability_check_all_leagues.py. Si el CLV de Serie A/"
          "Bundesliga tambien cae en 2324-2526 (donde ya sabemos que el ROI se derrumba), apunta a "
          "mercado mas eficiente con el tiempo. Si el CLV se mantiene positivo y estable ahi, apunta "
          "a que la seleccion de apuestas (winner's curse) empeoro con el tiempo, no el mercado.")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio que "
          "los chequeos de estabilidad temporal.")


if __name__ == "__main__":
    run()