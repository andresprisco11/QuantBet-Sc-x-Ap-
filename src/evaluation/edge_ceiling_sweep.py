"""
Fase 8, siguiente paso tras edge_magnitude_calibration_check.py.

Resultado real de ese script (2026-08-19): bucketeando las apuestas YA
seleccionadas (misma regla de siempre: min_edge=8%, max_odds=3.0) por
magnitud de edge, La Liga y Serie A muestran una secuencia de ROI
estrictamente decreciente a medida que el edge detectado crece -- la firma
clasica de winner's curse en la cola (el modelo confunde ruido de
estimacion con señal real cuanto mas se aleja del precio de mercado). EPL
y Bundesliga NO muestran el mismo patron limpio (buckets extremos con
muestra chica, ruidosos).

Un hallazgo separado, igual de importante: el mismo bucketeo por RANGO DE
CUOTA (no por edge) no tiene un patron universal -- La Liga y Serie A en
realidad MEJORAN su ROI en cuotas altas (2.00-3.00, la zona de underdogs
moderados que es el objetivo #2 del proyecto), mientras que Bundesliga se
hunde especificamente ahi (-16.65% en 2.50-3.00) y EPL es ruidoso. Esto
importa porque descarta la hipotesis mas simple ("cuotas altas = mal") y
deja a la magnitud del edge, no la cuota en si, como la señal mas
consistente entre ligas.

Pregunta concreta que este script responde: si se añade un TECHO de edge
(ademas del piso ya existente de min_edge=8%) a la regla de seleccion,
¿mejora el ROI realizado? Se prueba una grilla de techos candidatos,
elegidos a partir de los propios cortes de bucket ya usados en
edge_magnitude_calibration_check.py (25% y 40%, donde el ROI empezaba a
caer en La Liga/Serie A), mas un techo intermedio en 30% y 35%, y el caso
"sin techo" como base de comparacion (la regla actual del proyecto).

Esto NO asume que un techo va a funcionar en las 4 ligas por igual -- las
ejecuta y reporta todas, y explicita el trade-off entre ROI y volumen de
apuestas (un techo agresivo mejora ROI pero reduce n, lo cual importa para
Tier 1/Tier 2, que necesitan volumen suficiente para armar parlays).

Salida: data/runs/edge_ceiling_sweep.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

MIN_EDGE_THRESHOLD = 0.08
MAX_ODDS = 3.0

# None = sin techo (regla actual del proyecto, caso base de comparacion).
CEILINGS_TO_TEST = [None, 0.40, 0.35, 0.30, 0.25]

SIDES = [
    ("home", "PSH", "blend_prob_home", "H"),
    ("draw", "PSD", "blend_prob_draw", "D"),
    ("away", "PSA", "blend_prob_away", "A"),
]


def _select_bets(df: pd.DataFrame, min_edge_threshold: float, max_odds: float,
                  max_edge_ceiling: float = None) -> pd.DataFrame:
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
        if best is None or best["edge"] <= min_edge_threshold:
            continue
        if max_edge_ceiling is not None and best["edge"] > max_edge_ceiling:
            continue
        records.append({
            "Date": row["Date"], "fold_test_season": str(row["fold_test_season"]),
            "bet_side": best["side"], "bet_odds": best["odds"],
            "bet_edge": best["edge"], "won": row["FTR"] == best["ftr_code"],
        })
    return pd.DataFrame(records)


def _load_predictions(league_key: str) -> pd.DataFrame:
    league_dir = PROCESSED_DATA_DIR / league_key
    path = league_dir / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        path = league_dir / "model_predictions_oos_walkforward_v2.csv"  # fallback MLS
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe archivo de predicciones OOS.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_blend].copy()
    return df.sort_values("Date").reset_index(drop=True)


def _flat_roi(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    return bets.apply(lambda r: (r["bet_odds"] - 1) if r["won"] else -1, axis=1).sum() / len(bets)


def analyze_league(league_key: str) -> list:
    print(f"\n=== {league_key} ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []

    rows = []
    print(f"  {'techo':10s} {'n':>5s} {'win_rate':>9s} {'flat_roi':>9s} {'vs. base':>10s}")
    base_roi = None
    for ceiling in CEILINGS_TO_TEST:
        bets = _select_bets(df_eval, MIN_EDGE_THRESHOLD, MAX_ODDS, max_edge_ceiling=ceiling)
        label = "sin techo" if ceiling is None else f"<={ceiling:.0%}"
        if bets.empty:
            print(f"  {label:10s} {'--':>5s} {'--':>9s} {'--':>9s} {'--':>10s}")
            continue
        n = len(bets)
        win_rate = bets["won"].mean()
        roi = _flat_roi(bets)
        if ceiling is None:
            base_roi = roi
            delta_str = "(base)"
        else:
            delta_str = f"{roi - base_roi:+.2%}" if base_roi is not None else "--"
        print(f"  {label:10s} {n:5d} {win_rate:9.2%} {roi:9.2%} {delta_str:>10s}")
        rows.append({
            "league_key": league_key, "ceiling": label, "n": n,
            "win_rate": win_rate, "flat_roi": roi,
            "delta_vs_base": (roi - base_roi) if (ceiling is not None and base_roi is not None) else 0.0,
        })
    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "edge_ceiling_sweep.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: comparar cada fila contra 'sin techo' (la regla actual del proyecto) en la "
          "misma liga. Un techo es candidato a adoptarse solo si (a) el ROI sube de forma consistente "
          "vs. sin techo, Y (b) el volumen de apuestas (n) no cae tanto que comprometa Tier 1/Tier 2 "
          "(parlays necesitan pool suficiente de picks por semana). Si el techo mejora ROI en La Liga/"
          "Serie A (donde el hallazgo de edge_magnitude_calibration_check.py fue mas limpio) pero no en "
          "EPL/Bundesliga, la regla correcta puede ser un techo POR LIGA, no uno global -- consistente "
          "con que el proyecto ya trata cada liga como un modelo independiente.")
    print("\nNo se loggea en el sistema de tracking hasta decidir si se adopta como regla nueva -- "
          "mismo criterio que el resto de la cadena de Fase 8.")


if __name__ == "__main__":
    run()