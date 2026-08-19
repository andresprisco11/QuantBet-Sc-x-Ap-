"""
Fase 8, nueva linea de investigacion (2026-08-19) -- objetivo #2 del
proyecto, nunca resuelto: detectar con fiabilidad cuando "el chico le
gana al grande" (edge real en underdogs/cuotas altas).

Motivacion, tomada directamente de la bibliografia del proyecto, no de una
corazonada:

- Andrew Mack (Statistical Sports Models), sobre el "Benter Boost": el
  blend modelo+mercado solo tiene sentido pegado fuerte al mercado cuando
  el mercado es claramente mas eficiente que el modelo. En mercados/zonas
  menos eficientes, pegarse al mercado borra exactamente la ventaja que el
  modelo podria aportar. Este proyecto usa un blend weight practicamente
  constante (~50%) en todas las cuotas -- no diferenciado por que tan
  eficiente es esa zona especifica del mercado.
- El propio `selection_bias_check.py` (Fase 8) ya mostro que el sesgo de
  seleccion (winner's curse) es mas severo justo en el rango de cuota
  2.00-3.00 -- la zona de underdogs moderados donde este proyecto quiere
  encontrar edge real.

Pregunta concreta que este script responde, sin asumir nada nuevo: dentro
de las apuestas YA seleccionadas (misma regla de siempre: min_edge=8%,
max_odds=3.0), ¿el TAMAÑO del edge detectado por el modelo predice el
resultado en la direccion esperada (edges mas grandes -> mejor win rate)
o al reves (edges mas grandes -> peor win rate)? Si es al reves, es la
firma clasica de winner's curse en la cola: el modelo esta confundiendo
ruido de estimacion con señal real cuanto mas se aleja del precio de
mercado -- y sugiere que un TECHO de edge (no solo un piso) podria
mejorar la seleccion, especificamente en el segmento de cuotas altas
donde el proyecto busca edge en underdogs.

Se mide bucketeando las apuestas seleccionadas por magnitud de edge, y
por separado por rango de cuota, en las 4 ligas europeas + MLS. No se
adivina nada de la arquitectura del modelo -- solo se mide, igual que
selection_composition_temporal_check.py y el resto de la cadena de Fase 8.

Salida: data/runs/edge_magnitude_calibration_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ORIGINAL_RULE = {"min_edge_threshold": 0.08, "max_odds": 3.0}

EDGE_BUCKETS = [
    (0.08, 0.15, "08-15%"),
    (0.15, 0.25, "15-25%"),
    (0.25, 0.40, "25-40%"),
    (0.40, float("inf"), "40%+"),
]

ODDS_BUCKETS = [
    (1.0, 1.50, "1.00-1.50"),
    (1.50, 2.00, "1.50-2.00"),
    (2.00, 2.50, "2.00-2.50"),
    (2.50, 3.00, "2.50-3.00"),
]

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


def _bucket_report(bets: pd.DataFrame, buckets: list, value_col: str, label: str) -> list:
    rows = []
    print(f"\n  -- por {label} --")
    print(f"    {'rango':12s} {'n':>5s} {'win_rate':>9s} {'flat_roi':>9s}")
    for lo, hi, bucket_label in buckets:
        subset = bets[(bets[value_col] >= lo) & (bets[value_col] < hi)]
        if subset.empty:
            print(f"    {bucket_label:12s} {'--':>5s} {'--':>9s} {'--':>9s}")
            continue
        n = len(subset)
        win_rate = subset["won"].mean()
        # ROI flat-stake: gana (odds-1) si acierta, pierde 1 si falla
        flat_roi = (subset.apply(lambda r: (r["bet_odds"] - 1) if r["won"] else -1, axis=1).sum()) / n
        print(f"    {bucket_label:12s} {n:5d} {win_rate:9.2%} {flat_roi:9.2%}")
        rows.append({"bucket_type": label, "bucket": bucket_label, "n": n,
                     "win_rate": win_rate, "flat_roi": flat_roi})
    return rows


def analyze_league(league_key: str) -> list:
    print(f"\n=== {league_key} ===")
    df_eval = _load_predictions(league_key)
    if df_eval.empty:
        return []

    bets = _select_bets(df_eval, ORIGINAL_RULE["min_edge_threshold"], ORIGINAL_RULE["max_odds"])
    if bets.empty:
        print("[AVISO] Cero apuestas seleccionadas.")
        return []

    print(f"  Total apuestas seleccionadas: {len(bets)}  |  win_rate global: {bets['won'].mean():.2%}")

    rows = []
    rows += [dict(r, league_key=league_key) for r in _bucket_report(bets, EDGE_BUCKETS, "bet_edge", "magnitud de edge")]
    rows += [dict(r, league_key=league_key) for r in _bucket_report(bets, ODDS_BUCKETS, "bet_odds", "rango de cuota")]
    return rows


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(analyze_league(league_key))

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "edge_magnitude_calibration_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: si el win_rate/ROI BAJA a medida que el edge detectado crece (en vez de subir, "
          "que es lo que uno esperaria si un edge mas grande fuera una señal mas fuerte), es la firma "
          "de winner's curse en la cola -- el modelo confunde ruido de estimacion con señal real cuanto "
          "mas se aleja de la cuota de mercado. Si ademas ese patron se concentra en el rango de cuota "
          "2.00-3.00 (el segmento de underdogs moderados, donde el proyecto busca su objetivo #2), es "
          "el candidato mas concreto que ha tenido el proyecto hasta ahora para una regla de seleccion "
          "nueva: un TECHO de edge ademas del piso ya existente (min_edge=8%), o un blend weight variable "
          "por zona de eficiencia del mercado en vez de uno fijo (~50%) -- la sugerencia directa del "
          "Benter Boost de Andrew Mack.")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que el resto de la cadena de Fase 8.")


if __name__ == "__main__":
    run()