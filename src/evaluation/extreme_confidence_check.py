"""
Motivado por la aclaracion del usuario del gate de capital real (2026-08-20):
"80%" en el gate original no significaba 80% de acierto -- significaba, en
palabras del propio usuario, "de 2000 apuestas solo se hayan perdido 20 como
mucho". Haciendo la cuenta explicita: 20 perdidas sobre 2000 = 1% de perdidas
= 99% DE ACIERTO REAL, sostenido sobre >=2000 apuestas. Esto es un umbral
mucho mas exigente que el 80-82% que traian las instrucciones originales del
proyecto y que todos los scripts de Tier 1 (`tier1_probability_validation.py`,
`tier1_probability_validation_mls.py`) venian midiendo hasta ahora -- ninguno
de esos dos scripts mide mas alla de 90% de confianza del modelo, porque
nunca hizo falta.

Este script no entrena nada nuevo ni reimplementa ningun modelo -- toma TODO
lo que el proyecto ya calculo (4 ligas europeas + MLS + ATP + WTA, 6 mercados
con pipeline completo) y responde, con evidencia real y de una sola vez, la
pregunta que el gate aclarado exige: "cuando el modelo dice que esta MUY
seguro (90%, 95%, 98%, 99%), que tan seguido acierta de verdad, y cuanto
volumen real hay ahi?"

Fuentes reutilizadas (sin volver a calcular nada):
- Futbol (4 ligas): 'calibration_analysis_v4_long.csv' por liga (generado por
  calibration_analysis.py) -- 3 filas por partido evaluado (home/draw/away),
  con predicted_prob/odds/actual ya calculados. A umbrales altos (>=90%) el
  riesgo de contar dos lados del mismo partido es no-nulo en teoria pero
  irrelevante en la practica: si un lado supera 90%, los otros dos no pueden
  (suman ~1 entre los 3) -- mismo supuesto que ya usa
  tier1_probability_validation.py sin deduplicar explicitamente.
- MLS: 'calibration_analysis_mls_long.csv' (mismo esquema, calibration_analysis_mls.py).
- Tenis (ATP/WTA): 'predictions_v1.csv' (tennis_logistic_model.py) no tiene
  formato "long" todavia -- se construye aca mismo, tomando por partido el
  lado que el blend favorece (blend_prob_player1 o su complemento) como
  "predicted_prob", y si ese lado gano como "actual". Es el mismo criterio
  que _select_bets() de economic_backtest_tennis.py usa para elegir lado,
  pero SIN filtro de edge ni de cuota -- acá interesa la confianza del
  modelo en si misma, no la rentabilidad.

Salida: imprime, por mercado y luego TOTAL combinado, una tabla con
n_casos / prob_promedio_dicha / acierto_real / gap / cuota_promedio para
cada umbral en THRESHOLDS. Guarda el detalle combinado en
'data/runs/extreme_confidence_check.csv'.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
MLS_KEY = "MLS"
TENNIS_TOURS = ["ATP", "WTA"]

# El gate real, aclarado por el usuario 2026-08-20: como mucho 20 perdidas
# en 2000 apuestas -> 99% de acierto, sobre >=2000 apuestas.
GATE_MIN_ACCURACY = 0.99
GATE_MIN_N = 2000


def _load_football_long(league_key: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / league_key / "calibration_analysis_v4_long.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path}. Corre 'python -m src.evaluation.calibration_analysis' primero.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["market"] = league_key
    return df[["market", "predicted_prob", "odds", "actual"]]


def _load_mls_long() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / MLS_KEY / "calibration_analysis_mls_long.csv"
    if not path.exists():
        print(f"[SKIP] MLS: no existe {path}. Corre 'python -m src.evaluation.calibration_analysis_mls' primero.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["market"] = "MLS"
    return df[["market", "predicted_prob", "odds", "actual"]]


def _load_tennis_long(tour: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "predictions_v1.csv"
    if not path.exists():
        print(f"[SKIP] TENNIS_{tour}: no existe {path}. Corre "
              f"'python -m src.models.tennis_logistic_model' primero.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    has_market = df["blend_prob_player1"].notna() if "blend_prob_player1" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_market].copy()
    if df.empty:
        return pd.DataFrame()

    p1_favored = df["blend_prob_player1"] >= 0.5
    predicted_prob = df["blend_prob_player1"].where(p1_favored, 1.0 - df["blend_prob_player1"])
    odds = df["Player1_PS_Odds"].where(p1_favored, df["Player2_PS_Odds"])
    p1_won = df["Player1_Won"].astype(bool)
    actual = (p1_won == p1_favored).astype(int)

    out = pd.DataFrame({
        "market": f"TENNIS_{tour.upper()}",
        "predicted_prob": predicted_prob,
        "odds": odds,
        "actual": actual,
    })
    return out.dropna(subset=["predicted_prob", "odds"])


def _threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        subset = df[df["predicted_prob"] >= threshold]
        n = len(subset)
        if n == 0:
            rows.append({
                "umbral_confianza": f">={threshold:.0%}", "n_casos": 0,
                "prob_promedio_dicha": float("nan"), "acierto_real": float("nan"),
                "gap": float("nan"), "cuota_promedio": float("nan"),
                "perdidas": 0,
            })
            continue
        actual_rate = subset["actual"].mean()
        rows.append({
            "umbral_confianza": f">={threshold:.0%}",
            "n_casos": n,
            "prob_promedio_dicha": subset["predicted_prob"].mean(),
            "acierto_real": actual_rate,
            "gap": actual_rate - subset["predicted_prob"].mean(),
            "cuota_promedio": subset["odds"].mean(),
            "perdidas": int((subset["actual"] == 0).sum()),
        })
    return pd.DataFrame(rows).set_index("umbral_confianza")


def run() -> None:
    frames = []
    for league_key in LEAGUES:
        frames.append(_load_football_long(league_key))
    frames.append(_load_mls_long())
    for tour in TENNIS_TOURS:
        frames.append(_load_tennis_long(tour))

    frames = [f for f in frames if not f.empty]
    if not frames:
        print("[AVISO] No hay ningun archivo de predicciones disponible todavia -- nada que medir.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    for market in all_df["market"].unique():
        market_df = all_df[all_df["market"] == market]
        print(f"\n=== {market} (n total con probabilidad de mercado disponible: {len(market_df)}) ===")
        print(_threshold_table(market_df).round(4).to_string())

    print(f"\n\n=== TOTAL COMBINADO -- los {all_df['market'].nunique()} mercados juntos "
          f"({len(all_df)} casos evaluables en total) ===")
    combined_table = _threshold_table(all_df)
    print(combined_table.round(4).to_string())

    print(f"\n=== Lectura directa contra el gate aclarado por el usuario (2026-08-20): "
          f">={GATE_MIN_ACCURACY:.0%} de acierto real, sobre >={GATE_MIN_N} apuestas ===")
    for threshold in THRESHOLDS:
        row = combined_table.loc[f">={threshold:.0%}"]
        n = int(row["n_casos"]) if pd.notna(row["n_casos"]) else 0
        if n == 0:
            print(f"  >={threshold:.0%}: 0 casos en todo el proyecto -- el modelo nunca llega a este nivel de confianza.")
            continue
        acierto = row["acierto_real"]
        perdidas = int(row["perdidas"])
        cumple_accuracy = acierto >= GATE_MIN_ACCURACY
        cumple_volumen = n >= GATE_MIN_N
        print(f"  >={threshold:.0%}: n={n}, acierto real={acierto:.2%} ({perdidas} perdidas), "
              f"cuota promedio={row['cuota_promedio']:.2f} -- "
              f"{'CUMPLE' if cumple_accuracy else 'NO cumple'} accuracy del gate, "
              f"{'CUMPLE' if cumple_volumen else 'NO cumple'} volumen del gate.")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "extreme_confidence_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"\nGuardado detalle combinado (todos los mercados, todos los casos) -> {out_path}")


if __name__ == "__main__":
    run()