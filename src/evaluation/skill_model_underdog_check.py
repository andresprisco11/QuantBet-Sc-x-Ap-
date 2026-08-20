"""
Resultado real de `model_market_divergence_check.py` (2026-08-20, confirmado
por el usuario): CERO casos divergentes en los 7 mercados, en los 3 umbrales
probados (80/85/90%) -- el BLEND nunca está muy seguro de un lado que el
mercado considere underdog. Antes de leer esto como "el proyecto no tiene
ninguna señal de underdog todavía" (conclusión posible pero prematura), hace
falta separar dos causas muy distintas que ese resultado por sí solo no
distingue:

(a) el modelo de habilidad (Poisson/Dixon-Coles en fútbol, logístico+Elo en
    tenis), ANTES de mezclarse con el mercado, tampoco genera nunca una
    señal fuerte de underdog -- en ese caso el problema es de datos/features,
    no de la fórmula de blend.
(b) el modelo de habilidad SÍ genera señal de underdog alguna vez, pero el
    blend la diluye -- porque blend_prob es una regresión logística sobre
    [logit(model_prob), logit(market_prob)], y el mercado típicamente pesa
    ~50% o más en esa mezcla (ver roadmap, Fase 8: "el patron de v4 -- blend
    pierde contra el mercado, ~50-52% de peso -- se replica en las 4 ligas").
    Con ese peso, es case matemáticamente casi imposible que blend_prob
    llegue a 80%+ si market_prob está por debajo de 50% -- el mercado tira
    el promedio (en espacio logit) hacia abajo salvo que model_prob sea
    extremo. Si esto es lo que está pasando, el "cero" del script anterior
    no es evidencia de que el proyecto no tenga ninguna señal de underdog:
    es evidencia de que, SI la tiene, el blend la está tapando.

Este script no reentrena nada -- repite exactamente el mismo corte que
`model_market_divergence_check.py` (alineado con mercado vs. divergente/
underdog, mismos umbrales 80/85/90%, mismas fuentes de datos), pero usando
`model_prob_*` / `model_prob_player1` (la salida del modelo de habilidad
SOLO, antes de mezclarse con el mercado) como "predicted_prob" en vez de
`blend_prob_*`. Responde directo la pregunta (a) vs (b) de arriba con
evidencia, no con la fórmula del blend.

Fuentes reutilizadas, sin recalcular nada -- mismos archivos que
`model_market_divergence_check.py`:
- Futbol (4 ligas): 'model_predictions_oos_walkforward_v4.csv', columnas
  model_prob_home/draw/away (Poisson/Dixon-Coles v4, ANTES del blend) y
  pinnacle_close_prob_home/draw/away (referencia de mercado).
- MLS: 'model_predictions_oos_walkforward_mls.csv', mismo esquema.
- Tenis (ATP/WTA): 'predictions_v1.csv', columna model_prob_player1
  (logistico+Elo v3, ANTES del blend) y market_prob_player1.

Salida: misma estructura que model_market_divergence_check.py, impresa por
mercado y en TOTAL combinado. Guarda el detalle en
'data/runs/skill_model_underdog_check.csv'.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

THRESHOLDS = [0.60, 0.70, 0.80, 0.85, 0.90]
MLS_KEY = "MLS"
TENNIS_TOURS = ["ATP", "WTA"]

FOOTBALL_SIDES = [
    ("home", "PSH", "model_prob_home", "pinnacle_close_prob_home", "H"),
    ("draw", "PSD", "model_prob_draw", "pinnacle_close_prob_draw", "D"),
    ("away", "PSA", "model_prob_away", "pinnacle_close_prob_away", "A"),
]
MLS_SIDES = [
    ("home", "PSCH", "model_prob_home", "pinnacle_close_prob_home", "H"),
    ("draw", "PSCD", "model_prob_draw", "pinnacle_close_prob_draw", "D"),
    ("away", "PSCA", "model_prob_away", "pinnacle_close_prob_away", "A"),
]


def _unroll_football(df: pd.DataFrame, market: str, sides) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for side_name, odds_col, model_col, market_col, ftr_code in sides:
            model_prob = row.get(model_col)
            odds = row.get(odds_col)
            market_prob = row.get(market_col)
            if pd.isna(model_prob) or pd.isna(odds) or pd.isna(market_prob):
                continue
            records.append({
                "market": market,
                "predicted_prob": model_prob,
                "market_prob": market_prob,
                "odds": odds,
                "actual": 1 if row["FTR"] == ftr_code else 0,
            })
    return pd.DataFrame(records)


def _load_football(league_key: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path}.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _unroll_football(df, league_key, FOOTBALL_SIDES)


def _load_mls() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / MLS_KEY / "model_predictions_oos_walkforward_mls.csv"
    if not path.exists():
        print(f"[SKIP] MLS: no existe {path}.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _unroll_football(df, "MLS", MLS_SIDES)


def _load_tennis(tour: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "predictions_v1.csv"
    if not path.exists():
        print(f"[SKIP] TENNIS_{tour}: no existe {path}.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    has_model = df["model_prob_player1"].notna() if "model_prob_player1" in df.columns else pd.Series(False, index=df.index)
    has_market = df["market_prob_player1"].notna() if "market_prob_player1" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_model & has_market].copy()
    if df.empty:
        return pd.DataFrame()

    p1_favored = df["model_prob_player1"] >= 0.5
    predicted_prob = df["model_prob_player1"].where(p1_favored, 1.0 - df["model_prob_player1"])
    market_prob = df["market_prob_player1"].where(p1_favored, 1.0 - df["market_prob_player1"])
    odds = df["Player1_PS_Odds"].where(p1_favored, df["Player2_PS_Odds"])
    p1_won = df["Player1_Won"].astype(bool)
    actual = (p1_won == p1_favored).astype(int)

    out = pd.DataFrame({
        "market": f"TENNIS_{tour.upper()}",
        "predicted_prob": predicted_prob,
        "market_prob": market_prob,
        "odds": odds,
        "actual": actual,
    })
    return out.dropna(subset=["predicted_prob", "market_prob", "odds"])


def _divergence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        confident = df[df["predicted_prob"] >= threshold]
        aligned = confident[confident["market_prob"] >= 0.5]
        divergent = confident[confident["market_prob"] < 0.5]
        for label, subset in [("alineado_con_mercado", aligned), ("divergente_underdog", divergent)]:
            n = len(subset)
            if n == 0:
                rows.append({
                    "umbral_confianza": f">={threshold:.0%}", "tipo": label, "n_casos": 0,
                    "acierto_real": float("nan"), "prob_mercado_promedio": float("nan"),
                    "cuota_promedio": float("nan"), "perdidas": 0,
                })
                continue
            rows.append({
                "umbral_confianza": f">={threshold:.0%}",
                "tipo": label,
                "n_casos": n,
                "acierto_real": subset["actual"].mean(),
                "prob_mercado_promedio": subset["market_prob"].mean(),
                "cuota_promedio": subset["odds"].mean(),
                "perdidas": int((subset["actual"] == 0).sum()),
            })
    return pd.DataFrame(rows).set_index(["umbral_confianza", "tipo"])


def run() -> None:
    frames = []
    for league_key in LEAGUES:
        frames.append(_load_football(league_key))
    frames.append(_load_mls())
    for tour in TENNIS_TOURS:
        frames.append(_load_tennis(tour))

    frames = [f for f in frames if not f.empty]
    if not frames:
        print("[AVISO] No hay ningun archivo de predicciones disponible todavia -- nada que medir.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    for market in all_df["market"].unique():
        market_df = all_df[all_df["market"] == market]
        print(f"\n=== {market} (n total: {len(market_df)}) ===")
        print(_divergence_table(market_df).round(4).to_string())

    print(f"\n\n=== TOTAL COMBINADO -- los {all_df['market'].nunique()} mercados juntos "
          f"({len(all_df)} casos) ===")
    combined_table = _divergence_table(all_df)
    print(combined_table.round(4).to_string())

    print(f"\n=== Lectura directa: el modelo de HABILIDAD SOLO (sin mezclar con el mercado), "
          f"cuan seguido esta seguro de un lado que el mercado considera underdog ===")
    any_divergent = False
    for threshold in THRESHOLDS:
        row = combined_table.loc[(f">={threshold:.0%}", "divergente_underdog")]
        n = int(row["n_casos"]) if pd.notna(row["n_casos"]) else 0
        if n == 0:
            print(f"  >={threshold:.0%}: 0 casos divergentes.")
            continue
        any_divergent = True
        print(f"  >={threshold:.0%}: n={n} casos divergentes, acierto real={row['acierto_real']:.2%} "
              f"({int(row['perdidas'])} perdidas), prob. mercado promedio en ese lado="
              f"{row['prob_mercado_promedio']:.2%}, cuota promedio={row['cuota_promedio']:.2f}.")

    if any_divergent:
        print("\n  [HALLAZGO] El modelo de habilidad SI genera señal de underdog en algun umbral -- "
              "el blend la esta diluyendo/tapando (causa (b) del docstring). Esto abre la puerta a "
              "evaluar una regla de staking especifica para estos casos SIN blend, con la debida "
              "cautela de que apartarse del mercado es exactamente donde el proyecto ya encontro "
              "winner's curse en otros contextos -- no se adopta nada todavia, solo se mide.")
    else:
        print("\n  [HALLAZGO] Tampoco el modelo de habilidad solo genera nunca señal de underdog en "
              "estos umbrales -- el problema no es la formula de blend (causa (a) del docstring): "
              "el modelo mismo, con los datos y features actuales, nunca produce una lectura de "
              "underdog con esta confianza. Esto apunta a que hace falta MAS o MEJOR informacion "
              "(datos live, features nuevas), no un ajuste de la mezcla con el mercado.")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "skill_model_underdog_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"\nGuardado detalle combinado -> {out_path}")


if __name__ == "__main__":
    run()