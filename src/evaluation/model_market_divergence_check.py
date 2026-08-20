"""
Motivado por la aclaracion del usuario 2026-08-20 (la MISMA sesion que trajo
el gate de 99%): "yo no pienso solo apostarle a cuotas favoritisimas...
busco es la eficiencia brutal de las apuestas, la cantidad de perdidas nula
practicamente ese es el objetivo real". `extreme_confidence_check.py` (recien
corrido) confirmo el riesgo exacto que esto anticipaba: los unicos casos con
>=98-99% de acierto real son casi exclusivamente favoritos extremos con cuota
promedio 1.00-1.02 -- tecnicamente "casi sin perdidas", pero economicamente
vacios (cuota 1.00 = cero ganancia neta incluso acertando). Si el objetivo
real es "perdidas practicamente nulas" en TODO el espectro de cuotas
(incluyendo underdogs, no solo favoritos), hace falta separar dos cosas que
`extreme_confidence_check.py` mezcla sin querer: confianza alta que
simplemente REPITE lo que el mercado ya piensa (favorito obvio, sin
informacion nueva) vs. confianza alta que DIVERGE del mercado (el modelo ve
algo que el mercado no ve tan claro -- el terreno donde vive la meta original
del proyecto, "detectar cuando el chico le gana al grande", nunca resuelta).

Esto no es una prediccion nueva ni un modelo nuevo -- es un corte distinto
sobre las MISMAS predicciones ya generadas (igual que
`extreme_confidence_check.py`), separando cada apuesta de alta confianza en:
  - "Alineado con mercado": el mercado tambien considera favorito a ese
    mismo lado (probabilidad implicita del mercado >=50% en ese lado).
  - "Divergente del mercado": el modelo esta muy seguro pero el MERCADO
    considera ese lado el UNDERDOG (probabilidad implicita del mercado
    <50%) -- la firma exacta de "el modelo ve una victoria del chico que el
    mercado no ve venir". Se reporta cuan seguido ocurre esto y, cuando
    ocurre, si historicamente acierta o es ruido/winner's curse.

Fuentes reutilizadas, sin recalcular nada:
- Futbol (4 ligas): 'model_predictions_oos_walkforward_v4.csv' por liga
  (backtest_v4.py) -- ya trae blend_prob_*, pinnacle_close_prob_* (proxy de
  probabilidad de mercado, mismo benchmark que ya usa backtest_v4.py) y
  PSH/PSD/PSA (precio de ejecucion real, mismo que usa economic_backtest.py).
- MLS: 'model_predictions_oos_walkforward_mls.csv' -- mismas columnas, pero
  el precio de ejecucion es PSCH/PSCD/PSCA (cierre), no PSH/PSD/PSA (MLS no
  tiene apertura -- mismo criterio que economic_backtest_mls.py).
- Tenis (ATP/WTA): 'predictions_v1.csv' -- ya trae blend_prob_player1 y
  market_prob_player1 por separado, no hace falta reconstruir nada.

Salida: imprime, por mercado y en TOTAL combinado, para cada umbral de
confianza del modelo (80/85/90%), el desglose alineado vs. divergente
(n_casos, acierto_real, cuota_promedio, probabilidad_mercado_promedio).
Guarda el detalle combinado en 'data/runs/model_market_divergence_check.csv'.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

THRESHOLDS = [0.80, 0.85, 0.90]
MLS_KEY = "MLS"
TENNIS_TOURS = ["ATP", "WTA"]

FOOTBALL_SIDES = [
    ("home", "PSH", "blend_prob_home", "pinnacle_close_prob_home", "H"),
    ("draw", "PSD", "blend_prob_draw", "pinnacle_close_prob_draw", "D"),
    ("away", "PSA", "blend_prob_away", "pinnacle_close_prob_away", "A"),
]
MLS_SIDES = [
    ("home", "PSCH", "blend_prob_home", "pinnacle_close_prob_home", "H"),
    ("draw", "PSCD", "blend_prob_draw", "pinnacle_close_prob_draw", "D"),
    ("away", "PSCA", "blend_prob_away", "pinnacle_close_prob_away", "A"),
]


def _unroll_football(df: pd.DataFrame, market: str, sides) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for side_name, odds_col, blend_col, market_col, ftr_code in sides:
            blend_prob = row.get(blend_col)
            odds = row.get(odds_col)
            market_prob = row.get(market_col)
            if pd.isna(blend_prob) or pd.isna(odds) or pd.isna(market_prob):
                continue
            records.append({
                "market": market,
                "predicted_prob": blend_prob,
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
    has_market = df["blend_prob_player1"].notna() if "blend_prob_player1" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_market].copy()
    if df.empty:
        return pd.DataFrame()

    p1_favored = df["blend_prob_player1"] >= 0.5
    predicted_prob = df["blend_prob_player1"].where(p1_favored, 1.0 - df["blend_prob_player1"])
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

    print(f"\n=== Lectura directa: cuan seguido el modelo esta MUY seguro de un lado que el "
          f"mercado considera underdog, y si eso historicamente acierta ===")
    for threshold in THRESHOLDS:
        row = combined_table.loc[(f">={threshold:.0%}", "divergente_underdog")]
        n = int(row["n_casos"]) if pd.notna(row["n_casos"]) else 0
        if n == 0:
            print(f"  >={threshold:.0%}: 0 casos divergentes -- el modelo nunca esta esta seguro de "
                  f"un lado que el mercado considere underdog. Hasta ahora, la confianza alta SIEMPRE "
                  f"coincide con el favorito del mercado.")
            continue
        print(f"  >={threshold:.0%}: n={n} casos divergentes (posibles 'el chico le gana al grande'), "
              f"acierto real={row['acierto_real']:.2%} ({int(row['perdidas'])} perdidas), "
              f"prob. mercado promedio en ese lado={row['prob_mercado_promedio']:.2%}, "
              f"cuota promedio={row['cuota_promedio']:.2f}.")

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "model_market_divergence_check.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"\nGuardado detalle combinado -> {out_path}")


if __name__ == "__main__":
    run()