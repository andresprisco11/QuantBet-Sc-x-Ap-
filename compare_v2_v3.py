"""
Diagnostico puntual (no es parte del pipeline permanente, no necesita tracking):
compara las probabilidades de v2 vs v3 partido por partido, con mas precision
que los 4 decimales que se imprimen en pantalla durante el backtest, para
confirmar si el ajuste de Dixon-Coles realmente esta cambiando algo o si el
resultado identico impreso es sospechoso de un bug silencioso.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

v2 = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v2.csv")
v3 = pd.read_csv(PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v3.csv")

print(f"Filas v2: {len(v2)} | Filas v3: {len(v3)}")

for col in ["model_prob_home", "model_prob_draw", "model_prob_away"]:
    diff = (v2[col] - v3[col]).abs()
    print(f"\n{col}:")
    print(f"  diferencia maxima:    {diff.max():.10f}")
    print(f"  diferencia promedio:  {diff.mean():.10f}")
    print(f"  partidos con diferencia > 1e-6: {(diff > 1e-6).sum()} de {len(diff)}")

# Comparacion directa de un puñado de partidos con marcador bajo real (donde
# el ajuste de Dixon-Coles deberia notarse mas), para inspeccion visual.
if "FTHG" in v2.columns and "FTAG" in v2.columns:
    low_score = v2[(v2["FTHG"] <= 1) & (v2["FTAG"] <= 1)].head(5)
    print("\n--- Muestra de partidos con marcador bajo real (FTHG<=1, FTAG<=1) ---")
    for idx in low_score.index:
        print(f"  {v2.loc[idx, 'HomeTeam']} vs {v2.loc[idx, 'AwayTeam']} "
              f"({int(v2.loc[idx, 'FTHG'])}-{int(v2.loc[idx, 'FTAG'])}): "
              f"v2 prob_home={v2.loc[idx, 'model_prob_home']:.6f} vs "
              f"v3 prob_home={v3.loc[idx, 'model_prob_home']:.6f} "
              f"(rho={v3.loc[idx, 'fold_rho'] if 'fold_rho' in v3.columns else 'N/A'})")