"""
Validacion empirica de parlays Tier 1 (2-3 piernas, picks de >=80% de
confianza del modelo). tier1_probability_validation.py ya confirmo que,
partido por partido, el modelo acierta ~90% cuando dice estar >=80%
seguro -- pero Tier 1 no apuesta partidos sueltos, apuesta COMBOS de 2-3
piernas de la MISMA jornada. La pregunta que este script responde, con
datos historicos reales (no supuestos): cuando se arman esos combos, la
probabilidad conjunta real de acertar TODAS las piernas coincide con la
probabilidad ingenua (multiplicar las probabilidades individuales), o hay
correlacion que la rompe?

CORRECCION vs. la primera version de este script: agrupar por FECHA
EXACTA fue un error de diseno, no un hallazgo real -- dio solo 3 combos
de 2 piernas y 0 de 3 en las 6 temporadas completas, muestra inutil. Un
apostador armando su parlay de Tier 1 para el fin de semana no necesita
que los partidos sean el mismo dia calendario -- junta sabado y domingo
de la misma jornada de la EPL. Se agrupa ahora por SEMANA ISO
(año-semana), que captura una jornada completa de fin de semana (y
tambien partidos entre semana reprogramados dentro de la misma semana).
Este es un cambio de diseno mas realista, no un ajuste para forzar mas
datos -- asi es como se arma un parlay de Tier 1 en la practica.

METODOLOGIA (resto sin cambios):
1. Pick propio del modelo por partido: el resultado (home/draw/away) con
   mayor blend_prob, SIN filtrar por edge sobre el mercado (Tier 1 se
   define por confianza del modelo, no por valor vs. mercado).
2. Filtro a partidos donde ese pick alcanza el umbral Tier 1 (0.80).
3. Agrupacion por semana ISO (año-semana).
4. Para cada grupo con 2+ picks calificados, se arman TODAS las
   combinaciones posibles de 2 piernas (y de 3, si hay 3+) -- probabilidad
   ingenua = producto de las probabilidades individuales; resultado real =
   si TODAS las piernas de esa combinacion acertaron.
5. Se compara probabilidad ingenua promedio vs. tasa de acierto real
   observada -- la brecha es la señal de correlacion (o su ausencia).

No entrena nada nuevo: reutiliza 'model_predictions_oos_walkforward_v4.csv'.
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

TIER1_THRESHOLD = 0.80
SIDES = [("home", "blend_prob_home", "H"), ("draw", "blend_prob_draw", "D"), ("away", "blend_prob_away", "A")]


def _own_best_pick(row) -> dict:
    """El resultado con mayor blend_prob para este partido, sin filtrar por edge."""
    best_side, best_prob, best_code = None, -1.0, None
    for side_name, prob_col, ftr_code in SIDES:
        prob = row[prob_col]
        if pd.notna(prob) and prob > best_prob:
            best_side, best_prob, best_code = side_name, prob, ftr_code
    return {"side": best_side, "prob": best_prob, "won": row["FTR"] == best_code}


def _iso_week_key(date: pd.Timestamp) -> str:
    iso = date.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _evaluate_combos(picks_by_group: dict, leg_count: int) -> pd.DataFrame:
    records = []
    for group_key, picks in picks_by_group.items():
        if len(picks) < leg_count:
            continue
        for combo in combinations(picks, leg_count):
            naive_prob = 1.0
            all_won = True
            for pick in combo:
                naive_prob *= pick["prob"]
                all_won = all_won and pick["won"]
            records.append({"group": group_key, "naive_prob": naive_prob, "actual_hit": all_won})
    return pd.DataFrame(records)


def run():
    path = PROCESSED_DATA_DIR / "EPL" / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre 'python -m src.models.backtest_v4' primero."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df_eval = df.loc[has_blend].copy()

    picks_by_week = {}
    n_qualifying = 0
    for _, row in df_eval.iterrows():
        pick = _own_best_pick(row)
        if pick["prob"] >= TIER1_THRESHOLD:
            n_qualifying += 1
            week_key = _iso_week_key(row["Date"])
            picks_by_week.setdefault(week_key, []).append(pick)

    print(f"Partidos evaluados: {len(df_eval)}")
    print(f"Picks propios del modelo con confianza >= {TIER1_THRESHOLD:.0%}: {n_qualifying}")
    weeks_with_2plus = sum(1 for picks in picks_by_week.values() if len(picks) >= 2)
    weeks_with_3plus = sum(1 for picks in picks_by_week.values() if len(picks) >= 3)
    print(f"Semanas ISO con 2+ picks calificados: {weeks_with_2plus}")
    print(f"Semanas ISO con 3+ picks calificados: {weeks_with_3plus}")

    for leg_count in (2, 3):
        combos = _evaluate_combos(picks_by_week, leg_count)
        print(f"\n=== Parlays de {leg_count} piernas (picks Tier 1, >= {TIER1_THRESHOLD:.0%}, agrupados por semana ISO) ===")
        if combos.empty:
            print(f"[AVISO] Cero combinaciones de {leg_count} piernas disponibles todavia -- hace falta "
                  f"mas historial (mas temporadas o mas ligas) para validar Tier 1 de {leg_count} piernas.")
            continue
        n = len(combos)
        naive_mean = combos["naive_prob"].mean()
        actual_hit_rate = combos["actual_hit"].mean()
        gap = actual_hit_rate - naive_mean
        print(f"Combinaciones evaluadas: {n}")
        print(f"Probabilidad ingenua promedio (producto de probabilidades individuales): {naive_mean:.4f}")
        print(f"Tasa de acierto REAL observada (todas las piernas correctas): {actual_hit_rate:.4f}")
        print(f"Gap (real - ingenua): {gap:+.4f}")
        if n < 30:
            print(f"[AVISO] Solo {n} combinaciones -- muestra demasiado chica para sacar conclusiones "
                  f"firmes, tratar como señal preliminar, no como resultado confirmado.")


if __name__ == "__main__":
    run()