"""
Fase 8, "Proximos pasos" punto 3: La Liga es la unica de las 4 ligas
europeas que no alcanza la meta de Tier 1 (>=80% de acierto real) en el
umbral >=80% -- 78.57% real sobre n=28 (tier1_probability_validation.py,
ver roadmap). La pregunta pendiente, nunca resuelta: ¿es ruido de muestra
chica (con n=28, una sola temporada rara puede mover el numero varios
puntos) o una señal real de que el modelo v4 no captura bien la dinamica
de La Liga en su segmento de mas alta confianza?

Con la misma disciplina que ya se aplico a Serie A/Bundesliga (descomponer
en vez de aceptar un numero agregado): n=28 es lo bastante chico como para
listar CADA caso individual -- temporada, lado apostado, equipo, cuota,
probabilidad predicha, resultado real -- y mirar si los fallos (deberian
ser ~6 de 28 para llegar a 78.57%) estan dispersos entre temporadas/equipos
(consistente con ruido) o concentrados en un patron reconocible (temporada
especifica, lado especifico, equipo especifico -- señal real).

Fuente: mismas predicciones OOS ya usadas en tier1_probability_validation.py
(reconstruye desde 'model_predictions_oos_walkforward_v4.csv' de LALIGA,
sin asumir funciones privadas no confirmadas de ese script). Un caso
Tier 1 es cualquier fila donde el resultado con mayor probabilidad
(home/draw/away, cualquiera de los 3, no solo el de mayor edge de apuesta
-- a diferencia de los scripts de seleccion de apuestas anteriores, este es
sobre PROBABILIDAD del modelo, la definicion original de Tier 1) supera 80%.

Salida: data/runs/laliga_tier1_breakdown_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

LEAGUE_KEY = "LALIGA"
THRESHOLD = 0.80

SIDES = [
    ("home", "blend_prob_home", "H"),
    ("draw", "blend_prob_draw", "D"),
    ("away", "blend_prob_away", "A"),
]


def _load_predictions() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / LEAGUE_KEY / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        print(f"[ERROR] No existe {path}.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    has_blend = df["blend_prob_home"].notna() if "blend_prob_home" in df.columns else pd.Series(False, index=df.index)
    df = df.loc[has_blend].copy()
    return df.sort_values("Date").reset_index(drop=True)


def _tier1_cases(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        best = None
        for side_name, prob_col, ftr_code in SIDES:
            prob = row[prob_col]
            if pd.isna(prob):
                continue
            if best is None or prob > best["prob"]:
                best = {"side": side_name, "prob": prob, "ftr_code": ftr_code}
        if best is not None and best["prob"] >= THRESHOLD:
            record = {
                "Date": row["Date"].date(), "fold_test_season": str(row["fold_test_season"]),
                "predicted_side": best["side"], "predicted_prob": best["prob"],
                "actual_FTR": row["FTR"], "acerto": row["FTR"] == best["ftr_code"],
            }
            if "HomeTeam" in df.columns:
                record["HomeTeam"] = row.get("HomeTeam")
            if "AwayTeam" in df.columns:
                record["AwayTeam"] = row.get("AwayTeam")
            records.append(record)
    return pd.DataFrame(records)


def run() -> None:
    print(f"\n=== {LEAGUE_KEY} -- desglose caso por caso del umbral Tier 1 (>={THRESHOLD:.0%}) ===")
    df_eval = _load_predictions()
    if df_eval.empty:
        return

    cases = _tier1_cases(df_eval)
    if cases.empty:
        print("[AVISO] Cero casos >=80%.")
        return

    n = len(cases)
    n_correct = cases["acerto"].sum()
    n_wrong = n - n_correct
    print(f"Total de casos: {n}  |  Aciertos: {n_correct}  |  Fallos: {n_wrong}  |  "
          f"Acierto real: {n_correct / n:.2%}\n")

    cols_to_show = ["Date", "fold_test_season", "predicted_side", "predicted_prob", "actual_FTR", "acerto"]
    if "HomeTeam" in cases.columns:
        cols_to_show = ["Date", "fold_test_season", "HomeTeam", "AwayTeam",
                         "predicted_side", "predicted_prob", "actual_FTR", "acerto"]

    print("Todos los casos (ordenados por temporada, luego fecha):")
    cases_sorted = cases.sort_values(["fold_test_season", "Date"])
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(cases_sorted[cols_to_show].to_string(index=False))

    print("\n--- Fallos únicamente (los que arrastran el 78.57% por debajo de la meta) ---")
    wrong = cases_sorted[~cases_sorted["acerto"]]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(wrong[cols_to_show].to_string(index=False))

    print("\n--- Distribución de fallos por temporada ---")
    by_season = cases.groupby("fold_test_season").agg(n=("acerto", "size"), aciertos=("acerto", "sum"))
    by_season["fallos"] = by_season["n"] - by_season["aciertos"]
    by_season["acierto_pct"] = by_season["aciertos"] / by_season["n"]
    print(by_season.to_string())

    print("\n--- Distribución de fallos por lado apostado ---")
    by_side = cases.groupby("predicted_side").agg(n=("acerto", "size"), aciertos=("acerto", "sum"))
    by_side["fallos"] = by_side["n"] - by_side["aciertos"]
    print(by_side.to_string())

    if "HomeTeam" in cases.columns:
        print("\n--- Equipos involucrados en los fallos (home o away) ---")
        wrong_teams = pd.concat([wrong["HomeTeam"], wrong["AwayTeam"]]).value_counts()
        print(wrong_teams.to_string())

    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "laliga_tier1_breakdown_check.csv"
    cases_sorted.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: si los 6 (aprox.) fallos estan repartidos entre varias temporadas y lados "
          "distintos sin patron reconocible, es consistente con ruido de muestra chica (n=28 es "
          "poco para un umbral tan exigente) -- no hay que sobre-reaccionar. Si se concentran en una "
          "temporada, un lado (ej. casi todos 'draw', history conocida de ser el resultado mas dificil "
          "de predecir) o un puñado de equipos, es una señal real y accionable (ej. excluir draws del "
          "Tier 1 de La Liga, o esa temporada especifica).")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que los chequeos anteriores de esta cadena.")


if __name__ == "__main__":
    run()