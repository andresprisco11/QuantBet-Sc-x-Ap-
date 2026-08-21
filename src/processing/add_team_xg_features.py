"""
Fase 10 (futbol): variable de forma reciente basada en xG (expected goals),
construida ENCIMA de v4 (mismo criterio que Elo/v7 y corners/v6 -- construir
siempre sobre la mejor base confirmada, NO sobre el ultimo intento) para dar
una nueva oportunidad real a la hipotesis (c) que quedo pausada en Fase 8:
que el techo predictivo de Serie A/Bundesliga en particular puede deberse a
falta de informacion de CALIDAD de las ocasiones generadas (xG), no a falta
de mas recombinacion de goles/tiros/corners -- eso ya se probo (v5, v6) y
fallo, y Elo (v7) tampoco aporto nada nuevo porque C(team)/C(opponent) ya
capturan la fuerza relativa estatica.

A diferencia de home_recent_st_diff (tiros al arco, disponible en TODO el
historico), la cobertura real de xG via TheStatsAPI es PARCIAL -- solo
temporadas >=2022/23 (confirmado en Fase 10, merge_thestatsapi_xg.py). Esto
es una limitacion real, no oculta: para partidos de temporadas viejas
(2021-2223) simplemente no hay xG historico para construir la forma
reciente -- se usa el mismo prior neutro (0.0) que ya usa el proyecto para
"no tenemos informacion todavia" (PROMOTED_TEAM, primer partido de un
equipo en add_team_form_features.py). El efecto practico: el feature
empieza a tener senal real recien cuando el equipo acumula partidos reales
con xG en su ventana de 5 partidos anteriores -- progresivamente, no de
golpe. backtest_v8.py mide esto explicitamente separando el subconjunto de
folds con cobertura real de los que no.

REQUIERE 'matches_clean_with_xg.csv' por liga (correr primero
'python -m src.processing.merge_thestatsapi_xg --all'). Este script LEE de
ahi pero ESCRIBE las 2 columnas nuevas en 'matches_clean.csv' (el archivo
canonico que leen todos los modelos) -- mismo criterio que
add_team_elo_features.py, para que exista una sola fuente de verdad y no
haya que tocar ningun backtest existente para que sepa donde buscar el
feature.

Ventana=5 partidos anteriores (shift(1) antes del rolling, cero fuga --
mismo patron que add_team_form_features.py), diferencial neto
(xg_for - xg_against), NO separado en ataque/defensa (v5 ya probo que
separar la senal de tiros no ayudo, no se repite sin evidencia a favor
aca tampoco).

Idempotente de verdad -- mismo criterio que add_team_elo_features.py: si ya
existen las columnas de una corrida anterior, se descartan ANTES de
recalcular.

Uso: python -m src.processing.add_team_xg_features
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR

ROLLING_WINDOW = 5
NEW_COLS = ["home_recent_xg_diff", "away_recent_xg_diff"]


def _rolling_xg_diff(df_with_xg: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Construye la vista larga (una fila por equipo por partido) y calcula el
    diferencial neto de xG de los ultimos N partidos ANTERIORES (shift(1),
    cero fuga). El promedio movil de pandas ya ignora los NaN dentro de la
    ventana (no fuerza min_periods sobre filas sin xG) -- min_periods=1 solo
    exige al menos UN partido anterior con xG real para no ser NaN puro.
    """
    df = df_with_xg.reset_index(drop=True).copy()
    df["match_id"] = df.index

    home = pd.DataFrame({
        "match_id": df["match_id"], "Date": df["Date"], "team": df["HomeTeam"],
        "xg_for": df["home_xg"], "xg_against": df["away_xg"],
    })
    away = pd.DataFrame({
        "match_id": df["match_id"], "Date": df["Date"], "team": df["AwayTeam"],
        "xg_for": df["away_xg"], "xg_against": df["home_xg"],
    })
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["xg_diff"] = long_df["xg_for"] - long_df["xg_against"]
    long_df = long_df.sort_values(["team", "Date", "match_id"])

    long_df["recent_xg_diff"] = long_df.groupby("team")["xg_diff"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )

    home_side = long_df.merge(
        df[["match_id", "HomeTeam"]], left_on=["match_id", "team"], right_on=["match_id", "HomeTeam"], how="inner"
    )[["match_id", "recent_xg_diff"]].rename(columns={"recent_xg_diff": "home_recent_xg_diff"})

    away_side = long_df.merge(
        df[["match_id", "AwayTeam"]], left_on=["match_id", "team"], right_on=["match_id", "AwayTeam"], how="inner"
    )[["match_id", "recent_xg_diff"]].rename(columns={"recent_xg_diff": "away_recent_xg_diff"})

    out = df[["match_id", "Date", "HomeTeam", "AwayTeam"]].merge(home_side, on="match_id").merge(away_side, on="match_id")
    return out


def run(league_key: str) -> None:
    xg_path = PROCESSED_DATA_DIR / league_key / "matches_clean_with_xg.csv"
    clean_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not xg_path.exists():
        print(f"[SKIP] {league_key}: no existe {xg_path} -- corre primero "
              f"'python -m src.processing.merge_thestatsapi_xg --all'.")
        return
    if not clean_path.exists():
        print(f"[SKIP] {league_key}: no existe {clean_path}.")
        return

    df_xg = pd.read_csv(xg_path)
    df_xg["Date"] = pd.to_datetime(df_xg["Date"])

    features = _rolling_xg_diff(df_xg)
    n_with_real_xg = int((df_xg["home_xg"].notna() & df_xg["away_xg"].notna()).sum())
    print(f"\n=== {league_key} ===")
    print(f"Partidos con xG real (para construir la ventana de forma reciente): {n_with_real_xg}/{len(df_xg)}")

    df_clean = pd.read_csv(clean_path)
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])

    # Idempotencia real: ver docstring del modulo.
    existing = [c for c in NEW_COLS if c in df_clean.columns]
    if existing:
        df_clean = df_clean.drop(columns=existing)

    n_before = len(df_clean)
    merged = df_clean.merge(
        features[["Date", "HomeTeam", "AwayTeam", "home_recent_xg_diff", "away_recent_xg_diff"]],
        on=["Date", "HomeTeam", "AwayTeam"], how="left",
    )
    if len(merged) != n_before:
        print(f"[AVISO] {league_key}: el merge cambio el numero de filas ({n_before} -> {len(merged)}) -- "
              f"revisar si hay fechas/equipos duplicados antes de confiar en el resultado.")

    for c in NEW_COLS:
        merged[c] = merged[c].fillna(0.0)

    n_nonzero = int(((merged["home_recent_xg_diff"] != 0.0) | (merged["away_recent_xg_diff"] != 0.0)).sum())
    print(f"Partidos con senal real de xG reciente (no en el prior neutro 0.0): {n_nonzero}/{len(merged)} "
          f"({n_nonzero/len(merged):.1%})")

    merged.to_csv(clean_path, index=False)
    print(f"Guardado (columnas home_recent_xg_diff/away_recent_xg_diff agregadas) -> {clean_path}")


if __name__ == "__main__":
    for league_key in LEAGUES:
        run(league_key)
