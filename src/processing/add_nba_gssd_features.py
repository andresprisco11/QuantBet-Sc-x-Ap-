"""
Fase 10 (NBA) -- feature GSSD (Game Scores Standard Deviation), adaptado del
modelo GSSD descrito en el libro de Andrew Mack (`Statistical Sports Models
in Excel`, base teorica del proyecto). El libro lo describe como
"frequently among the best performing generalized sport models" -- usa el
promedio de puntos anotados/recibidos de cada equipo, separado por local y
visitante (PFH, PAH, PFA, PAA), como regresores de OLS para el marcador
esperado.

**Adaptacion necesaria, documentada explicita**: la version original del
libro (hoja de Excel) calcula PFH/PAH/PFA/PAA como promedio de TODA la
temporada (pasado Y futuro) y ajusta una sola regresion retrospectiva sobre
esa temporada completa -- eso es un modelo DESCRIPTIVO, no sirve para
prediccion prospectiva walk-forward (usaria partidos futuros para predecir
partidos pasados de la misma temporada, fuga de datos). Aca se recalcula
como una ventana MOVIL de los ultimos N partidos de local/visitante de cada
equipo, con `shift(1)` para excluir explicitamente el partido actual --
misma disciplina anti-fuga que Elo y las features de descanso.

**Definicion de las 4 variables** (ventana N=10, sin reiniciar por
temporada -- igual que Elo, el equipo arrastra su forma reciente entre
temporadas en vez de resetear a cero cada octubre):
- `home_off_l10`: promedio de puntos ANOTADOS por el equipo en sus ultimos
  10 partidos DE LOCAL (no cuenta partidos de visitante).
- `home_def_l10`: promedio de puntos RECIBIDOS por el equipo en sus
  ultimos 10 partidos de local.
- `away_off_l10` / `away_def_l10`: simetrico, usando los ultimos 10
  partidos DE VISITANTE del equipo.

Se exige un minimo de 3 partidos previos (`min_periods=3`) antes de
producir un valor -- con menos de 3 partidos el promedio es demasiado
ruidoso para ser util, se deja NaN explicito en vez de un numero poco
confiable.

**Screening antes de construir esto**: confirmado con OLS full-sample,
controlando por elo_diff + b2b_diff + 3in4_diff + home_is_denver (ya
confirmados) -- las 4 variables GSSD son significativas Y con el signo
correcto: home_off_l10 (+, mas ataque local = mas margen), home_def_l10
(-, mas puntos recibidos de local = menos margen), away_off_l10 (-, mas
ataque visitante = menos margen para el local), away_def_l10 (+, mas
puntos recibidos de visitante = mas margen para el local). R^2 sube de
0.177 (v3) a 0.180 con estas 4 variables agregadas.

Requiere que games_clean.csv ya exista (`clean_nba_data.py`). Agrega
home_off_l10/home_def_l10/away_off_l10/away_def_l10, idempotente (mismo
criterio drop-antes-de-recalcular que el resto del proyecto).

Uso: python -m src.processing.add_nba_gssd_features
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"
NEW_COLS = ["home_off_l10", "home_def_l10", "away_off_l10", "away_def_l10"]
WINDOW = 10
MIN_PERIODS = 3


def _trailing_home_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio movil (ultimos WINDOW partidos DE LOCAL de cada equipo) de
    puntos anotados/recibidos, excluyendo el partido actual via shift(1)."""
    rows = df[["game_date", "home_team", "home_pts", "away_pts"]].rename(
        columns={"home_team": "team", "home_pts": "pts_for", "away_pts": "pts_against"}
    ).sort_values(["team", "game_date"])

    off_shift = rows.groupby("team")["pts_for"].shift(1)
    def_shift = rows.groupby("team")["pts_against"].shift(1)
    rows["home_off_l10"] = off_shift.groupby(rows["team"]).transform(
        lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )
    rows["home_def_l10"] = def_shift.groupby(rows["team"]).transform(
        lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )
    return rows[["game_date", "team", "home_off_l10", "home_def_l10"]]


def _trailing_away_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Simetrico a `_trailing_home_stats` pero para partidos DE VISITANTE."""
    rows = df[["game_date", "away_team", "away_pts", "home_pts"]].rename(
        columns={"away_team": "team", "away_pts": "pts_for", "home_pts": "pts_against"}
    ).sort_values(["team", "game_date"])

    off_shift = rows.groupby("team")["pts_for"].shift(1)
    def_shift = rows.groupby("team")["pts_against"].shift(1)
    rows["away_off_l10"] = off_shift.groupby(rows["team"]).transform(
        lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )
    rows["away_def_l10"] = def_shift.groupby(rows["team"]).transform(
        lambda s: s.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )
    return rows[["game_date", "team", "away_off_l10", "away_def_l10"]]


def run() -> None:
    if not DATA_PATH.exists():
        print(f"[SKIP] No existe {DATA_PATH} -- corre 'python -m src.processing.clean_nba_data' primero.")
        return

    df = pd.read_csv(DATA_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])

    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    home_stats = _trailing_home_stats(df)
    away_stats = _trailing_away_stats(df)

    df = df.merge(
        home_stats.rename(columns={"team": "home_team"}), on=["game_date", "home_team"], how="left"
    )
    df = df.merge(
        away_stats.rename(columns={"team": "away_team"}), on=["game_date", "away_team"], how="left"
    )

    print(f"Partidos procesados: {len(df)}")
    for col in NEW_COLS:
        n_missing = int(df[col].isna().sum())
        print(f"Sin {col} calculable (menos de {MIN_PERIODS} partidos previos en ese contexto): {n_missing}")

    # Chequeo de sanidad OBJETIVO -- el promedio historico de puntos por
    # partido en NBA moderna ronda 100-115, un valor muy fuera de ese rango
    # (ej. <60 o >160) en la mediana seria señal de un bug real de datos.
    print(f"\nChequeo de sanidad -- mediana de home_off_l10: {df['home_off_l10'].median():.1f} "
          f"(deberia estar en el rango tipico de puntos de NBA, ~95-120)")
    print(f"Mediana de away_off_l10: {df['away_off_l10'].median():.1f}")

    df.to_csv(DATA_PATH, index=False)
    print(f"\nGuardado (columnas {', '.join(NEW_COLS)} agregadas) -> {DATA_PATH}")


if __name__ == "__main__":
    run()
