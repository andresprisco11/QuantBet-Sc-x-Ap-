"""
Fase 10 (NBA) -- features GSSD (Game Scores Standard Deviation), adaptado
del modelo GSSD descrito en el libro de Andrew Mack (`Statistical Sports
Models in Excel`, base teorica del proyecto). El libro lo describe como
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
como una ventana de los ultimos partidos de local/visitante de cada equipo,
con `shift(1)` para excluir explicitamente el partido actual -- misma
disciplina anti-fuga que Elo y las features de descanso.

**Dos versiones, agregadas en dos rondas (la segunda motivada por el
resultado real de la primera)**:

1. **Promedio movil simple, ventana N=10** (`home_off_l10`/`home_def_l10`/
   `away_off_l10`/`away_def_l10`) -- version original, usada en v4.
   Confirmada con OLS full-sample: las 4 significativas y con signo
   correcto, R^2 sube de 0.177 (v3) a 0.180.
2. **NUEVO -- promedio exponencial (EWM), span=15**
   (`home_off_ewm`/`home_def_ewm`/`away_off_ewm`/`away_def_ewm`) -- motivado
   por la pregunta real de si ponderar mas los partidos recientes (en vez
   de pesar igual los ultimos 10) mejora la señal. Se probo walk-forward
   completo con distintas ventanas/spans ANTES de elegir uno:
   - Promedio movil simple: N=5 (gap 0.003341), N=10 (0.003329), N=15
     (0.003348), N=20 (0.003368) -- N=10 ya era el mejor de la familia
     simple, confirmando que v4 no se quedo corto ahi.
   - EWM: span=5 (0.003344), span=10 (0.003267), **span=15 (0.003251,
     MEJOR de todos los probados)**, span=20 (0.003261), span=25
     (0.003287), span=30 (0.003321), span=40 (0.003391), span=60
     (0.003479) -- EWM le gana a la ventana simple en TODOS los spans
     probados, con un optimo claro en span=15 (no en un extremo del rango
     probado, confirma que no es un artefacto de borde). Metrica: gap
     blend vs. mercado, walk-forward completo (mismo criterio que todas
     las decisiones anteriores del proyecto).
   Conclusion: ponderar mas lo reciente SI aporta señal real sobre el
   promedio simple -- consistente con que Elo tambien pondera mas lo
   reciente (K-factor) en vez de tratar todo el historico por igual.

**Se mantienen AMBAS versiones en el archivo** (no se borra `_l10`, se
agrega `_ewm` aparte) -- `nba_margin_model_v4.py` sigue dependiendo de las
columnas `_l10` y no se quiere romper ese archivo ya confirmado. El modelo
`v5` usa las nuevas columnas `_ewm`.

**Definicion de EWM(span=15) sin fuga**: `pandas.Series.ewm(span=15)` sobre
la serie YA desplazada un partido (`shift(1)`) -- el peso de un partido a
`k` partidos de distancia decae geometricamente, span=15 equivale
aproximadamente a un promedio con "vida media" de ~10 partidos (formula de
pandas: alpha=2/(span+1)). Igual que la version simple, `min_periods=3` y
sin reiniciar por temporada.

Requiere que games_clean.csv ya exista (`clean_nba_data.py`). Agrega las 8
columnas (`_l10` + `_ewm`), idempotente (mismo criterio
drop-antes-de-recalcular que el resto del proyecto).

Uso: python -m src.processing.add_nba_gssd_features
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "NBA" / "games_clean.csv"
NEW_COLS = [
    "home_off_l10", "home_def_l10", "away_off_l10", "away_def_l10",
    "home_off_ewm", "home_def_ewm", "away_off_ewm", "away_def_ewm",
]
WINDOW = 10
EWM_SPAN = 15
MIN_PERIODS = 3


def _trailing_home_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio movil simple (ultimos WINDOW partidos DE LOCAL) Y promedio
    exponencial (span=EWM_SPAN) de puntos anotados/recibidos, excluyendo el
    partido actual via shift(1)."""
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
    rows["home_off_ewm"] = off_shift.groupby(rows["team"]).transform(
        lambda s: s.ewm(span=EWM_SPAN, min_periods=MIN_PERIODS).mean()
    )
    rows["home_def_ewm"] = def_shift.groupby(rows["team"]).transform(
        lambda s: s.ewm(span=EWM_SPAN, min_periods=MIN_PERIODS).mean()
    )
    return rows[["game_date", "team", "home_off_l10", "home_def_l10", "home_off_ewm", "home_def_ewm"]]


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
    rows["away_off_ewm"] = off_shift.groupby(rows["team"]).transform(
        lambda s: s.ewm(span=EWM_SPAN, min_periods=MIN_PERIODS).mean()
    )
    rows["away_def_ewm"] = def_shift.groupby(rows["team"]).transform(
        lambda s: s.ewm(span=EWM_SPAN, min_periods=MIN_PERIODS).mean()
    )
    return rows[["game_date", "team", "away_off_l10", "away_def_l10", "away_off_ewm", "away_def_ewm"]]


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
    print(f"Mediana de home_off_ewm: {df['home_off_ewm'].median():.1f} (deberia ser similar a home_off_l10)")
    print(f"Mediana de away_off_l10: {df['away_off_l10'].median():.1f}")
    print(f"Mediana de away_off_ewm: {df['away_off_ewm'].median():.1f}")

    df.to_csv(DATA_PATH, index=False)
    print(f"\nGuardado (columnas {', '.join(NEW_COLS)} agregadas) -> {DATA_PATH}")


if __name__ == "__main__":
    run()
