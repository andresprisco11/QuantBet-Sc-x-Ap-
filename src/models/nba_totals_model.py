"""
Fase 10 (NBA) -- primer modelo de TOTALES (over/under) de NBA. Hasta ahora
el proyecto solo tenia moneyline (P(gana local), via margen de puntos).
Este modelo predice el TOTAL de puntos del partido (home_pts + away_pts),
abriendo una pata de mercado completamente nueva para piernas de parlay
(Tier 1/Tier 2 ya no dependen solo de moneyline).

**Arquitectura**: mismo patron Normal(mu, sigma) que el modelo de margen,
pero sobre `game_total` en vez de `point_margin`. Reutiliza las mismas
variables GSSD (EWM, ver `add_nba_gssd_features.py`) que ya probaron ser
la señal mas fuerte para margen -- tiene sentido teorico directo: el total
de un partido depende de la capacidad ofensiva/defensiva de ambos equipos,
exactamente lo que GSSD mide.

**Screening antes de construir esto** (OLS full-sample): partiendo de las
mismas variables candidatas usadas en el modelo de margen (GSSD, Elo,
calendario, Denver), se descartaron las que NO fueron significativas para
TOTALES especificamente -- una variable puede importar para margen y no
para el ritmo/volumen de puntos, son preguntas distintas:
- `home_off_ewm`, `home_def_ewm`, `away_off_ewm`, `away_def_ewm`: los 4
  altamente significativos (p<0.0001), coeficientes positivos y de
  magnitud similar (~0.47-0.58) -- logico, mas ataque O mas puntos
  recibidos historicamente (indicador de que el equipo juega partidos de
  ritmo alto) suben el total esperado.
- `elo_sum` (home_elo+away_elo, nivel combinado de ambos equipos):
  significativo (p=0.004) pero chico (coef=-0.0038) -- equipos mas
  fuertes en conjunto tienden a un total LIGERAMENTE mas bajo (control de
  ritmo de equipos de elite). Se mantiene por ser significativo, aunque el
  efecto es marginal.
- `home_is_denver`: significativo (p<0.0001), coef=+2.45 -- los partidos
  en Denver tienen un total mas alto (altitud, efecto real y documentado,
  mismo mecanismo que ya se confirmo para margen).
- **Descartados por NO ser significativos para totales** (a diferencia de
  margen, donde SI importaban): `b2b_sum` (ambos equipos en back-to-back,
  p=0.983 -- el descanso afecta quien gana, no cuanto se anota en total) y
  `3in4_sum` (p=0.085, no cruza el umbral de 0.05).
R^2 full-sample: 0.428 -- mucho mas alto que el modelo de margen (0.180),
esperado: el volumen total de puntos es intrinsecamente mas predecible que
quien gana.

**LIMITACION REAL, documentada explicita -- SIN comparacion de mercado
todavia**: `theoddsapi_historical_loader.py` solo descargo el mercado
`h2h` (moneyline) hasta ahora -- no hay lineas de totales reales
descargadas. Este modelo se evalua SOLO contra un baseline ingenuo interno
(promedio historico expandido de temporadas anteriores, sin fuga), NO
contra un mercado real todavia. Agregar el mercado de totales requeriria
volver a correr `theoddsapi_historical_loader.py` con
`markets="h2h,totals"` -- esto duplica aproximadamente el costo en
creditos de la descarga historica (cada mercado adicional cuesta creditos
por separado), decision que le queda al usuario antes de gastarlo
automaticamente.

Requiere que games_clean.csv ya tenga home_elo/away_elo Y
home_off_ewm/home_def_ewm/away_off_ewm/away_def_ewm
(`add_nba_gssd_features.py`, version con EWM).

Este archivo NO se corre standalone -- lo importa `backtest_nba_totals.py`.
"""
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

REQUIRED_COLS = [
    "home_elo", "away_elo", "home_pts", "away_pts", "home_team",
    "home_off_ewm", "home_def_ewm", "away_off_ewm", "away_def_ewm",
]
FEATURE_COLS = ["home_off_ewm", "home_def_ewm", "away_off_ewm", "away_def_ewm", "elo_sum", "home_is_denver"]
DENVER_TEAM_NAME = "Denver Nuggets"


def _check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} -- corre 'python -m src.processing.add_nba_elo_features' y "
            f"'python -m src.processing.add_nba_gssd_features' (version con EWM) sobre games_clean.csv."
        )


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_total"] = df["home_pts"] + df["away_pts"]
    df["elo_sum"] = df["home_elo"] + df["away_elo"]
    df["home_is_denver"] = (df["home_team"] == DENVER_TEAM_NAME).astype(float)
    return df


def fit_totals_model(train_df: pd.DataFrame):
    """OLS: game_total ~ home_off_ewm + home_def_ewm + away_off_ewm +
    away_def_ewm + elo_sum + home_is_denver. Devuelve el modelo ajustado y
    sigma (desvio estandar de los residuos del propio training set)."""
    _check_columns(train_df)
    train_df = _add_features(train_df)

    before = len(train_df)
    train_df = train_df.dropna(subset=FEATURE_COLS + ["game_total"])
    dropped = before - len(train_df)
    if dropped:
        print(f"  [INFO] {dropped} partidos de training sin features GSSD calculables -- excluidos del "
              f"entrenamiento, no rellenados.")

    X = sm.add_constant(train_df[FEATURE_COLS])
    y = train_df["game_total"]
    model = sm.OLS(y, X).fit()

    sigma = float(model.resid.std(ddof=1))
    return model, sigma


def predict_dataframe(model, sigma: float, df: pd.DataFrame, total_line_col: str = None) -> pd.DataFrame:
    """Aplica el modelo a un DataFrame completo, vectorizado. Si se pasa
    `total_line_col` (nombre de una columna con la linea de totales del
    mercado), tambien calcula P(over)/P(under) para esa linea -- hoy no
    existe esa columna en el proyecto (ver limitacion en el docstring del
    modulo), se deja preparado para cuando se descargue el mercado de
    totales."""
    _check_columns(df)
    feats = _add_features(df)
    X = sm.add_constant(feats[FEATURE_COLS], has_constant="add")

    out = pd.DataFrame(index=df.index)
    out["mu_total"] = model.predict(X).values
    out["sigma_total"] = sigma

    if total_line_col is not None and total_line_col in df.columns:
        out["model_prob_over"] = 1.0 - norm.cdf(
            df[total_line_col].values, loc=out["mu_total"], scale=out["sigma_total"]
        )
        out["model_prob_under"] = 1.0 - out["model_prob_over"]

    return out
