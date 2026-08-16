"""
Módulo de limpieza y feature base: toma los CSV crudos de data/raw/<liga>/,
los consolida, extrae las cuotas de las casas prioritarias y calcula
probabilidades implícitas normalizadas (sin margen de la casa) para
apertura y cierre — el insumo base para medir CLV (Closing Line Value).

Salida: un único CSV limpio por liga en data/processed/<liga>/matches_clean.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PRIORITY_BOOKMAKERS, PROCESSED_DATA_DIR, RAW_DATA_DIR

# Columnas base que siempre deben existir en un CSV de football-data.co.uk
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_raw_league(league_key: str) -> pd.DataFrame:
    """Concatena todos los CSV crudos de una liga en un solo DataFrame."""
    league_dir = RAW_DATA_DIR / league_key
    csv_files = sorted(league_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No hay CSVs crudos en {league_dir}. Corre la ingesta primero.")

    frames = [pd.read_csv(f) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


def implied_prob_no_vig(odds_home: pd.Series, odds_draw: pd.Series, odds_away: pd.Series) -> pd.DataFrame:
    """
    Convierte cuotas decimales en probabilidades implícitas SIN el margen
    de la casa (overround). Formula estándar:
        prob_bruta = 1 / cuota
        prob_justa = prob_bruta / suma(prob_brutas)
    Esto es matemática determinística (no una predicción) — el punto de
    partida estándar en la literatura de Benter/Mack para comparar contra
    el modelo propio.
    """
    raw_home = 1 / odds_home
    raw_draw = 1 / odds_draw
    raw_away = 1 / odds_away
    overround = raw_home + raw_draw + raw_away

    return pd.DataFrame({
        "prob_home": raw_home / overround,
        "prob_draw": raw_draw / overround,
        "prob_away": raw_away / overround,
        "overround": overround,  # >1.0 = margen de la casa; útil para medir "sharpness"
    })


def add_bookmaker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada casa prioritaria configurada en settings, calcula probabilidades
    implícitas de apertura y cierre, y el movimiento de línea (CLV proxy).
    Si una casa no tiene columnas en el CSV (temporadas viejas), se omite
    sin romper el pipeline.
    """
    out = df.copy()

    for book_name, cols in PRIORITY_BOOKMAKERS.items():
        required = [cols["open"], cols["draw"], f"{cols['open'][:-1]}A", cols["close_home"], cols["close_draw"], cols["close_away"]]
        # cols["open"] es tipo "PSH" -> away de apertura es "PSA"
        open_away_col = cols["open"][:-1] + "A"

        needed_cols = [cols["open"], cols["draw"], open_away_col,
                       cols["close_home"], cols["close_draw"], cols["close_away"]]

        if not all(c in df.columns for c in needed_cols):
            print(f"[SKIP] {book_name}: columnas no disponibles en este dataset, se omite.")
            continue

        open_probs = implied_prob_no_vig(df[cols["open"]], df[cols["draw"]], df[open_away_col])
        close_probs = implied_prob_no_vig(df[cols["close_home"]], df[cols["close_draw"]], df[cols["close_away"]])

        out[f"{book_name}_open_prob_home"] = open_probs["prob_home"]
        out[f"{book_name}_open_prob_draw"] = open_probs["prob_draw"]
        out[f"{book_name}_open_prob_away"] = open_probs["prob_away"]
        out[f"{book_name}_open_overround"] = open_probs["overround"]

        out[f"{book_name}_close_prob_home"] = close_probs["prob_home"]
        out[f"{book_name}_close_prob_draw"] = close_probs["prob_draw"]
        out[f"{book_name}_close_prob_away"] = close_probs["prob_away"]
        out[f"{book_name}_close_overround"] = close_probs["overround"]

        # CLV proxy: movimiento de probabilidad justa entre apertura y cierre.
        # Positivo en "home" = el mercado le dio más chance al local al cerrar.
        out[f"{book_name}_clv_home"] = close_probs["prob_home"] - open_probs["prob_home"]
        out[f"{book_name}_clv_draw"] = close_probs["prob_draw"] - open_probs["prob_draw"]
        out[f"{book_name}_clv_away"] = close_probs["prob_away"] - open_probs["prob_away"]

    return out


def clean_league(league_key: str) -> Path:
    """Pipeline completo: cargar crudo -> validar columnas core -> features de cuotas -> guardar."""
    df = load_raw_league(league_key)

    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(f"Faltan columnas core en {league_key}: {missing_core}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "FTR"])  # partidos sin resultado no sirven para entrenar

    df = add_bookmaker_features(df)

    out_dir = PROCESSED_DATA_DIR / league_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matches_clean.csv"
    df.to_csv(out_path, index=False)

    return out_path


if __name__ == "__main__":
    for league_key in LEAGUES:
        path = clean_league(league_key)
        print(f"[OK] {league_key} limpio -> {path}")
        