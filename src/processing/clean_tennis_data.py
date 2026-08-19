"""
Escalamiento a Tenis, paso 2 (tras tennis_data_loader.py): limpieza y
normalización. Consolida los CSV crudos por año en un único
matches_clean.csv por tour, y resuelve un problema de diseño real que NO
existía en el pipeline de fútbol.

PROBLEMA DE FUGA DE INFORMACIÓN (leakage), encontrado al inspeccionar el
esquema real via probe(): tennis-data.co.uk etiqueta las columnas como
`Winner`/`Loser` -- es decir, POST-partido. En fútbol, `HomeTeam`/
`AwayTeam` son neutrales (se conocen antes del partido); acá no hay
equivalente neutral en el archivo crudo. Si se entrena un modelo con
`Winner`/`Loser` tal cual, el modelo "aprendería" trivialmente que el
ganador es el que aparece en la columna Winner -- no es una feature real,
es el propio target filtrado hacia atrás.

Solución: reetiquetar cada partido a `Player1`/`Player2` con un orden
NEUTRAL decidido ANTES de mirar el resultado (alfabético por apellido,
la unica funcion disponible en los datos crudos que no depende de quien
gano) y agregar `Player1_Won` (bool) como target explicito. Todas las
columnas de cuota/ranking/puntos se remapean junto con el jugador
(Player1_*/Player2_*) para que el esquema completo quede sin fuga.

METODOLOGÍA:
1. Consolida data/raw/TENNIS_<TOUR>/*.csv (un archivo por año, de
   tennis_data_loader.py) en un único DataFrame por tour.
2. Filtra partidos incompletos (Comment != 'Completed' -- retiros y
   walkovers no reflejan una diferencia de nivel real entre jugadores,
   mismo criterio de higiene de datos que el resto del proyecto aplica
   a resultados que no representan la pregunta que se quiere predecir).
3. Reetiqueta Winner/Loser -> Player1/Player2 (orden alfabético,
   neutral) + Player1_Won.
4. Probabilidad implícita SIN MARGEN (no-vig) a partir de las cuotas de
   Pinnacle (PSW/PSL, confirmadas via probe() -- no B365 ni otra casa,
   mismo estándar que el proyecto ya usa en fútbol con Pinnacle).
5. NO hay cuota de apertura separada en este dataset (confirmado en
   probe() -- notes.txt dice que la cuota es "la más reciente antes de
   que arranque el partido") -- por lo tanto CLV tal como está
   implementado en fútbol (cierre - apertura) NO se calcula acá. Se deja
   guardada la cuota Avg/Max de Oddsportal como proxy de consenso de
   mercado, sin llamarla CLV, para no inventar una métrica que el
   dataset no permite calcular de verdad.

Salida: data/processed/TENNIS_ATP/matches_clean.csv,
        data/processed/TENNIS_WTA/matches_clean.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"  # misma inferencia que tennis_data_loader.py

ODDS_PAIRS = [("PS", "PSW", "PSL"), ("B365", "B365W", "B365L"),
              ("Max", "MaxW", "MaxL"), ("Avg", "AvgW", "AvgL")]


def _load_raw_tour(tour: str) -> pd.DataFrame:
    raw_dir = RAW_DATA_DIR / f"TENNIS_{tour.upper()}"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"No existe {raw_dir}. Corre 'python -m src.ingestion.tennis_data_loader --download-all' primero."
        )
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay archivos CSV en {raw_dir}.")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _filter_completed(df: pd.DataFrame) -> pd.DataFrame:
    if "Comment" not in df.columns:
        print("  [AVISO] no existe columna 'Comment' -- no se puede filtrar retiros/walkovers, se sigue sin filtrar.")
        return df
    n_before = len(df)
    df = df[df["Comment"].astype(str).str.strip().str.lower() == "completed"].copy()
    n_after = len(df)
    print(f"  Filtro 'Completed' (excluye retiros/walkovers -- no reflejan diferencia de nivel real): "
          f"{n_before} -> {n_after} partidos ({n_before - n_after} descartados).")
    return df


def _neutralize_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordena Winner/Loser -> Player1/Player2 por orden alfabetico del
    nombre (neutral, no depende del resultado) y remapea cuotas/ranking/
    puntos junto con el jugador. Agrega Player1_Won como target.
    """
    winner_is_p1 = df["Winner"] < df["Loser"]  # orden alfabetico -- decidido SIN mirar quien gano

    out = pd.DataFrame({
        "Date": df["Date"],
        "Tournament": df.get("Tournament"),
        "Location": df.get("Location"),
        "Surface": df.get("Surface"),
        "Court": df.get("Court"),
        "Series": df.get("Series"),
        "Round": df.get("Round"),
        "Best_of": df.get("Best of"),
        "tour": df.get("tour"),
        "Player1": df["Winner"].where(winner_is_p1, df["Loser"]),
        "Player2": df["Loser"].where(winner_is_p1, df["Winner"]),
        "Player1_Won": winner_is_p1,
    })

    if "WRank" in df.columns and "LRank" in df.columns:
        out["Player1_Rank"] = df["WRank"].where(winner_is_p1, df["LRank"])
        out["Player2_Rank"] = df["LRank"].where(winner_is_p1, df["WRank"])
    if "WPts" in df.columns and "LPts" in df.columns:
        out["Player1_Pts"] = df["WPts"].where(winner_is_p1, df["LPts"])
        out["Player2_Pts"] = df["LPts"].where(winner_is_p1, df["WPts"])

    for label, w_col, l_col in ODDS_PAIRS:
        if w_col in df.columns and l_col in df.columns:
            out[f"Player1_{label}_Odds"] = df[w_col].where(winner_is_p1, df[l_col])
            out[f"Player2_{label}_Odds"] = df[l_col].where(winner_is_p1, df[w_col])

    return out


def _add_no_vig_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad implicita SIN MARGEN a partir de las cuotas de Pinnacle (PS)."""
    if "Player1_PS_Odds" not in df.columns or "Player2_PS_Odds" not in df.columns:
        print("  [AVISO] no hay columnas Player1_PS_Odds/Player2_PS_Odds -- no se puede calcular no-vig prob.")
        return df

    valid = df["Player1_PS_Odds"].notna() & df["Player2_PS_Odds"].notna()
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"  [AVISO] {n_missing} partidos sin cuota de Pinnacle en alguno de los dos jugadores "
              f"-- quedan con no_vig_prob = NaN, no se descartan del archivo.")

    implied_p1 = 1.0 / df["Player1_PS_Odds"]
    implied_p2 = 1.0 / df["Player2_PS_Odds"]
    margin = implied_p1 + implied_p2

    df["no_vig_prob_player1"] = implied_p1 / margin
    df["no_vig_prob_player2"] = implied_p2 / margin
    df["market_margin"] = margin - 1.0
    return df


def run(tour: str) -> None:
    print(f"\n=== {tour.upper()} ===")
    df_raw = _load_raw_tour(tour)
    print(f"Cargados {len(df_raw)} partidos crudos de {RAW_DATA_DIR / f'TENNIS_{tour.upper()}'}")

    df = _filter_completed(df_raw)
    df = _neutralize_players(df)
    df = _add_no_vig_probs(df)

    win_rate_p1 = df["Player1_Won"].mean()
    print(f"  Chequeo de sanidad del reetiquetado neutral: Player1 gana {win_rate_p1:.2%} de las veces "
          f"(debe rondar 50% -- si está muy lejos de 50%, el orden alfabético introdujo un sesgo, revisar).")

    out_dir = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matches_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"  Guardado -> {out_path} ({len(df)} partidos, {len(df.columns)} columnas)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Limpieza y normalización de datos de tenis -- consolida años, resuelve leakage Winner/Loser, calcula no-vig prob de Pinnacle.")
    parser.add_argument("--tours", type=str, default="ATP,WTA", help="Tours a limpiar, separados por coma (default: ATP,WTA).")
    args = parser.parse_args()

    for tour in args.tours.split(","):
        try:
            run(tour)
        except FileNotFoundError as e:
            print(f"[SKIP] {tour}: {e}")