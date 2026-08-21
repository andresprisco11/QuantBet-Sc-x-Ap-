"""
Fase 10 -- limpieza de NBA, mismo patron "un script de limpieza por deporte"
que clean_data.py/clean_tennis_data.py/clean_nfl_data.py, con 3 diferencias
reales CONFIRMADAS por la corrida real de probe() (nba_data_loader.py,
2026-08-21, no asumidas):

1. **2 FILAS POR PARTIDO, no 1** -- LeagueGameFinder devuelve una fila por
   EQUIPO (confirmado: temporada 2023-24 = 2460 filas / 1230 GAME_ID unicos =
   ratio 2.00 exacto). Este script pivotea esas 2 filas en 1 fila home/away,
   parseando el texto de MATCHUP: "XXX vs. YYY" = XXX es LOCAL, YYY visitante;
   "XXX @ YYY" = XXX es VISITANTE, YYY local. Confirmado con el ejemplo real
   del probe: fila "BOS vs. WAS" (BOS local, gano 132-122) + fila "WAS @ BOS"
   (WAS visitante, perdio) = mismo GAME_ID.

   **Bug real de stats.nba.com, confirmado investigando los 10 GAME_ID que la
   primera version de este script descartaba (2026-08-21, no una suposicion)**:
   en 10 de 35,546 partidos (0.03%, todos de temporadas 2024-25/2025-26), LAS
   DOS filas del mismo partido traen el MISMO texto de MATCHUP en formato "@"
   (ej. tanto la fila de Miami Heat como la de Washington Wizards dicen
   "MIA @ WAS" -- la fila de Washington deberia decir "WAS vs. MIA"). El
   contenido del texto sigue siendo correcto (SI dice quien es local/
   visitante), solo falla la logica de "cada fila cuenta su propia
   perspectiva". Por eso el parseo NO depende de que fila dice que -- se
   parsea el string UNA vez (los abbr antes/despues de "vs."/"@") y despues
   se hace match contra TEAM_ABBREVIATION de cada fila, algo que funciona
   igual de bien tanto en el caso normal como en este bug real -- ya no hace
   falta descartar estos 10 partidos.

2. **NBA NO PERMITE EMPATE** -- a diferencia de NFL (que si, rara vez). Todo
   partido de NBA se decide (con tiempo extra si hace falta), asi que FTR es
   siempre H o A, nunca T. Si aparece un GAME_ID sin resultado claro, se
   reporta como anomalia, no se asume.

3. **SOLO TEMPORADA REGULAR por ahora** -- `nba_data_loader.py` pide
   `season_type_nullable="Regular Season"` por default. Playoffs quedan
   deliberadamente afuera de esta primera version (alcance documentado, no
   oculto) -- agregarlos requiere otra corrida de descarga con
   `season_type="Playoffs"`, no esta en el mandato original de este script.

**SIN cuotas de mercado en este archivo** -- a diferencia de clean_data.py/
clean_nfl_data.py, NBA no tiene cuotas en el dataset de resultados (eso vive
aparte, en theoddsapi_historical_loader.py). El merge de resultados + cuotas
es un script FUTURO, no este -- mismo principio de separacion que ya se uso
para pegar el xG de TheStatsAPI contra matches_clean.csv de futbol
(merge_thestatsapi_xg.py): nunca mezclar ingesta/limpieza de dos fuentes
distintas en el mismo paso sin un merge explicito y auditado aparte.

Consolida data/raw/NBA/nba_*.csv -> data/processed/NBA/games_clean.csv.

Uso: python -m src.processing.clean_nba_data
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw" / "NBA"
OUT_DIR = PROCESSED_DATA_DIR / "NBA"

# Columnas de estadisticas de caja que se duplican con prefijo home_/away_
# al pivotear -- confirmado contra las 28 columnas reales que devolvio el
# probe() de nba_data_loader.py (2026-08-21).
STAT_COLS = [
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
    "BLK", "TOV", "PF", "PLUS_MINUS", "MIN",
]


def _load_all_seasons() -> pd.DataFrame:
    files = sorted(RAW_DATA_DIR.glob("nba_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hay CSVs en {RAW_DATA_DIR} -- corre "
            "'python -m src.ingestion.nba_data_loader --download-all' primero."
        )
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def _season_label(season_id) -> str:
    """SEASON_ID real viene como '22023' (digito de tipo + año) -- confirmado
    en el probe. Se quita el primer digito y se arma 'YYYY-YY'."""
    year = int(str(season_id)[1:])
    return f"{year}-{str(year + 1)[-2:]}"


def _parse_matchup_abbrs(matchup: str):
    """Devuelve (home_abbr, away_abbr) parseando el TEXTO del MATCHUP, sin
    asumir de que fila viene -- esto es lo que hace que el bug real de
    stats.nba.com (ver docstring del archivo, ambas filas con el mismo string
    '@') no rompa el pivoteo: el string en si sigue codificando bien quien es
    local/visitante, solo hay que leerlo una vez, no por fila."""
    if " @ " in matchup:
        away_abbr, home_abbr = matchup.split(" @ ")
        return home_abbr.strip(), away_abbr.strip()
    if " vs. " in matchup:
        home_abbr, away_abbr = matchup.split(" vs. ")
        return home_abbr.strip(), away_abbr.strip()
    return None, None


def _pivot_game(group: pd.DataFrame):
    """Convierte las 2 filas (una por equipo) de un mismo GAME_ID en 1 fila
    home/away, usando el texto de MATCHUP (parseado una sola vez, ver
    _parse_matchup_abbrs) contra TEAM_ABBREVIATION de cada fila para asignar
    quien es local/visitante. Devuelve None solo si de verdad no se puede
    determinar (menos de 2 filas, o los abbr parseados no matchean ningun
    TEAM_ABBREVIATION real del grupo) -- se reporta como anomalia genuina en
    vez de forzar un resultado que no esta confirmado."""
    if len(group) != 2:
        return None

    home_abbr = away_abbr = None
    for _, r in group.iterrows():
        h, a = _parse_matchup_abbrs(r["MATCHUP"])
        if h is not None:
            home_abbr, away_abbr = h, a
            break
    if home_abbr is None:
        return None

    home_rows = group[group["TEAM_ABBREVIATION"] == home_abbr]
    away_rows = group[group["TEAM_ABBREVIATION"] == away_abbr]
    if len(home_rows) != 1 or len(away_rows) != 1:
        return None

    home = home_rows.iloc[0]
    away = away_rows.iloc[0]

    row = {
        "game_id": home["GAME_ID"],
        "season": _season_label(home["SEASON_ID"]),
        "game_date": home["GAME_DATE"],
        "home_team": home["TEAM_NAME"],
        "away_team": away["TEAM_NAME"],
        "home_team_abbr": home["TEAM_ABBREVIATION"],
        "away_team_abbr": away["TEAM_ABBREVIATION"],
    }
    for col in STAT_COLS:
        row[f"home_{col.lower()}"] = home[col]
        row[f"away_{col.lower()}"] = away[col]
    return row


def _outcome(row) -> str:
    if row["home_pts"] > row["away_pts"]:
        return "H"
    if row["home_pts"] < row["away_pts"]:
        return "A"
    return "T"  # no deberia pasar en NBA -- ver chequeo de sanidad abajo


def run() -> None:
    df = _load_all_seasons()

    pivoted = []
    n_anomalias = 0
    for game_id, group in df.groupby("GAME_ID"):
        row = _pivot_game(group)
        if row is None:
            n_anomalias += 1
            continue
        pivoted.append(row)

    if n_anomalias:
        print(f"[AVISO] {n_anomalias} GAME_ID realmente no se pudieron determinar "
              f"(ni con el parseo robusto por texto de MATCHUP) -- descartados, no "
              f"forzados. Investigar estos puntualmente antes de confiar en el resto.")

    out = pd.DataFrame(pivoted)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["FTR"] = out.apply(_outcome, axis=1)
    out["point_margin"] = out["home_pts"] - out["away_pts"]

    n_ties = int((out["FTR"] == "T").sum())
    if n_ties:
        print(f"[AVISO] {n_ties} partidos con FTR='T' (empate) -- NBA no deberia "
              f"permitir esto, revisar esos GAME_ID especificos antes de confiar "
              f"en el resto del dataset.")

    home_win_rate = (out["FTR"] == "H").mean()
    print(f"Chequeo de sanidad -- % de victorias del local: {home_win_rate:.2%} "
          f"(la ventaja de local en NBA historicamente ronda 55-60% segun fuentes "
          f"publicas -- si esto sale muy fuera de ese rango, revisar el mapeo "
          f"home/away antes de seguir).")

    print(f"Temporadas incluidas: {sorted(out['season'].unique())}")
    print(f"Total de partidos: {len(out)} (de {df['GAME_ID'].nunique()} GAME_ID reales en crudo, "
          f"{n_anomalias} descartados por anomalia)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "games_clean.csv"
    out.sort_values("game_date").to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")


if __name__ == "__main__":
    run()
