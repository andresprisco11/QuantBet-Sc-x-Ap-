"""
Fase 10 -- pega el xG real de TheStatsAPI (thestatsapi_xg_loader.py) contra
matches_clean.csv del pipeline de futbol. Los dos datasets no comparten
llave (TheStatsAPI usa sus propios match_id/team_id) -- el merge es por
fecha + nombre de equipo, con un mapeo de nombres CONFIRMADO contra datos
reales liga por liga (nunca fuzzy-matching automatico ni adivinado).

**BUG REAL encontrado y corregido (2026-08-21), antes de mapear EPL/LaLiga/
SerieA**: la version anterior de este archivo parseaba `Date` con
`pd.to_datetime(fb["Date"], dayfirst=True, errors="coerce")` -- pero la
columna `Date` de `matches_clean.csv` YA esta en formato ISO
(`YYYY-MM-DD`), sin ambiguedad. Pandas 3.x infiere el formato UNA SOLA VEZ
a partir de la primera fila del array; si esa primera fila tiene un dia
<=12 (ambiguo) y `dayfirst=True`, pandas puede inferir el formato como
"%Y-%d-%m" para TODO el array, corrompiendo la fecha en la mayoria de las
filas siguientes. Confirmado con datos reales: en EPL esto cambiaba
2,222 de 2,280 fechas (97.5%); en Bundesliga, por una casualidad del orden
de los datos (la primera fila no era ambigua), el mismo bug NO producia
ningun cambio (0 de 1,836) -- **por eso el merge de Bundesliga ya
entregado (1222/1232, 99.2%) sigue siendo valido, verificado explicitamente
corriendo el merge con y sin el bug sobre los mismos datos reales: resultado
identico**. Si se hubiera mapeado EPL/LaLiga/SerieA con el codigo viejo sin
detectar esto, el merge de esas 3 ligas habria fallado silenciosamente para
la gran mayoria de partidos. Arreglado usando `format="%Y-%m-%d"` explicito
en vez de `dayfirst=True` -- mas rapido, y sin ambiguedad posible.

**Bundesliga -- confirmado y probado (2026-08-21)**: 23/23 equipos de
football-data.co.uk mapeados 1:1 sin ambiguedad contra los nombres de
TheStatsAPI. Merge real: 1222/1232 partidos de xG pegados (99.2%). Los 10
restantes se investigaron uno por uno, no se descartaron a ciegas: 8 son
partidos de REPECHAJE/PROMOCION (Bundesliga-2.Bundesliga, ej. Hamburgo vs
Stuttgart 2023, Paderborn vs Wolfsburg 2026) que TheStatsAPI cuenta bajo el
mismo competition_id de Bundesliga pero que football-data.co.uk NO incluye
en su archivo de liga regular -- diferencia real de alcance entre fuentes,
no un bug del merge. Los otros 2 quedan sin explicar todavia.

**EPL/LaLiga/SerieA -- mapeo confirmado (2026-08-21)**, mismo proceso que
Bundesliga: comparar la lista real de nombres de equipo de las dos fuentes
(temporadas >=22/23, con el fix de fecha aplicado) y completar el
diccionario a mano.
- **EPL**: 25/25 equipos, 1:1 exacto, sin ambiguedad.
- **LaLiga**: 26/26 equipos reales mapeados 1:1. Dos equipos de
  `matches_clean.csv` quedan deliberadamente SIN mapear -- "Dep. A Coruna"
  y "Santander" (Racing Santander) -- investigados y confirmados como
  partidos REALES de la temporada 2026-27 recien arrancada (16-17 de
  agosto de 2026), fuera del rango de xG confirmado por TheStatsAPI
  (2022/23-2025/26) -- no es un error, es exactamente el mismo caso que
  "temporadas viejas sin cobertura" pero del lado nuevo del calendario.
- **SerieA**: 27/27 equipos, 1:1 exacto -- unico caso real de nombre
  distinto: "Verona" (football-data.co.uk) = "Hellas Verona" (TheStatsAPI).

Uso:
    python -m src.processing.merge_thestatsapi_xg --diff-names EPL   # paso 1: ver nombres reales (ya no hace falta para las 4 ligas actuales)
    python -m src.processing.merge_thestatsapi_xg --league EPL       # paso 2: correr el merge
    python -m src.processing.merge_thestatsapi_xg --all              # corre las 4 ligas
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"  # mismo criterio inferido que en thestatsapi_xg_loader.py

# Confirmado con datos reales (2026-08-21): ver docstring del modulo para
# el detalle de como se armo cada mapeo y las excepciones documentadas.
NAME_MAPS = {
    "BUNDESLIGA": {
        "Augsburg": "FC Augsburg",
        "Bayern Munich": "FC Bayern München",
        "Bochum": "VfL Bochum 1848",
        "Darmstadt": "Darmstadt 98",
        "Dortmund": "Borussia Dortmund",
        "Ein Frankfurt": "Eintracht Frankfurt",
        "FC Koln": "1. FC Köln",
        "Freiburg": "SC Freiburg",
        "Hamburg": "Hamburger SV",
        "Heidenheim": "1. FC Heidenheim",
        "Hertha": "Hertha BSC",
        "Hoffenheim": "TSG Hoffenheim",
        "Holstein Kiel": "Holstein Kiel",
        "Leverkusen": "Bayer 04 Leverkusen",
        "M'gladbach": "Borussia M'gladbach",
        "Mainz": "1. FSV Mainz 05",
        "RB Leipzig": "RB Leipzig",
        "Schalke 04": "FC Schalke 04",
        "St Pauli": "FC St. Pauli",
        "Stuttgart": "VfB Stuttgart",
        "Union Berlin": "1. FC Union Berlin",
        "Werder Bremen": "SV Werder Bremen",
        "Wolfsburg": "VfL Wolfsburg",
    },
    "EPL": {
        "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
        "Brentford": "Brentford", "Brighton": "Brighton & Hove Albion", "Burnley": "Burnley",
        "Chelsea": "Chelsea", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
        "Fulham": "Fulham", "Ipswich": "Ipswich Town", "Leeds": "Leeds United",
        "Leicester": "Leicester City", "Liverpool": "Liverpool", "Luton": "Luton Town",
        "Man City": "Manchester City", "Man United": "Manchester United",
        "Newcastle": "Newcastle United", "Nott'm Forest": "Nottingham Forest",
        "Sheffield United": "Sheffield United", "Southampton": "Southampton",
        "Sunderland": "Sunderland", "Tottenham": "Tottenham Hotspur",
        "West Ham": "West Ham United", "Wolves": "Wolverhampton",
    },
    "LALIGA": {
        "Alaves": "Deportivo Alavés", "Almeria": "Almería", "Ath Bilbao": "Athletic Club",
        "Ath Madrid": "Atlético Madrid", "Barcelona": "Barcelona", "Betis": "Real Betis",
        "Cadiz": "Cádiz", "Celta": "Celta Vigo", "Elche": "Elche", "Espanol": "Espanyol",
        "Getafe": "Getafe", "Girona": "Girona FC", "Granada": "Granada",
        "Las Palmas": "Las Palmas", "Leganes": "Leganés", "Levante": "Levante UD",
        "Mallorca": "Mallorca", "Osasuna": "Osasuna", "Oviedo": "Real Oviedo",
        "Real Madrid": "Real Madrid", "Sevilla": "Sevilla", "Sociedad": "Real Sociedad",
        "Valencia": "Valencia", "Valladolid": "Real Valladolid", "Vallecano": "Rayo Vallecano",
        "Villarreal": "Villarreal",
        # "Dep. A Coruna" y "Santander" deliberadamente NO mapeados -- ver docstring
        # del modulo (partidos reales de temporada 2026-27, fuera de cobertura de xG).
    },
    "SERIEA": {
        "Atalanta": "Atalanta", "Bologna": "Bologna", "Cagliari": "Cagliari", "Como": "Como",
        "Cremonese": "Cremonese", "Empoli": "Empoli", "Fiorentina": "Fiorentina",
        "Frosinone": "Frosinone", "Genoa": "Genoa", "Inter": "Inter", "Juventus": "Juventus",
        "Lazio": "Lazio", "Lecce": "Lecce", "Milan": "Milan", "Monza": "Monza",
        "Napoli": "Napoli", "Parma": "Parma", "Pisa": "Pisa", "Roma": "Roma",
        "Salernitana": "Salernitana", "Sampdoria": "Sampdoria", "Sassuolo": "Sassuolo",
        "Spezia": "Spezia", "Torino": "Torino", "Udinese": "Udinese", "Venezia": "Venezia",
        "Verona": "Hellas Verona",
    },
}


def _print_team_name_diff(league_key: str) -> None:
    """Paso 1 opcional para revalidar el mapeo de una liga -- imprime las
    dos listas reales de nombres para armar/confirmar el mapeo a mano,
    mismo proceso que se uso para las 4 ligas actuales."""
    fb_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    xg_path = RAW_DATA_DIR / "THESTATSAPI" / f"{league_key}_xg_raw.csv"
    if not fb_path.exists() or not xg_path.exists():
        print(f"[SKIP] Falta {fb_path} o {xg_path} -- confirmar que ambos existen antes de comparar nombres.")
        return

    fb = pd.read_csv(fb_path)
    xg = pd.read_csv(xg_path)
    fb["Date"] = pd.to_datetime(fb["Date"], format="%Y-%m-%d", errors="coerce")
    fb_recent = fb[fb["Date"] >= "2022-07-01"]

    fb_teams = sorted(fb_recent["HomeTeam"].unique())
    xg_teams = sorted(set(xg["home_team_name"].unique()) | set(xg["away_team_name"].unique()))

    print(f"\n=== {league_key} -- equipos en matches_clean.csv (temporadas >=22/23): {len(fb_teams)} ===")
    print(fb_teams)
    print(f"\n=== {league_key} -- equipos en {league_key}_xg_raw.csv (TheStatsAPI): {len(xg_teams)} ===")
    print(xg_teams)
    print(f"\n[SIGUIENTE PASO MANUAL] Completar NAME_MAPS['{league_key}'] en este archivo con el mapeo real "
          f"1:1 entre las dos listas de arriba -- no asumir que el patron de otra liga se repite igual.")


def merge_league(league_key: str) -> None:
    name_map = NAME_MAPS.get(league_key)
    if name_map is None:
        print(f"[SKIP] NAME_MAPS['{league_key}'] todavia no esta confirmado -- correr primero "
              f"'--diff-names {league_key}' y completar el diccionario a mano.")
        return

    fb_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    xg_path = RAW_DATA_DIR / "THESTATSAPI" / f"{league_key}_xg_raw.csv"
    if not fb_path.exists() or not xg_path.exists():
        print(f"[SKIP] Falta {fb_path} o {xg_path}.")
        return

    fb = pd.read_csv(fb_path).copy()  # .copy() evita el PerformanceWarning de fragmentacion al agregar columnas
    xg = pd.read_csv(xg_path)

    # Fix real (ver docstring del modulo): la columna Date de matches_clean.csv
    # ya esta en ISO (YYYY-MM-DD) -- format explicito, NUNCA dayfirst=True aca.
    fb["Date"] = pd.to_datetime(fb["Date"], format="%Y-%m-%d", errors="coerce")
    fb["xg_home_team"] = fb["HomeTeam"].map(name_map)
    fb["xg_away_team"] = fb["AwayTeam"].map(name_map)
    fb["match_date"] = fb["Date"].dt.date
    xg["match_date"] = pd.to_datetime(xg["utc_date"]).dt.date

    n_unmapped = int((fb["xg_home_team"].isna() | fb["xg_away_team"].isna()).sum())
    # Esperado: filas de temporadas viejas (pre-22/23) o, en el caso de
    # LaLiga, equipos recien promovidos a la temporada 2026-27 todavia sin
    # cobertura de xG (Dep. A Coruna, Santander) -- no es un error si son
    # muchas, es la parte del historico/futuro sin cobertura confirmada.

    merged = fb.merge(
        xg[["match_date", "home_team_name", "away_team_name",
            "home_xg", "away_xg", "home_npxg", "away_npxg",
            "home_big_chances", "away_big_chances",
            "home_total_shots", "away_total_shots",
            "home_possession", "away_possession"]],
        left_on=["match_date", "xg_home_team", "xg_away_team"],
        right_on=["match_date", "home_team_name", "away_team_name"],
        how="left",
    )
    merged = merged.drop(columns=["home_team_name", "away_team_name", "xg_home_team", "xg_away_team"])

    n_matched = int(merged["home_xg"].notna().sum())
    n_xg_total = len(xg)
    print(f"=== {league_key} ===")
    print(f"Filas de matches_clean.csv sin mapeo de nombre (esperado, temporadas viejas/nuevas sin xG): {n_unmapped}")
    print(f"Partidos de xG disponibles: {n_xg_total}")
    print(f"Partidos con xG pegado exitosamente: {n_matched} ({n_matched/n_xg_total:.1%} del xG disponible)")
    if n_matched / n_xg_total < 0.95:
        print(f"[AVISO] cobertura del merge por debajo del 95% -- revisar el mapeo de nombres o fechas "
              f"antes de confiar en este resultado (para Bundesliga el 0.8% sin match fueron partidos de "
              f"repechaje que football-data.co.uk no incluye -- confirmar si aca pasa lo mismo).")

    out_path = PROCESSED_DATA_DIR / league_key / "matches_clean_with_xg.csv"
    merged.to_csv(out_path, index=False)
    print(f"Guardado -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--diff-names", metavar="LIGA", help="Paso 1: imprime nombres reales para armar el mapeo")
    parser.add_argument("--league", metavar="LIGA", help="Paso 2: corre el merge de una liga (requiere mapeo confirmado)")
    parser.add_argument("--all", action="store_true", help="Corre el merge de las 4 ligas")
    args = parser.parse_args()

    if args.diff_names:
        _print_team_name_diff(args.diff_names)
    elif args.all:
        for liga in NAME_MAPS:
            merge_league(liga)
    elif args.league:
        merge_league(args.league)
    else:
        parser.print_help()
