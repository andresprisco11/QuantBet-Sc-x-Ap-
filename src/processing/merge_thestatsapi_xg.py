"""
Fase 10 -- pega el xG real de TheStatsAPI (thestatsapi_xg_loader.py) contra
matches_clean.csv del pipeline de futbol. Los dos datasets no comparten
llave (TheStatsAPI usa sus propios match_id/team_id) -- el merge es por
fecha + nombre de equipo, con un mapeo de nombres CONFIRMADO contra datos
reales liga por liga (nunca fuzzy-matching automatico ni adivinado).

**Bundesliga -- confirmado y probado (2026-08-21)**: 23/23 equipos de
football-data.co.uk mapeados 1:1 sin ambiguedad contra los nombres de
TheStatsAPI. Merge real: 1222/1232 partidos de xG pegados (99.2%). Los 10
restantes se investigaron uno por uno, no se descartaron a ciegas: 8 son
partidos de REPECHAJE/PROMOCION (Bundesliga-2.Bundesliga, ej. Hamburgo vs
Stuttgart 2023, Paderborn vs Wolfsburg 2026) que TheStatsAPI cuenta bajo el
mismo competition_id de Bundesliga pero que football-data.co.uk NO incluye
en su archivo de liga regular -- diferencia real de alcance entre fuentes,
no un bug del merge. Los otros 2 quedan sin explicar todavia (diferencia
menor entre el conteo del merge y el conteo por llave exacta, no
investigado a fondo -- documentado, no oculto).

**EPL/LaLiga/SerieA -- mapeo TODAVIA NO CONFIRMADO** (`NAME_MAPS[liga] is
None`). Antes de correr `merge_league()` para esas ligas, hay que repetir
el mismo proceso que se hizo para Bundesliga: comparar la lista real de
nombres de equipo de las dos fuentes (`_print_team_name_diff()`, incluido
aca abajo) y completar el diccionario a mano con el resultado real -- NO
asumir que el patron de Bundesliga (agregar "FC"/quitar apellido del club)
se repite igual en las otras ligas.

Uso:
    python -m src.processing.merge_thestatsapi_xg --diff-names EPL   # paso 1: ver nombres reales
    python -m src.processing.merge_thestatsapi_xg --league BUNDESLIGA # paso 2: correr el merge ya confirmado
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"  # mismo criterio inferido que en thestatsapi_xg_loader.py

# Confirmado con datos reales (2026-08-21): 23/23 equipos, sin ambiguedad,
# comparando HomeTeam de matches_clean.csv (temporadas 22/23 en adelante)
# contra home_team_name/away_team_name de BUNDESLIGA_xg_raw.csv.
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
    "EPL": None,       # TODO: completar con --diff-names EPL una vez bajada
    "LALIGA": None,    # TODO: completar con --diff-names LALIGA una vez bajada
    "SERIEA": None,    # TODO: completar con --diff-names SERIEA una vez bajada
}


def _print_team_name_diff(league_key: str) -> None:
    """Paso 1 obligatorio antes de completar NAME_MAPS para una liga nueva
    -- imprime las dos listas reales de nombres para armar el mapeo a
    mano, mismo proceso que se uso para confirmar Bundesliga."""
    fb_path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    xg_path = RAW_DATA_DIR / "THESTATSAPI" / f"{league_key}_xg_raw.csv"
    if not fb_path.exists() or not xg_path.exists():
        print(f"[SKIP] Falta {fb_path} o {xg_path} -- confirmar que ambos existen antes de comparar nombres.")
        return

    fb = pd.read_csv(fb_path)
    xg = pd.read_csv(xg_path)
    fb["Date"] = pd.to_datetime(fb["Date"], dayfirst=True, errors="coerce")
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

    fb = pd.read_csv(fb_path)
    xg = pd.read_csv(xg_path)

    fb["Date"] = pd.to_datetime(fb["Date"], dayfirst=True, errors="coerce")
    fb["xg_home_team"] = fb["HomeTeam"].map(name_map)
    fb["xg_away_team"] = fb["AwayTeam"].map(name_map)
    fb["match_date"] = fb["Date"].dt.date
    xg["match_date"] = pd.to_datetime(xg["utc_date"]).dt.date

    n_unmapped = int((fb["xg_home_team"].isna() | fb["xg_away_team"].isna()).sum())
    # Esperado: filas de temporadas viejas (pre-22/23), que nunca van a tener xG --
    # no es un error si son muchas, es la parte del historico sin cobertura confirmada.

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
    print(f"Filas de matches_clean.csv sin mapeo de nombre (esperado, temporadas viejas): {n_unmapped}")
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
    parser.add_argument("--league", metavar="LIGA", help="Paso 2: corre el merge (requiere mapeo ya confirmado)")
    args = parser.parse_args()

    if args.diff_names:
        _print_team_name_diff(args.diff_names)
    elif args.league:
        merge_league(args.league)
    else:
        parser.print_help()
