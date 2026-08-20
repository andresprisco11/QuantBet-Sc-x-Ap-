"""
Escalamiento a NFL (mandato original del proyecto: "de futbol a NFL, Tenis
y NBA"). Tercer deporte del proyecto. Tenis se hizo primero (decisión del
CTO, 2026-08-19, documentada en `tennis_data_loader.py`) porque reutilizaba
casi toda la arquitectura de probabilidad/blend ya construida (moneyline,
resultado binario sin empate). **NFL SÍ requiere un modelo estadístico
nuevo** -- el mercado de NFL gira en torno al SPREAD DE PUNTOS, no solo al
moneyline (ganar/perder) -- así que el modelo predictivo (próximo paso,
no este script) va a necesitar predecir una distribución de diferencial de
puntos, no solo una probabilidad de victoria. Este script es SOLO ingesta
-- confirma la fuente de datos real antes de construir nada más, mismo
criterio de disciplina que ya se aplicó a tennis-data.co.uk y a Sportmonks
("nunca adivinar un esquema/columna -- confirmar con datos reales primero").

FUENTE ELEGIDA (decisión del CTO, 2026-08-20): `nflreadpy`
(https://github.com/nflverse/nflreadpy), el paquete Python oficial de la
comunidad nflverse (sucesor de `nfl_data_py`, que está archivado). Se
prefiere sobre el CSV crudo `nflverse/nfldata/games.csv` porque
`load_schedules()` SÍ trae moneyline (`away_moneyline`/`home_moneyline`),
que el CSV crudo no tiene -- confirmado directo contra los dos, no solo
contra la documentación (la documentación pública de `nfldata/games.csv`
en GitHub no menciona moneyline en absoluto).

CONFIRMADO CON DATOS REALES (probe(), 2026-08-20, no solo documentación):
- `nfl.load_schedules()` devuelve un DataFrame de **polars** (no pandas)
  con 46 columnas -- este script convierte a pandas inmediatamente
  (`.to_pandas()`) para mantener consistencia con el resto del proyecto.
  Requiere `pyarrow` instalado ademas de `nflreadpy` (dependencia de la
  conversion polars->pandas).
- Cobertura de temporadas: 1999-2026 (partidos + resultados). Tipos de
  partido incluidos: REG, WC, DIV, CON, SB (temporada regular + playoffs
  completos, Super Bowl incluido).
- **Cobertura de cuotas (moneyline/spread/total) NO es pareja en todo el
  rango -- confirmado por conteo real, no asumido**:
  - 2002-2005: 0 partidos con moneyline (cero cobertura).
  - 2006-2009: cobertura parcial e irregular (194 a 266 de 267 partidos/temporada).
  - 2010 en adelante: cobertura completa (267/267 o el total real de esa
    temporada) hasta la actualidad -- moneyline, spread_line Y total_line
    presentes en practicamente el 100% de los partidos.
  - 2026: temporada en curso, cobertura parcial obvia (112/272 partidos
    jugados a la fecha de esta corrida).
  **Decisión: el pipeline de limpieza (próximo script, no este) debe
  tratar 2010 en adelante como el rango confiable para cualquier
  metodología que dependa de cuotas** -- igual que MLS se trató distinto
  de las 4 ligas europeas por su propia limitación de datos, no se fuerza
  todo el histórico a la misma vara.
- **Formato de cuotas confirmado: americano (moneyline tipo -148/+124,
  spread_odds/over_odds/under_odds tipo -110/-108)** -- NO decimal como
  football-data.co.uk/tennis-data.co.uk/Sportmonks. La conversión a
  probabilidad implícita usa la fórmula estándar de odds americanas
  (positivo: 100/(odds+100); negativo: -odds/(-odds+100)), NO la fórmula
  de odds decimales (1/odds) que usa el resto del proyecto -- esto va en
  `clean_nfl_data.py` (próximo script), documentado explícitamente ahí
  para no mezclar las dos fórmulas por error.
- **PROVENANCIA DE LA CUOTA: NO CONFIRMADA.** A diferencia de
  football-data.co.uk (Pinnacle explícito) y Sportmonks, `load_schedules()`
  no trae ninguna columna que identifique la casa de apuestas o si es una
  cuota de apertura/cierre/consenso. La documentación pública de nflverse
  tampoco lo especifica. **Esto significa que la metodología de CLV
  (comparar apertura vs. cierre de UN libro sharp conocido) no se puede
  replicar 1:1 como en fútbol** hasta confirmar la fuente real -- se
  documenta como pregunta abierta, no se asume que es Pinnacle ni que es
  una cuota de cierre. El benchmark de mercado para Brier/blend sigue
  siendo válido (es una probabilidad de mercado real, venga de donde
  venga), pero cualquier conclusión de tipo "sharp money"/CLV necesita
  este dato confirmado primero.
- `spread_line`: positivo = el LOCAL es favorito por esa cantidad de
  puntos; negativo = el visitante es favorito. Confirmado con un ejemplo
  real (BAL @ KC, 2024-09-05: KC moneyline -148 favorito, spread_line=3.0
  positivo -- consistente).

Guarda un CSV por temporada en data/raw/NFL/ -- mismo patrón "un archivo
por temporada" que football_data_loader.py y tennis_data_loader.py, para
poder re-procesar sin volver a pegarle a la fuente.

Requiere: pip install nflreadpy pyarrow

Uso:
    python -m src.ingestion.nfl_data_loader --probe
    python -m src.ingestion.nfl_data_loader --download-all
    python -m src.ingestion.nfl_data_loader --download-all --start-season 2010 --end-season 2026
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

RAW_DATA_DIR = PROCESSED_DATA_DIR.parent / "raw"  # mismo criterio inferido que tennis_data_loader.py

# Confirmado por probe() real (2026-08-20): antes de esta temporada la cobertura de
# cuotas es 0 o muy irregular -- ver docstring. No es una opinion, es un conteo real.
RELIABLE_ODDS_START_SEASON = 2010


def _load_schedules_pandas():
    try:
        import nflreadpy as nfl
    except ImportError as e:
        raise ImportError(
            "Falta 'nflreadpy'. Instalar con: pip install nflreadpy pyarrow"
        ) from e
    return nfl.load_schedules().to_pandas()


def probe() -> None:
    """Confirma el esquema real y la cobertura de cuotas contra datos reales --
    mismo criterio de disciplina que tennis_data_loader.py/sportmonks_loader.py:
    nunca asumir, siempre confirmar antes de construir el resto del pipeline."""
    df = _load_schedules_pandas()
    print(f"Tipo tras .to_pandas(): {type(df)}")
    print(f"Shape: {df.shape}")
    print(f"\nColumnas ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}")

    print(f"\nRango de temporadas: {df['season'].min()}-{df['season'].max()}")
    print(f"Tipos de partido: {sorted(df['game_type'].unique())}")

    print(f"\nCobertura de cuotas por temporada (moneyline/spread/total):")
    coverage = df.groupby("season").agg(
        n_partidos=("game_id", "count"),
        n_moneyline=("home_moneyline", lambda s: s.notna().sum()),
        n_spread=("spread_line", lambda s: s.notna().sum()),
        n_total=("total_line", lambda s: s.notna().sum()),
    )
    print(coverage.to_string())

    reliable = df[df["season"] >= RELIABLE_ODDS_START_SEASON]
    pct_reliable_with_odds = reliable["home_moneyline"].notna().mean()
    print(f"\nDesde temporada {RELIABLE_ODDS_START_SEASON} (rango confiable elegido): "
          f"{pct_reliable_with_odds:.1%} de partidos con moneyline disponible.")

    print("\nMuestra real (5 partidos, ultima temporada completa disponible):")
    last_full_season = df.loc[df["home_score"].notna(), "season"].max()
    sample_cols = ["gameday", "away_team", "home_team", "away_score", "home_score",
                   "away_moneyline", "home_moneyline", "spread_line", "total_line"]
    print(df[df["season"] == last_full_season][sample_cols].head(5).to_string(index=False))


def download_all(start_season: int = 1999, end_season: int = 2026) -> None:
    df = _load_schedules_pandas()
    df = df[(df["season"] >= start_season) & (df["season"] <= end_season)]

    out_dir = RAW_DATA_DIR / "NFL"
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons_saved = 0
    for season, season_df in df.groupby("season"):
        out_path = out_dir / f"nfl_{season}.csv"
        season_df.to_csv(out_path, index=False)
        has_odds = season_df["home_moneyline"].notna().sum()
        print(f"Temporada {season}: {len(season_df)} partidos, {has_odds} con moneyline -> {out_path}")
        seasons_saved += 1

    print(f"\n{seasons_saved} temporadas guardadas en {out_dir}")
    print(f"[AVISO] Cobertura de cuotas confiable recien desde {RELIABLE_ODDS_START_SEASON} -- "
          f"ver probe()/docstring para el detalle real por temporada. El proximo script "
          f"(clean_nfl_data.py) decide como tratar las temporadas anteriores, no este.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--start-season", type=int, default=1999)
    parser.add_argument("--end-season", type=int, default=2026)
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.download_all:
        download_all(args.start_season, args.end_season)
    else:
        parser.print_help()