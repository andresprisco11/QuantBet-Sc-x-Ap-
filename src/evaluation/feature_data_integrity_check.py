"""
Fase 8, siguiente paso tras established_team_breakdown_check.py.

Resultado real de ese script (2026-08-18): en Serie A/Bundesliga, el
colapso de win_rate en "Recientes" (2324-2526) es AMPLIO, no concentrado.

- Serie A: 13 equipos distintos con >=3 apuestas en 'Recientes' (102 de 107
  apuestas cubiertas), win rates dispersos de 0% (Sassuolo, n=3) a 85.71%
  (Milan, n=7) -- ni siquiera los equipos GRANDES se salvan: Napoli 25%
  (n=12), Inter 28.57% (n=7), Fiorentina 33.33% (n=12). Solo Milan y Roma
  (muestras chicas, n=7 y n=4) rinden claramente bien.
- Bundesliga: 12 equipos distintos con >=3 apuestas (78 de 85 apuestas
  cubiertas), mismo patron disperso -- Dortmund, Hoffenheim, Leverkusen,
  Wolfsburg, FC Koln todos rondando 33%, RB Leipzig 22.22%, Union Berlin
  12.50%, Mainz 0%. Solo M'gladbach (n=7), Freiburg (n=4) y Ein Frankfurt
  (n=4) rinden por encima del promedio historico.
- Contraste con el grupo de control: en EPL/La Liga tambien hay dispersion
  equipo a equipo, pero el promedio GLOBAL no colapsa (EPL incluso mejora),
  y no hay evidencia de que la liga entera rinda mal simultaneamente.

Conclusion de established_team_breakdown_check.py: el problema NO es
idiosincratico de 2-3 clubes puntuales (no hay un "villano" concreto que
excluir quirurgicamente) -- es un patron que afecta a la liga casi entera
al mismo tiempo. Eso descarta el fix quirurgico (excluir equipos) y sube de
prioridad la hipotesis (c) del roadmap: el modelo v4 (recencia por tiros al
arco) puede tener un problema real y sistemico especificamente en Serie A/
Bundesliga en las ultimas 3 temporadas -- no un problema de que EQUIPO se
elige, sino de que tan bien la feature de tiros al arco sigue prediciendo
ahi.

Antes de saltar a gastar en Sportmonks (features nuevas, xG) solo por
descarte de proceso, hay una pregunta mas barata y mas basica que probar
primero, siguiendo la misma disciplina que ya encontro 6 bugs reales en el
proyecto (ver "Incidentes de integridad de codigo" en el roadmap): ¿los
DATOS CRUDOS de tiros al arco (HST/AST, columnas ya confirmadas en
matches_clean.csv desde Fase 8) tienen algun problema de calidad/cobertura
especificamente en Serie A/Bundesliga en las temporadas recientes? Por
ejemplo: mas valores faltantes, valores en cero sospechosos, o un cambio
de distribucion que no coincide con nada futbolisticamente real (indicio
de un cambio silencioso en la fuente de datos de football-data.co.uk).

Esto NO asume que hay un bug -- lo mide directamente, por temporada, en
las 4 ligas europeas, sobre las columnas crudas HST/AST de
matches_clean.csv (no las features derivadas, para no depender de nombres
de columna no confirmados en este entorno -- el script busca ademas,
defensivamente, cualquier columna que contenga 'shot' o 'hst'/'ast' en el
CSV de predicciones OOS, y reporta lo que encuentra sin asumir nombres).

Metricas por temporada y liga:
- % de partidos con HST o AST faltante o en cero.
- Media y desvio estandar de HST+AST (tiros al arco totales del partido).
- Un cambio de nivel abrupto entre temporadas (delta de la media respecto
  a la temporada anterior) que no tenga una razon futbolistica obvia es la
  señal de alerta, no la conclusion final -- este script solo mide y
  reporta, no interpreta causalidad futbolistica.

Salida: data/runs/feature_data_integrity_check.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import LEAGUES, PROCESSED_DATA_DIR


def _raw_shots_integrity(league_key: str) -> list:
    path = PROCESSED_DATA_DIR / league_key / "matches_clean.csv"
    if not path.exists():
        print(f"[SKIP] {league_key}: no existe {path}.")
        return []

    df = pd.read_csv(path)
    if "HST" not in df.columns or "AST" not in df.columns:
        print(f"[AVISO] {league_key}: no hay columnas HST/AST en matches_clean.csv.")
        return []

    df["season"] = df["season"].astype(str)
    seasons = sorted(df["season"].unique())

    rows = []
    print(f"\n=== {league_key} -- integridad de HST/AST (tiros al arco crudos) por temporada ===")
    print(f"{'temporada':10s} {'n_partidos':>10s} {'%faltante':>10s} {'%en_cero':>9s} "
          f"{'media_HST+AST':>14s} {'std_HST+AST':>12s}")
    prev_mean = None
    for season in seasons:
        season_df = df[df["season"] == season]
        n = len(season_df)
        missing = season_df["HST"].isna() | season_df["AST"].isna()
        pct_missing = missing.mean()
        valid = season_df.loc[~missing]
        total_shots = valid["HST"] + valid["AST"]
        pct_zero = (total_shots == 0).mean() if len(valid) > 0 else float("nan")
        mean_shots = total_shots.mean() if len(valid) > 0 else float("nan")
        std_shots = total_shots.std() if len(valid) > 0 else float("nan")

        delta_str = ""
        if prev_mean is not None and pd.notna(mean_shots) and pd.notna(prev_mean):
            delta = mean_shots - prev_mean
            delta_str = f"  (delta vs. temporada anterior: {delta:+.2f})"
        prev_mean = mean_shots if pd.notna(mean_shots) else prev_mean

        print(f"{season:10s} {n:10d} {pct_missing:10.2%} {pct_zero:9.2%} "
              f"{mean_shots:14.2f} {std_shots:12.2f}{delta_str}")
        rows.append({
            "league_key": league_key, "temporada": season, "n_partidos": n,
            "pct_faltante": pct_missing, "pct_en_cero": pct_zero,
            "media_hst_ast": mean_shots, "std_hst_ast": std_shots,
        })

    return rows


def _scan_predictions_for_shot_features(league_key: str) -> None:
    """Busqueda defensiva: reporta que columnas relacionadas a tiros existen
    en el CSV de predicciones OOS, sin asumir nombres exactos no confirmados."""
    path = PROCESSED_DATA_DIR / league_key / "model_predictions_oos_walkforward_v4.csv"
    if not path.exists():
        return
    try:
        cols = pd.read_csv(path, nrows=1).columns.tolist()
    except Exception as e:
        print(f"  [AVISO] no se pudo leer encabezado de {path}: {e}")
        return
    shot_like = [c for c in cols if any(k in c.lower() for k in ["shot", "hst", "ast", "hs_", "as_"])]
    if shot_like:
        print(f"  Columnas relacionadas a tiros encontradas en predicciones OOS de {league_key}: {shot_like}")
    else:
        print(f"  [AVISO] {league_key}: ninguna columna con 'shot'/'hst'/'ast' en el nombre "
              f"encontrada en predicciones OOS -- no se puede inspeccionar la feature derivada "
              f"directamente, solo los datos crudos (arriba).")


def run() -> None:
    all_rows = []
    for league_key in LEAGUES.keys():
        all_rows.extend(_raw_shots_integrity(league_key))
        _scan_predictions_for_shot_features(league_key)

    if not all_rows:
        print("\n[AVISO] No se pudo evaluar ninguna liga.")
        return

    out_df = pd.DataFrame(all_rows)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "runs" / "feature_data_integrity_check.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nGuardado -> {out_path}")
    print("\nLectura: comparar SERIEA/BUNDESLIGA contra EPL/LALIGA (control) en las temporadas "
          "2324-2526. Si %faltante o %en_cero sube de forma notoria, o la media de HST+AST tiene "
          "un salto brusco sin explicacion obvia (ej. cambio de metodologia de conteo de la fuente), "
          "eso es un candidato a bug de datos silencioso, no un problema del modelo -- se corrige "
          "ahi, no con features nuevas. Si las 4 ligas se ven parejas en estas metricas, la integridad "
          "de datos queda descartada y el candidato principal pasa a ser la hipotesis (c): el modelo "
          "v4 (recencia por tiros al arco) tiene un techo real y especifico en Serie A/Bundesliga que "
          "datos de mejor calidad (xG, Sportmonks) podrian resolver.")
    print("\nNo se loggea en el sistema de tracking (diagnostico exploratorio) -- mismo criterio "
          "que established_team_breakdown_check.py.")


if __name__ == "__main__":
    run()