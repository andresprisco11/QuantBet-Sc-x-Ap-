"""
Escalamiento a Tenis, paso 4 (tras tennis_data_loader.py, clean_tennis_data.py
y add_tennis_form_features.py -- los tres ya corridos y confirmados
2026-08-19: ATP 28,424 partidos / 871 jugadores, WTA 26,590 / 905 jugadores,
cobertura de features en rangos razonables, ver roadmap).

PRIMER MODELO PREDICTIVO DE TENIS. El resultado de tenis es binario, sin
empate -- exactamente el tipo de problema para el que Bill Benter diseñó su
regresión logística original en carreras de caballos (a diferencia de
fútbol, donde el proyecto usa Poisson de goles porque el resultado no es
binario). Por eso acá el "modelo de habilidad" (skill model) es una
regresión logística directa sobre las features de add_tennis_form_features.py,
no un Poisson.

METODOLOGÍA -- misma disciplina que poisson_model_v4.py en fútbol:
1. Walk-forward con ventana EXPANSIVA por año calendario: para predecir el
   año Y, se entrena solo con partidos de años ANTERIORES a Y (nunca del
   mismo año ni futuros) -- cero fuga temporal.
2. Modelo de habilidad (regresión logística) entrenado sobre las features
   diff (P1 - P2) de add_tennis_form_features.py, con NaN imputado a 0 más
   una columna indicadora "_missing" por feature (para que el modelo pueda
   aprender a no confiar en una feature que en realidad no existía para
   ese partido, en vez de que un 0 imputado se confunda con "diferencia
   real de cero").
3. Blend Benter Boost con el mercado: una SEGUNDA regresión logística,
   entrenada tambien solo con datos pasados, sobre [logit(prob_modelo),
   logit(prob_mercado)] -> resultado real. Esto replica el mismo principio
   que ya está validado en fútbol (el mercado es difícil de vencer solo
   con el modelo -- se combina, no se reemplaza). Donde no hay cuota de
   Pinnacle disponible (~7% de los partidos, ver clean_tennis_data.py), el
   blend cae de vuelta al modelo solo -- no se inventa una probabilidad de
   mercado que no existe.
4. Brier score OOS por año y agregado, para modelo/mercado/blend -- mismo
   framework de comparación que backtest_v4.py usa en fútbol.

FEATURES v3 (2026-08-19, tras confirmar que v2 -- log-ratios de ranking +
contexto estructural del partido -- NO movió el Brier respecto a v1,
Brier modelo prácticamente igual, 0.2189->0.2191 ATP / 0.2178->0.2183 WTA:
se agregan Elo general y Elo por superficie, actualizados walk-forward
partido a partido en add_tennis_form_features.py -- ver ese script para
el detalle. A diferencia de WinRate20/rank/pts, Elo pondera la FUERZA DEL
RIVAL vencido, que ninguna feature anterior capturaba. Requiere haber
vuelto a correr add_tennis_form_features.py (regenera matches_features.csv
con las columnas Elo_diff/Elo_Surface_diff) antes de este script.

FEATURES v2 (2026-08-19, tras confirmar winner's curse en
tennis_selection_bias_check.py y que el modelo apenas mejora sobre el
mercado -- Brier 0.218-0.219 vs. 0.202, ver roadmap): dos huecos reales
en las features v1 que valía la pena cerrar ANTES de aceptar "el mercado
de tenis es simplemente más eficiente" como conclusión final:
1. `rank_diff`/`pts_diff` (v1) eran DIFERENCIA LINEAL de ranking/puntos --
   una escala que no refleja la habilidad real: la diferencia entre el
   puesto 1 y el puesto 2 del ranking es un salto de nivel enorme, la
   diferencia entre el puesto 500 y el 501 es prácticamente ruido, y una
   resta lineal les da el mismo peso (diff=1 en los dos casos). Se
   reemplaza por `log_rank_ratio` = log(Player2_Rank / Player1_Rank) y
   `log_pts_ratio` = log(Player1_Pts / Player2_Pts) -- razones
   logarítmicas, no diferencias, exactamente el tipo de transformación
   que reconoce que el ranking es una escala de cola larga, no lineal.
2. El modelo v1 no usaba NINGUNA información directa del contexto del
   partido -- superficie, nivel de torneo (Grand Slam/Masters/etc.), tipo
   de cancha, o partido a 3 vs. 5 sets. `SurfaceWinRate_diff` (v1) es un
   proxy indirecto (rendimiento histórico del jugador en esa superficie)
   pero nunca se le decía al modelo EN QUÉ superficie/torneo/formato está
   el partido actual en sí. Se agrega one-hot de `Surface`/`Series`/
   `Court` (dummy_na=True -- un valor faltante se codifica como su propia
   categoría explícita, nunca se descarta silenciosamente) más
   `is_best_of_5` (partido a 5 sets, formato de Grand Slams en hombres --
   relevante porque cambia la dinámica de resistencia/remontada). Estas
   son features ESTRUCTURALES del partido (se conocen antes de jugarse,
   no dependen de ningún resultado) -- no requieren walk-forward, se
   calculan directo sobre el dataset completo sin riesgo de fuga.

FIX 2026-08-19 (encontrado en la primera corrida real, ver roadmap): el
modelo de habilidad no escalaba las features antes de entrenar la
regresión logística -- con features en escalas muy distintas (win rate
0-1, rank_diff/pts_diff en cientos o miles, DaysRest en decenas/cientos
de días), el solver lbfgs no llegaba a converger de verdad en 1000
iteraciones (ConvergenceWarning en los 9 años de ambos tours). No era
cosmético: un modelo que no convergió no está en su óptimo real, así que
los Brier reportados en esa corrida no eran confiables. Corregido con
StandardScaler (fit SOLO en el set de entrenamiento de cada año, transform
aplicado a train y test -- cero fuga) + max_iter subido a 2000 como margen
adicional.

NO SE INTEGRA TODAVÍA CON src.tracking.run_logger.log_run() -- el proyecto
tiene la regla de nunca adivinar un esquema/firma de función sin
confirmarlo primero (misma disciplina que RAW_DATA_DIR inferido en
tennis_data_loader.py), y la firma exacta de log_run() no está confirmada
en este script. Se deja como pendiente explícito, no como olvido -- ver
roadmap.

Requiere haber corrido antes:
    python -m src.processing.add_tennis_form_features --tours ATP,WTA

Salida: data/processed/TENNIS_ATP/predictions_v1.csv,
        data/processed/TENNIS_WTA/predictions_v1.csv
(una fila por partido OOS, con prob_modelo/prob_mercado/prob_blend MAS las
cuotas reales de Pinnacle de ambos jugadores (Player1_PS_Odds/
Player2_PS_Odds, con margen -- necesarias para calcular el pago real de
una apuesta) -- insumo para el economic_backtest de tenis, ver
economic_backtest_tennis.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR

DIFF_FEATURES = [
    "WinRate20_diff", "SurfaceWinRate_diff", "MatchesPlayed_diff",
    "DaysRest_diff", "H2H_WinRate_diff", "Elo_diff", "Elo_Surface_diff",
]

# Features v2 -- razones logaritmicas en vez de diferencia lineal (ver docstring arriba).
RATIO_FEATURE_PAIRS = [
    # (nombre_salida, columna_numerador, columna_denominador)
    ("log_rank_ratio", "Player2_Rank", "Player1_Rank"),  # positivo = Player1 mejor rankeado (rank mas bajo)
    ("log_pts_ratio", "Player1_Pts", "Player2_Pts"),      # positivo = Player1 con mas puntos
]

# Features v2 -- contexto estructural del partido (conocido antes de jugarse, sin fuga).
CATEGORICAL_MATCH_FEATURES = ["Surface", "Series", "Court"]

MIN_TRAIN_YEARS = 3     # años de historia minima antes de generar la primera prediccion OOS
MIN_TRAIN_ROWS = 200    # si hay menos filas de entrenamiento que esto para un año, se salta (muestra insuficiente)
PROB_CLIP = 1e-6        # evita log(0)/log(inf) en el logit


def _normalize_player1_won(df: pd.DataFrame) -> pd.DataFrame:
    """Misma normalizacion defensiva que add_tennis_form_features.py --
    no confiar en que pandas siempre infiera bool nativo al releer el CSV."""
    if df["Player1_Won"].dtype != bool:
        df["Player1_Won"] = (
            df["Player1_Won"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )
    return df


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """log(numerador/denominador) -- NaN si falta cualquiera de los dos o si
    alguno es <=0 (no deberia pasar con rank/puntos reales, pero se cubre
    en vez de dejar que log() explote con un valor corrupto silencioso)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((numerator > 0) & (denominator > 0), np.log(numerator / denominator), np.nan)
    return pd.Series(ratio, index=numerator.index)


def _prepare_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN a 0 en cada feature numerica + agrega una columna
    '_missing' por feature -- el modelo puede aprender a descontar una
    feature que en realidad no existia para ese partido, en vez de
    confundir un 0 imputado con una diferencia real de habilidad nula.
    Incluye las features diff walk-forward (v1), las razones logaritmicas
    de rank/puntos y el contexto estructural del partido (v2, ver
    docstring)."""
    X = pd.DataFrame(index=df.index)

    numeric_cols = list(DIFF_FEATURES)
    for name, num_col, den_col in RATIO_FEATURE_PAIRS:
        if num_col in df.columns and den_col in df.columns:
            df[name] = _log_ratio(df[num_col], df[den_col])
            numeric_cols.append(name)
        else:
            print(f"  [AVISO] faltan '{num_col}'/'{den_col}' -- no se calcula '{name}'.")

    for col in numeric_cols:
        if col not in df.columns:
            print(f"  [AVISO] falta la columna '{col}' -- se omite del modelo.")
            continue
        missing = df[col].isna()
        X[col] = df[col].fillna(0.0)
        X[f"{col}_missing"] = missing.astype(int)

    # -- Best_of: partido a 5 sets (Grand Slam masculino) vs. a 3 --
    if "Best_of" in df.columns:
        best_of_numeric = pd.to_numeric(df["Best_of"], errors="coerce")
        X["is_best_of_5"] = (best_of_numeric == 5).astype(int)
    else:
        print("  [AVISO] falta la columna 'Best_of' -- no se calcula 'is_best_of_5'.")

    # -- contexto estructural categorico: one-hot, dummy_na=True para no descartar
    #    partidos con superficie/torneo/cancha faltante, se codifican como su propia
    #    categoria explicita en vez de perderse silenciosamente. --
    present_categorical = [c for c in CATEGORICAL_MATCH_FEATURES if c in df.columns]
    missing_categorical = [c for c in CATEGORICAL_MATCH_FEATURES if c not in df.columns]
    if missing_categorical:
        print(f"  [AVISO] faltan columnas categoricas {missing_categorical} -- se omiten del modelo.")
    if present_categorical:
        dummies = pd.get_dummies(df[present_categorical], columns=present_categorical, dummy_na=True)
        X = pd.concat([X, dummies.astype(int)], axis=1)

    return X


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def run(tour: str) -> None:
    print(f"\n=== {tour.upper()} ===")
    in_path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "matches_features.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe {in_path}. Corre 'python -m src.processing.add_tennis_form_features --tours {tour}' primero."
        )

    df = pd.read_csv(in_path, parse_dates=["Date"], low_memory=False)
    df = _normalize_player1_won(df)
    df = df.sort_values("Date").reset_index(drop=True)
    df["year"] = df["Date"].dt.year

    y_all = df["Player1_Won"].astype(int).to_numpy()
    market_prob_all = df["no_vig_prob_player1"].to_numpy() if "no_vig_prob_player1" in df.columns else np.full(len(df), np.nan)
    has_market_all = ~np.isnan(market_prob_all)

    X_all = _prepare_model_matrix(df)
    feature_cols = list(X_all.columns)

    years = sorted(df["year"].unique())
    first_test_year = years[0] + MIN_TRAIN_YEARS
    test_years = [y for y in years if y >= first_test_year]

    print(f"Cargados {len(df)} partidos ({years[0]}-{years[-1]}). "
          f"Ventana de entrenamiento minima: {MIN_TRAIN_YEARS} anios -- primer anio OOS: {first_test_year}.")

    results = []  # filas de salida (OOS)
    year_summary = []

    for test_year in test_years:
        train_mask = (df["year"] < test_year).to_numpy()
        test_mask = (df["year"] == test_year).to_numpy()

        n_train = int(train_mask.sum())
        if n_train < MIN_TRAIN_ROWS:
            print(f"  [SKIP] {test_year}: solo {n_train} partidos de entrenamiento (< {MIN_TRAIN_ROWS}), se salta.")
            continue

        # -- 1. modelo de habilidad (regresion logistica sobre features del jugador) --
        # FIX 2026-08-19: la primera corrida (sin escalar) tiro ConvergenceWarning en
        # TODOS los anios -- las features tienen escalas muy distintas (win rate 0-1,
        # rank_diff/pts_diff en cientos o miles, DaysRest en decenas/cientos de dias),
        # lo que hace que lbfgs no converja de verdad en 1000 iteraciones. Sin esto, el
        # modelo de habilidad NO estaba entrenado hasta el optimo real -- los Brier de
        # esa corrida no son confiables. Escalar (fit SOLO con datos de entrenamiento,
        # transform aplicado a train y test -- cero fuga) resuelve la causa real, no
        # solo el sintoma del warning.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_all.loc[train_mask])
        X_test_scaled = scaler.transform(X_all.loc[test_mask])

        skill_model = LogisticRegression(max_iter=2000)
        skill_model.fit(X_train_scaled, y_all[train_mask])

        model_prob_train = skill_model.predict_proba(X_train_scaled)[:, 1]
        model_prob_test = skill_model.predict_proba(X_test_scaled)[:, 1]

        # -- 2. blend Benter Boost con el mercado, entrenado SOLO con partidos
        #    pasados que tenian cuota de Pinnacle disponible --
        train_has_market = has_market_all[train_mask]
        blend_model = None
        if train_has_market.sum() >= MIN_TRAIN_ROWS:
            blend_X_train = np.column_stack([
                _logit(model_prob_train[train_has_market]),
                _logit(market_prob_all[train_mask][train_has_market]),
            ])
            blend_model = LogisticRegression(max_iter=2000)
            blend_model.fit(blend_X_train, y_all[train_mask][train_has_market])
        else:
            print(f"  [AVISO] {test_year}: solo {int(train_has_market.sum())} partidos de entrenamiento con cuota "
                  f"de mercado (< {MIN_TRAIN_ROWS}) -- blend no se entrena este anio, blend_prob = model_prob.")

        test_has_market = has_market_all[test_mask]
        blend_prob_test = model_prob_test.copy()
        if blend_model is not None and test_has_market.any():
            blend_X_test = np.column_stack([
                _logit(model_prob_test[test_has_market]),
                _logit(market_prob_all[test_mask][test_has_market]),
            ])
            blend_prob_test[test_has_market] = blend_model.predict_proba(blend_X_test)[:, 1]

        # -- guardar filas OOS --
        # FIX 2026-08-19: se agregan las cuotas REALES de Pinnacle (con margen,
        # no la prob no-vig) -- sin esto un futuro backtest economico no puede
        # calcular el pago real de una apuesta, solo tendria probabilidades.
        output_cols = ["Date", "Tournament", "Surface", "Player1", "Player2", "Player1_Won"]
        for odds_col in ["Player1_PS_Odds", "Player2_PS_Odds"]:
            if odds_col in df.columns:
                output_cols.append(odds_col)
        test_df = df.loc[test_mask, output_cols].copy()
        test_df["model_prob_player1"] = model_prob_test
        test_df["market_prob_player1"] = market_prob_all[test_mask]
        test_df["blend_prob_player1"] = blend_prob_test
        test_df["has_market"] = test_has_market
        results.append(test_df)

        # -- Brier por año --
        y_test = y_all[test_mask]
        brier_model = _brier(model_prob_test, y_test)
        brier_blend = _brier(blend_prob_test, y_test)
        if test_has_market.any():
            brier_market = _brier(market_prob_all[test_mask][test_has_market], y_test[test_has_market])
        else:
            brier_market = np.nan

        year_summary.append({
            "year": test_year, "n_bets": int(test_mask.sum()), "n_train": n_train,
            "brier_model": brier_model, "brier_market": brier_market, "brier_blend": brier_blend,
        })

    if not results:
        print(f"  [AVISO] {tour}: no se genero ninguna prediccion OOS -- revisar MIN_TRAIN_YEARS/MIN_TRAIN_ROWS.")
        return

    out_df = pd.concat(results, ignore_index=True)
    out_path = PROCESSED_DATA_DIR / f"TENNIS_{tour.upper()}" / "predictions_v1.csv"
    out_df.to_csv(out_path, index=False)

    summary_df = pd.DataFrame(year_summary)
    print(f"\nResumen Brier OOS por año ({tour.upper()}, {len(feature_cols)} features "
          f"de entrada al modelo de habilidad):")
    with pd.option_context("display.width", 140):
        print(summary_df.to_string(index=False, formatters={
            "brier_model": "{:.4f}".format, "brier_market": "{:.4f}".format, "brier_blend": "{:.4f}".format,
        }))

    total_n = int(out_df["Player1_Won"].shape[0])
    agg_brier_model = _brier(out_df["model_prob_player1"].to_numpy(), out_df["Player1_Won"].astype(int).to_numpy())
    agg_brier_blend = _brier(out_df["blend_prob_player1"].to_numpy(), out_df["Player1_Won"].astype(int).to_numpy())
    has_mkt = out_df["has_market"].to_numpy()
    agg_brier_market = _brier(out_df.loc[has_mkt, "market_prob_player1"].to_numpy(),
                               out_df.loc[has_mkt, "Player1_Won"].astype(int).to_numpy()) if has_mkt.any() else np.nan

    print(f"\nAgregado OOS completo ({total_n} partidos, {years[-1] - first_test_year + 1} años):")
    print(f"  Brier modelo:  {agg_brier_model:.4f}")
    print(f"  Brier mercado: {agg_brier_market:.4f}  (solo partidos con cuota de Pinnacle, n={int(has_mkt.sum())})")
    print(f"  Brier blend:   {agg_brier_blend:.4f}")
    print(f"  Gap (blend - mercado): {agg_brier_blend - agg_brier_market:+.4f}  "
          f"({'blend peor que el mercado solo -- esperado, mismo patron que futbol' if agg_brier_blend > agg_brier_market else 'blend mejor que el mercado solo -- inusual, revisar'})")
    print(f"\nGuardado -> {out_path} ({len(out_df)} partidos OOS, {len(out_df.columns)} columnas)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Primer modelo predictivo de tenis: regresion logistica walk-forward + blend Benter Boost con el mercado."
    )
    parser.add_argument("--tours", type=str, default="ATP,WTA", help="Tours a correr, separados por coma (default: ATP,WTA).")
    args = parser.parse_args()

    for tour in args.tours.split(","):
        try:
            run(tour)
        except FileNotFoundError as e:
            print(f"[SKIP] {tour}: {e}")