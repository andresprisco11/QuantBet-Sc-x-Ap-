"""
Fase 13 -- Tier 2 (parlays largos): motor de EV, correlacion y tasa de
crecimiento. Primera pieza real del mandato original de parlays de 10-25
piernas, construida ahora para tener el andamiaje entrenado y medido mucho
antes de que haya dinero de por medio.

### La pregunta que este modulo responde con numeros, no con opinion

Un parlay de piernas +EV, ¿escala el EV o lo destruye? Las dos intuiciones
que circulan son ambas incorrectas a medias:

  - "los parlays son trampa de casino" -- FALSO en general. Si cada pierna
    tiene +EV y la casa multiplica cuotas de forma ingenua (que es lo que
    hacen casi todas para combinadas entre partidos distintos), el EV
    COMPUESTA multiplicativamente: 15 piernas al +5% dan 1.05^15 - 1 =
    +108% de EV teorico. Eso es real.
  - "entonces conviene armar el parlay mas largo posible" -- tambien FALSO.
    El EV esperado no es el criterio correcto para decidir cuanto apostar
    ni que estructura usar: lo es la TASA DE CRECIMIENTO LOGARITMICO
    (Kelly). Y ahi la varianza del parlay largo pesa muchisimo.

Este modulo calcula las dos cosas sobre las MISMAS piernas reales que
detecta soft_book_edge.py, para que la decision de longitud de parlay salga
de una medicion y no de una corazonada.

### Correlacion: por que importa tanto y como se modela aca

Si dos piernas estan POSITIVAMENTE correlacionadas y la casa las precia
como independientes (multiplicando), la probabilidad conjunta REAL es mayor
que el producto -> la casa esta subvaluando el boleto -> hay edge extra que
no viene de las piernas sino de la estructura. Ese es exactamente el
"efecto domino" que persigue el mandato Tier 2, y es un exploit real y
documentado (correlated parlays).

La contracara honesta: las casas lo saben. Bloquean o reprecian las
combinadas del MISMO partido (same-game parlays) justamente por esto. La
correlacion explotable vive entre partidos distintos y es mucho mas
sutil (entorno de goles de una jornada, criterio arbitral, clima regional,
fatiga por calendario europeo).

Aca la correlacion se modela con una **copula gaussiana equicorrelacionada**:
se sortea Z ~ N(0, Sigma) con Sigma = (1-rho)I + rho*11', y la pierna i
gana si Z_i < Phi^-1(p_i). Esto genera Bernoullis correlacionadas
respetando exactamente las probabilidades marginales de cada pierna -- no
es una aproximacion inventada, es el metodo estandar.

rho=0 es independencia (el caso base, y lo unico que se puede asumir sin
evidencia). Los valores de rho>0 son ESCENARIOS, no mediciones: sirven
para responder "cuanta correlacion haria falta para que un parlay de 15
piernas convenga", que es una pregunta accionable. Medir el rho real de
verdad requiere historial de resultados conjuntos -- pendiente, y esta
marcado como pendiente.

Uso:
    python -m src.models.parlay_engine --file data/raw/THEODDSAPI/soft_book_edges_latest.csv
    python -m src.models.parlay_engine --file <csv> --rho 0.05
    python -m src.models.parlay_engine --file <csv> --max-legs 20 --sims 200000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

KELLY_FRACTION = 0.10      # mismo Kelly fraccional que el resto del proyecto
MAX_STAKE_FRACTION = 0.05


def parlay_metrics(probs, odds, rho=0.0, sims=100_000, seed=7):
    """Metricas de un boleto combinado.

    Devuelve la probabilidad REAL de acertarlo (con correlacion rho), la
    cuota combinada que paga la casa (producto -- precio ingenuo), el EV, y
    la tasa de crecimiento logaritmico a Kelly fraccional.

    rho=0 -> se usa la formula cerrada (producto). rho>0 -> Monte Carlo con
    copula gaussiana."""
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    n = len(probs)
    cuota_combinada = float(np.prod(odds))

    if rho <= 0 or n == 1:
        p_real = float(np.prod(probs))
    else:
        rng = np.random.default_rng(seed)
        # Copula gaussiana equicorrelacionada: Z = sqrt(rho)*F + sqrt(1-rho)*E
        # da exactamente corr(Z_i,Z_j) = rho, sin construir la matriz entera.
        factor = rng.standard_normal((sims, 1))
        idio = rng.standard_normal((sims, n))
        Z = np.sqrt(rho) * factor + np.sqrt(1 - rho) * idio
        umbrales = norm.ppf(probs)
        p_real = float(np.all(Z < umbrales, axis=1).mean())

    ev = p_real * cuota_combinada - 1.0
    kelly_full = ((p_real * cuota_combinada - 1.0) / (cuota_combinada - 1.0)
                  if cuota_combinada > 1 else 0.0)
    f = min(max(kelly_full * KELLY_FRACTION, 0.0), MAX_STAKE_FRACTION)
    # Tasa de crecimiento logaritmico esperada por boleto apostado con esa
    # fraccion. ES el criterio correcto para comparar estructuras -- el EV
    # solo ignora que una ruina no se recupera.
    if f > 0:
        growth = p_real * np.log(1 + f * (cuota_combinada - 1)) + (1 - p_real) * np.log(1 - f)
    else:
        growth = 0.0
    return {
        "n_legs": n, "p_real": p_real, "cuota_combinada": cuota_combinada,
        "ev": ev, "kelly_frac": f, "growth_por_boleto": float(growth),
    }


def medir_rho(serie: pd.Series, grupos: pd.Series):
    """Correlacion intraclase (ICC): cuanto se parecen entre si los
    resultados que caen en el MISMO grupo (jornada, semana), mas alla del
    promedio general. Es exactamente el rho que consume parlay_metrics(),
    pero estimado con datos reales en vez de supuesto."""
    tmp = pd.DataFrame({"v": serie.values, "g": grupos.values}).dropna()
    bloques = [g["v"].values for _, g in tmp.groupby("g") if len(g) >= 2]
    if not bloques:
        return None, 0
    todos = np.concatenate(bloques)
    mu, var = todos.mean(), todos.var()
    if var <= 0:
        return None, 0
    num = npares = 0
    for b in bloques:
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                num += (b[i] - mu) * (b[j] - mu)
                npares += 1
    return ((num / npares) / var, npares) if npares else (None, 0)


def modo_medir_rho(liga: str):
    """Estima el rho REAL entre partidos de una misma jornada sobre el
    historico limpio.

    MEDIDO el 2026-08-22 en LaLiga (1,950 partidos, 6 temporadas): todos
    los rho quedaron entre -0.0065 y +0.0042, indistinguibles de CERO.
    Conclusion: los resultados de partidos DISTINTOS de una misma jornada
    son, a efectos practicos, independientes -- y por lo tanto el optimo de
    crecimiento es 1 pierna, no un parlay largo."""
    from config.settings import PROCESSED_DATA_DIR
    path = PROCESSED_DATA_DIR / liga / "matches_clean.csv"
    if not path.exists():
        print(f"[SKIP] {liga}: no existe {path}")
        return
    df = pd.read_csv(path).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"\n=== {liga} -- correlacion real dentro de la misma jornada ===")
    mercados = {
        "gana el LOCAL": (df["FTR"] == "H").astype(float),
        "hay EMPATE": (df["FTR"] == "D").astype(float),
        "mas de 2.5 goles": ((df["FTHG"] + df["FTAG"]) > 2.5).astype(float),
        "ambos marcan": ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(float),
    }
    for etiqueta, serie in mercados.items():
        for nombre, g in [("dia", df["Date"].dt.to_period("D")),
                          ("semana", df["Date"].dt.to_period("W"))]:
            rho, npares = medir_rho(serie, g)
            if rho is not None:
                print(f"  {etiqueta:<20} [{nombre:<6}] rho = {rho:+.4f}  ({npares} pares)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="CSV de soft_book_edge.py")
    ap.add_argument("--measure-rho", action="store_true",
                    help="Estimar el rho REAL entre partidos de una jornada, sobre el historico.")
    ap.add_argument("--rho", type=float, default=0.0,
                    help="Correlacion entre piernas (0 = independientes, el unico "
                         "supuesto defendible sin evidencia). >0 es ESCENARIO.")
    ap.add_argument("--max-legs", type=int, default=15)
    ap.add_argument("--sims", type=int, default=100_000)
    args = ap.parse_args()

    if args.measure_rho:
        from config.settings import LEAGUES
        for liga in LEAGUES:
            modo_medir_rho(liga)
        print(f"\n[LECTURA] Un rho cercano a 0 significa independencia -> el optimo de "
              f"crecimiento es 1 pierna. Para que un parlay de 3 piernas convenga hace "
              f"falta rho >= ~0.05 (ver tabla de sensibilidad en el roadmap).")
        return

    if not args.file:
        ap.error("hace falta --file (o --measure-rho)")
    df = pd.read_csv(args.file)
    # Una pierna por PARTIDO: dos resultados del mismo partido son
    # mutuamente excluyentes, combinarlos da probabilidad cero.
    df = df.sort_values("edge", ascending=False).drop_duplicates("match")
    probs = df["fair_prob"].values
    odds = df["book_odds"].values
    edges = df["edge"].values

    print(f"Piernas +EV disponibles (una por partido): {len(df)}")
    print(f"  edge medio {edges.mean():.1%} | probabilidad justa media {probs.mean():.1%}")
    print(f"  correlacion asumida: rho = {args.rho}"
          f"{'  (independencia)' if args.rho == 0 else '  (ESCENARIO, no medido)'}\n")

    print(f"{'piernas':>8}{'prob. acertar':>16}{'cuota':>14}{'EV':>10}"
          f"{'stake':>8}{'crecim./boleto':>16}{'1 acierto cada':>16}")
    print("-" * 88)

    filas = []
    for n in range(1, min(args.max_legs, len(df)) + 1):
        m = parlay_metrics(probs[:n], odds[:n], rho=args.rho, sims=args.sims)
        cada = 1 / m["p_real"] if m["p_real"] > 0 else float("inf")
        print(f"{n:>8}{m['p_real']:>15.4%}{m['cuota_combinada']:>14.1f}"
              f"{m['ev']:>+10.1%}{m['kelly_frac']:>8.2%}"
              f"{m['growth_por_boleto']:>16.6f}"
              f"{(f'{cada:,.0f} boletos' if cada < 1e12 else 'nunca'):>16}")
        filas.append(m)

    res = pd.DataFrame(filas)
    mejor = res.loc[res["growth_por_boleto"].idxmax()]
    print(f"\n--> EV maximo con {int(res.loc[res['ev'].idxmax(),'n_legs'])} piernas "
          f"({res['ev'].max():+.1%})")
    print(f"--> CRECIMIENTO maximo con {int(mejor['n_legs'])} piernas "
          f"(growth {mejor['growth_por_boleto']:.6f}, prob. acertar {mejor['p_real']:.2%})")
    print(f"\n    El EV crece con la longitud pero el CRECIMIENTO no: son criterios "
          f"distintos.\n    Kelly optimiza el segundo porque una ruina no se recupera "
          f"con EV positivo.")

    print(f"\n[PENDIENTE REAL] rho todavia NO esta medido con datos: hace falta historial "
          f"de resultados conjuntos por jornada para estimarlo. Los rho>0 de este script "
          f"son escenarios para dimensionar cuanta correlacion haria falta, no evidencia "
          f"de que exista.")


if __name__ == "__main__":
    main()
