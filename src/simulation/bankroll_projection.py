"""
Fase 15 -- Proyeccion de banca: ¿a donde lleva todo esto, en numeros?

### La pregunta que responde

Ya sabemos QUE detectamos (precios malos en casas blandas contra Pinnacle) y
COMO se valida (CLV). Falta lo tercero: si esto funciona, ¿a que ritmo
crece la banca, y de que depende ese ritmo?

Este modulo NO predice el futuro. Simula la banca bajo supuestos explicitos,
usando la distribucion REAL de edges y probabilidades que detecta
soft_book_edge.py, para separar tres cosas que la intuicion mezcla:

  1. lo que el sistema puede dar si el edge es real
  2. cuanta varianza hay que aguantar en el camino
  3. cual de las palancas mueve de verdad la aguja

### El supuesto que lo domina todo, dicho de frente

Toda la simulacion asume **que el edge medido contra Pinnacle es real**.
Si el CLV resulta negativo, nada de esto aplica y el resultado correcto es
0 (o negativo). Por eso el CLV va primero y esto va despues: esto dimensiona
el premio, no lo demuestra.

Uso:
    python -m src.simulation.bankroll_projection --file data/raw/THEODDSAPI/soft_book_edges_latest.csv
    python -m src.simulation.bankroll_projection --file <csv> --apuestas-semana 200 --semanas 52
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def simular(probs, cuotas, stakes, apuestas_semana, semanas, sims=4000, seed=13):
    """Monte Carlo de la banca. Cada semana se muestrean `apuestas_semana`
    oportunidades (con reemplazo) de la distribucion real detectada, se
    apuesta la fraccion de Kelly correspondiente y se acumula.

    Las apuestas de una semana se resuelven de forma INDEPENDIENTE -- es lo
    que midio parlay_engine.py sobre 6 temporadas reales (rho ~ 0), asi que
    no es un supuesto comodo sino el resultado de una medicion."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    bancas = np.ones(sims)
    pico = np.ones(sims)
    max_dd = np.zeros(sims)
    trayectoria = np.zeros((semanas + 1, sims))
    trayectoria[0] = 1.0

    for semana in range(semanas):
        idx = rng.integers(0, n, size=(sims, apuestas_semana))
        p = probs[idx]
        o = cuotas[idx]
        f = stakes[idx]
        gana = rng.random((sims, apuestas_semana)) < p
        # Retorno de cada apuesta como fraccion de la banca del momento.
        # Se suma dentro de la semana (apuestas simultaneas, no secuenciales)
        # -- es lo realista: los partidos de un fin de semana se juegan a la vez.
        ret = np.where(gana, f * (o - 1.0), -f).sum(axis=1)
        bancas = bancas * (1.0 + ret)
        bancas = np.maximum(bancas, 1e-9)  # ruina numerica
        pico = np.maximum(pico, bancas)
        max_dd = np.maximum(max_dd, 1.0 - bancas / pico)
        trayectoria[semana + 1] = bancas

    return bancas, max_dd, trayectoria


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--apuestas-semana", type=int, default=None)
    ap.add_argument("--semanas", type=int, default=52)
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    probs = df["fair_prob"].values
    cuotas = df["book_odds"].values
    stakes = df["kelly_stake_frac"].values
    edges = df["edge"].values

    print(f"Distribucion REAL detectada ({len(df)} oportunidades):")
    print(f"   edge medio            : {edges.mean():.2%}")
    print(f"   probabilidad justa    : {probs.mean():.1%}")
    print(f"   stake Kelly medio     : {stakes.mean():.3%} de la banca")
    print(f"   EV por apuesta        : {(probs*cuotas-1).mean():.2%}")
    print(f"   EV en banca por apuesta: {(stakes*(probs*cuotas-1)).mean():.4%}\n")

    escenarios = ([args.apuestas_semana] if args.apuestas_semana
                  else [20, 60, 150, 300])

    print(f"Proyeccion a {args.semanas} semanas ({args.sims} simulaciones).")
    print("Banca inicial = 1.00. Las apuestas de cada semana se resuelven en "
          "paralelo e independientes\n(rho~0, medido sobre 6 temporadas reales).\n")
    print(f"{'apuestas/sem':>13}{'mediana':>10}{'media':>10}{'p10':>9}{'p90':>10}"
          f"{'% pierde':>10}{'caida max':>11}")
    print("-" * 73)

    for n_sem in escenarios:
        finales, dd, _ = simular(probs, cuotas, stakes, n_sem, args.semanas, args.sims)
        print(f"{n_sem:>13}{np.median(finales):>10.2f}{finales.mean():>10.2f}"
              f"{np.percentile(finales,10):>9.2f}{np.percentile(finales,90):>10.2f}"
              f"{(finales<1).mean():>10.1%}{np.median(dd):>11.1%}")

    print(f"\n(mediana 2.00 = la banca se duplico. 'caida max' = peor bajon "
          f"tipico en el camino,\n la mediana entre simulaciones -- lo que hay "
          f"que aguantar sin cambiar el plan.)")

    # --- Que palanca mueve de verdad la aguja ---
    print(f"\n\n### Sensibilidad: que pasa si cambia cada supuesto ###")
    base = 150
    finales_base, _, _ = simular(probs, cuotas, stakes, base, args.semanas, args.sims)
    med_base = np.median(finales_base)
    print(f"Base: {base} apuestas/semana, {args.semanas} semanas -> mediana {med_base:.2f}x\n")

    print(f"{'cambio':<46}{'mediana':>10}{'vs base':>10}")
    print("-" * 66)
    for etiqueta, kw in [
        ("volumen x2 (300/sem)", dict(n=300)),
        ("volumen /2 (75/sem)", dict(n=75)),
        ("stake x2 (mas Kelly)", dict(mult_stake=2.0)),
        ("stake x4", dict(mult_stake=4.0)),
        ("el edge real es la MITAD de lo medido", dict(mult_edge=0.5)),
        ("el edge real es CERO (Shin sesgado)", dict(mult_edge=0.0)),
    ]:
        n = kw.get("n", base)
        s = stakes * kw.get("mult_stake", 1.0)
        p = probs.copy()
        if "mult_edge" in kw:
            # Reducir el edge = bajar la probabilidad real manteniendo la cuota
            ev_obj = (probs * cuotas - 1.0) * kw["mult_edge"]
            p = (1.0 + ev_obj) / cuotas
        f, _, _ = simular(p, cuotas, s, n, args.semanas, args.sims)
        m = np.median(f)
        print(f"{etiqueta:<46}{m:>10.2f}{(m/med_base-1):>+9.0%}")

    print(f"\n[LECTURA] Si la fila 'edge real = CERO' da ~1.00 o menos, esta "
          f"todo dicho:\nsin edge real no hay crecimiento, por mucho volumen o "
          f"stake que se ponga.\nPor eso el CLV es la prioridad y no un tramite.")


if __name__ == "__main__":
    main()
