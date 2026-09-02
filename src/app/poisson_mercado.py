"""
Traduce los precios del mercado a una grilla de marcadores.

### Que es y que NO es

Tu mockup tenia "LIKELIEST SCORES" y una tabla de mercados (over 1.5, BTTS,
etc.). Se los saque porque solo exportabamos 1X2, y devolverlos con la salida
del modelo v4 seria mostrar numeros que ya demostramos que no valen.

Esta es la version honesta: **no predice nada**. Toma los precios que el
mercado YA pone (1X2 y totales), y busca el par de goles esperados
(lambda_local, lambda_visitante) que mejor reproduce esos precios bajo un
Poisson independiente. Despues expande esa grilla a marcadores exactos.

O sea: es el mercado, traducido a un formato legible. Si el mercado dice que
el over 2.5 paga 53% y el local gana 49%, la grilla es la consecuencia
aritmetica de esas dos cosas. No hay opinion nuestra adentro.

Por eso la etiqueta en la interfaz dice "mercado" y no "modelo".

### Limitacion declarada

El Poisson independiente subestima levemente los empates: los goles de los
dos equipos estan correlacionados negativamente en la practica (el que va
ganando se repliega). Existe el Poisson bivariado y la correccion de
Dixon-Coles para los marcadores bajos. No se aplican todavia porque
introducirian un parametro estimado por nosotros, y el punto de este modulo
es que NO haya nada nuestro adentro. El error tipico en 0-0, 1-1 y 2-2 es
del orden de 1-2 puntos porcentuales.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOLES = 10


def grilla(lam_h: float, lam_a: float) -> np.ndarray:
    """Matriz P[i,j] = probabilidad de marcador i-j bajo Poisson independiente."""
    ph = poisson.pmf(np.arange(MAX_GOLES + 1), lam_h)
    pa = poisson.pmf(np.arange(MAX_GOLES + 1), lam_a)
    return np.outer(ph, pa)


def _1x2(g: np.ndarray) -> tuple[float, float, float]:
    return float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())


def _over(g: np.ndarray, linea: float) -> float:
    i, j = np.indices(g.shape)
    return float(g[(i + j) > linea].sum())


def ajustar(p_home, p_draw, p_away, totales=None):
    """Encuentra (lam_h, lam_a) que reproduce los precios del mercado.

    `totales` es {linea: prob_over} desvigueado, opcional. Cuantas mas
    restricciones, mejor determinado queda el par -- con solo el 1X2 hay
    cierta indeterminacion en el total de goles."""
    objetivos = [("1x2", None, (p_home, p_draw, p_away))]
    for linea, p_over in (totales or {}).items():
        objetivos.append(("over", float(linea), float(p_over)))

    def error(x):
        lam_h, lam_a = np.exp(x)          # exp mantiene positivo sin restricciones
        g = grilla(lam_h, lam_a)
        e = 0.0
        for tipo, linea, obj in objetivos:
            if tipo == "1x2":
                h, d, a = _1x2(g)
                # el 1X2 pesa el doble: es el mercado mas liquido y mejor preciado
                e += 2.0 * ((h-obj[0])**2 + (d-obj[1])**2 + (a-obj[2])**2)
            else:
                e += (_over(g, linea) - obj) ** 2
        return e

    mejor, mejor_e = None, np.inf
    # varios puntos de arranque: la superficie tiene minimos locales suaves
    for h0 in (0.9, 1.3, 1.8):
        for a0 in (0.9, 1.3, 1.8):
            r = minimize(error, np.log([h0, a0]), method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 600})
            if r.fun < mejor_e:
                mejor, mejor_e = np.exp(r.x), r.fun
    return float(mejor[0]), float(mejor[1]), float(np.sqrt(mejor_e))


def derivar(lam_h: float, lam_a: float, top: int = 5,
            local: str = "local", visitante: str = "visitante") -> dict:
    """Expande la grilla a todo lo que la interfaz muestra."""
    g = grilla(lam_h, lam_a)
    i, j = np.indices(g.shape)

    marcadores = sorted(
        [{"s": f"{a}-{b}", "p": float(g[a, b])}
         for a in range(6) for b in range(6)],
        key=lambda d: -d["p"])[:top]

    btts = float(g[1:, 1:].sum())
    bets = [{"name": f"over {l} goals", "p": round(_over(g, l), 4)} for l in (1.5, 2.5, 3.5)]
    bets.append({"name": "both teams score", "p": round(btts, 4)})
    bets.append({"name": f"{local} 2+ goles", "p": round(float(g[2:, :].sum()), 4)})
    bets.append({"name": f"{visitante} 2+ goles", "p": round(float(g[:, 2:].sum()), 4)})
    bets.append({"name": f"{local} gana por 2+", "p": round(float(g[(i - j) >= 2].sum()), 4)})
    bets.append({"name": f"{visitante} gana por 2+", "p": round(float(g[(j - i) >= 2].sum()), 4)})

    return {"xg": [round(lam_h, 2), round(lam_a, 2)],
            "scores": marcadores, "bets": bets}
