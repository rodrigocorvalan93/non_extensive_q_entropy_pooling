#%%
"""
s_main_verificaciones.py
========================
Verificaciones numéricas que cita el paper (Sección 4.3 y Apéndice B).

Todo lo que el paper afirma haber comprobado numéricamente sale de este
script, para que cualquiera pueda rehacerlo desde un clon del repositorio:

  1. Barrido de q sobre la base completa de Meucci (100.000 escenarios):
     error relativo ||p̃_q − p|| / ||p|| para Shannon y para Tsallis en una
     grilla de q; q que minimiza el error; comparación con los valores
     obtenidos con la implementación en GNU Octave (Tabla 1 y Figura 1).
  2. q = 2 contra una proyección euclídea calculada con un solver de
     programación cuadrática independiente (cvxpy / OSQP), sobre la base
     reducida de 1.000 escenarios.
  3. Números efectivos de escenarios de orden 1 (entropía; Meucci, 2012) y
     de orden 2 (Kish) a lo largo de la grilla de q: cada q maximiza el
     número efectivo de su propio orden.
  4. Límite q → 1: el posterior de Tsallis tiende al de Shannon.
  5. Rényi y Tsallis dan el mismo posterior: se resuelve el problema PRIMAL
     de Rényi con un optimizador genérico (SLSQP), sin usar la solución dual
     de Tsallis, y se comparan los resultados.
  6. Views aleatorios con cinco semillas: q que minimiza el error euclídeo.
  7. Tiempos de resolución con 100.000 escenarios.

Salidas
-------
  results/verificaciones.json      todos los números
  results/barrido_q.csv            el barrido completo
  paper/figs/optimal_q_zoom_{es,en}.png

View usado (el mismo de s_main_optimal_q.py): Lower=3, Upper=2 (índices
0-based) y ViewMethod=1,4, es decir la desigualdad
    1,4 · E[X_3] <= E[X_4]      (activos 3 y 4 en numeración 1-based),
con confianza c = 0,5 para el barrido.

cvxpy es opcional: si no está instalado se omite la verificación 2.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from entropy_pooling_v2 import entropy_prog
from s_main_optimal_q import load_mat_data, views_generator, sweep_q, sweep_q_random_seeds

RESULTS_DIR = HERE / "results"
FIGS_DIR = HERE / "paper" / "figs"
MAT_FULL = HERE / "octave-matlab-versions" / "ReturnsDistribution.mat"
MAT_SHORT = HERE / "ReturnsDistributionShort.mat"

LOWER, UPPER, VIEW_METHOD, CONFIDENCE = 3, 2, 1.4, 0.5
Q_TABLE = [1.5, 1.9, 2.0, 2.1, 2.5]
SEEDS = [42, 123, 777, 2024, 314]

# Valores obtenidos con la implementación en GNU Octave
# (octave-matlab-versions/S_MAIN.m, misma base y mismo view).
OCTAVE = {"Shannon": 0.046042, 1.5: 0.045968, 1.9: 0.045945,
          2.0: 0.045944, 2.1: 0.045945, 2.5: 0.045969}


def restricciones(X):
    J = X.shape[0]
    V = views_generator(X, LOWER, UPPER, VIEW_METHOD).reshape(-1, 1)
    return V.T, np.zeros(1), np.ones((1, J)), np.array([1.0])


def n_efectivo(x, orden):
    """Número efectivo de escenarios (número de Hill) de orden 1 o 2."""
    y = x[x > 0]
    if orden == 1:
        return float(np.exp(-np.sum(y * np.log(y))))
    return float(1.0 / np.sum(y ** 2))


# ═══════════════════════════════════════════════════════════════
# 1. Barrido sobre la base completa
# ═══════════════════════════════════════════════════════════════

def barrido_base_completa(out):
    X, p = load_mat_data(str(MAT_FULL))
    p = p / p.sum()
    J = p.size
    q_grid = np.unique(np.round(np.concatenate([
        np.arange(1.1, 3.01, 0.1), np.arange(1.8, 2.201, 0.02)]), 4))
    q_vals, errors, err_sh, q_opt, idx = sweep_q(
        X, p, LOWER, UPPER, ViewMethod=VIEW_METHOD, q_values=q_grid,
        confidence=CONFIDENCE, verbose=False)
    tabla = {"Shannon": err_sh}
    for q in Q_TABLE:
        tabla[q] = float(errors[np.argmin(np.abs(q_vals - q))])
    dif_octave = max(abs(tabla[k] - OCTAVE[k]) for k in OCTAVE)
    out["base_completa"] = {
        "J": int(J), "N": int(X.shape[1]),
        "error_shannon": err_sh,
        "errores_tabla": {str(k): v for k, v in tabla.items()},
        "octave": {str(k): v for k, v in OCTAVE.items()},
        "max_dif_python_octave": dif_octave,
        "q_optimo": q_opt,
        "error_minimo": float(errors[idx]),
        "mejora_pct_q2_vs_shannon": float((err_sh - tabla[2.0]) / err_sh * 100),
    }
    print(f"1) Base completa (J={J}): Shannon {err_sh:.6f} | q*={q_opt:.2f} "
          f"err={errors[idx]:.6f} | mejora de q=2: {out['base_completa']['mejora_pct_q2_vs_shannon']:.2f}% "
          f"| máx |Python − Octave| = {dif_octave:.1e}")
    pd.DataFrame({"q": q_vals, "error_relativo": errors}).to_csv(
        RESULTS_DIR / "barrido_q.csv", index=False, float_format="%.8f")

    # Tiempos (mediana de 5 corridas)
    A, b, Aeq, beq = restricciones(X)
    tiempos = {}
    for nombre, kw in (("Shannon", dict(entropy_family="S")),
                       ("Tsallis q=2", dict(entropy_family="T", q=2.0))):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            entropy_prog(p, A, b, Aeq, beq, **kw)
            ts.append(time.perf_counter() - t0)
        tiempos[nombre] = float(np.median(ts) * 1000)
    # Los tiempos dependen de la máquina y de su carga: se imprimen, y en el
    # archivo de resultados queda solo si se cumple la cota que cita el paper.
    out["resuelve_en_menos_de_1s_J100k"] = bool(max(tiempos.values()) < 1000)
    print(f"7) Tiempo de resolución con J={J}: Shannon {tiempos['Shannon']:.0f} ms, "
          f"Tsallis q=2 {tiempos['Tsallis q=2']:.0f} ms (incluye el arranque desde Shannon)")

    # Views aleatorios
    rs = sweep_q_random_seeds(X, p, LOWER, UPPER, seeds=SEEDS,
                              q_values=np.round(np.arange(1.1, 3.01, 0.1), 2),
                              confidence=CONFIDENCE, verbose=False)
    out["views_aleatorios_q_optimo"] = {str(s): float(rs["all_q_opt"][s]) for s in SEEDS}
    print(f"6) Views aleatorios, q que minimiza el error euclídeo por semilla: "
          f"{out['views_aleatorios_q_optimo']}")
    return q_vals, errors, err_sh, q_opt, idx


def figura_barrido(q_vals, errors, err_sh, q_opt, idx):
    textos = {
        "en": {"y": "Relative Euclidean distance to the prior", "sh": "Shannon (classical EP)",
               "lbl": "Tsallis, order q", "opt": "minimum at q = 2"},
        "es": {"y": "Distancia euclídea relativa al prior", "sh": "Shannon (EP clásico)",
               "lbl": "Tsallis, orden q", "opt": "mínimo en q = 2"},
    }
    mask = (q_vals >= 1.5) & (q_vals <= 2.5) & ~np.isnan(errors)
    for lang, tx in textos.items():
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.plot(q_vals[mask], errors[mask], "s-", color="#2E86AB", lw=1.8, ms=5, label=tx["lbl"])
        ax.axhline(err_sh, color="#E8475A", ls="--", lw=1.5, label=tx["sh"])
        ax.plot(q_opt, errors[idx], "D", color="#F5A623", ms=10,
                markeredgecolor="black", markeredgewidth=1.2, label=tx["opt"])
        ax.set_xlabel("q", fontsize=12)
        ax.set_ylabel(tx["y"], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if lang == "es":                      # coma decimal en los ejes
            fmt = lambda dec: FuncFormatter(lambda v, _: f"{v:.{dec}f}".replace(".", ","))
            ax.xaxis.set_major_formatter(fmt(1))
            ax.yaxis.set_major_formatter(fmt(5))
        fig.tight_layout()
        out = FIGS_DIR / f"optimal_q_zoom_{lang}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"   Figura guardada en: {out}")


# ═══════════════════════════════════════════════════════════════
# 2 a 5. Verificaciones sobre la base reducida
# ═══════════════════════════════════════════════════════════════

def verificaciones_base_reducida(out):
    X, p = load_mat_data(str(MAT_SHORT))
    p = p / p.sum()
    J = p.size
    A, b, Aeq, beq = restricciones(X)
    pS, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S")
    p2, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=2.0)
    res = {"J": int(J)}

    # 2. q = 2 contra un QP independiente
    try:
        import cvxpy as cp
        x = cp.Variable(J, nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [A @ x <= b, cp.sum(x) == 1])
        prob.solve(solver=cp.OSQP, eps_abs=1e-12, eps_rel=1e-12, max_iter=200_000)
        p_qp = np.maximum(x.value, 0.0)
        p_qp = p_qp / p_qp.sum()
        res["q2_vs_qp_max_abs"] = float(np.abs(p2 - p_qp).max())
        res["q2_vs_qp_max_rel_al_uniforme"] = float(np.abs(p2 - p_qp).max() * J)
        print(f"2) q=2 contra la proyección euclídea por QP (OSQP), J={J}: "
              f"máx |dif| = {res['q2_vs_qp_max_abs']:.1e}")
    except ImportError:
        res["q2_vs_qp_max_abs"] = None
        print("2) cvxpy no está instalado: se omite la comparación con el QP")

    # 3. Cada q maximiza el número efectivo de su propio orden
    filas = [{"q": 1.0, "N1_sobre_J": n_efectivo(pS, 1) / J, "N2_sobre_J": n_efectivo(pS, 2) / J}]
    for q in (1.25, 1.5, 2.0, 2.5, 3.0):
        pq, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
        filas.append({"q": q, "N1_sobre_J": n_efectivo(pq, 1) / J,
                      "N2_sobre_J": n_efectivo(pq, 2) / J})
    tab = pd.DataFrame(filas)
    res["numeros_efectivos"] = tab.to_dict(orient="records")
    res["q_max_N1"] = float(tab.loc[tab["N1_sobre_J"].idxmax(), "q"])
    res["q_max_N2"] = float(tab.loc[tab["N2_sobre_J"].idxmax(), "q"])
    print(f"3) Número efectivo de orden 1 máximo en q={res['q_max_N1']:.2f} (Shannon) "
          f"y de orden 2 máximo en q={res['q_max_N2']:.2f}")
    with pd.option_context("display.float_format", lambda v: f"{v:.8f}"):
        print(tab.to_string(index=False))

    # 4. Límite q → 1
    lim = {}
    for q in (1.1, 1.01, 1.001):
        pq, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
        lim[str(q)] = float(np.abs(pq - pS).max())
    res["limite_q1_max_abs"] = lim
    res["desplazamiento_shannon_max_abs"] = float(np.abs(pS - p).max())
    print(f"4) Límite q→1: máx |p̃_q − p̃_Shannon| = {lim} "
          f"(desplazamiento de Shannon respecto del prior: {res['desplazamiento_shannon_max_abs']:.1e})")

    # 5. Rényi contra Tsallis, resolviendo el primal de Rényi con SLSQP
    #    sobre una submuestra (el primal tiene una variable por escenario).
    m = 60
    Xs = X[:m]
    ps = np.full(m, 1.0 / m)
    As, bs, Aeqs, beqs = restricciones(Xs)
    ren = {}
    for q in (1.5, 2.0, 2.5):
        pT, _, _ = entropy_prog(ps, As, bs, Aeqs, beqs, entropy_family="T", q=q)

        def renyi(x, q=q):
            x = np.maximum(x, 1e-300)
            return np.log(np.sum(ps * (x / ps) ** q)) / (q - 1.0)

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
                {"type": "ineq", "fun": lambda x: bs - As @ x}]
        sol = minimize(renyi, ps.copy(), method="SLSQP", bounds=[(0.0, 1.0)] * m,
                       constraints=cons, options={"ftol": 1e-15, "maxiter": 2000})
        ren[str(q)] = float(np.abs(sol.x - pT).max())
    res["renyi_primal_vs_tsallis_max_abs"] = ren
    res["renyi_submuestra_J"] = m
    print(f"5) Rényi (primal, SLSQP, J={m}) contra Tsallis (dual): máx |dif| = {ren}")

    out["base_reducida"] = res


def redondear(x, cifras: int = 10):
    """Redondea los flotantes de una estructura anidada a `cifras` cifras
    significativas, para que el archivo de resultados no cambie de una corrida
    a otra por diferencias en el último dígito."""
    if isinstance(x, dict):
        return {k: redondear(v, cifras) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [redondear(v, cifras) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.{cifras}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {}
    q_vals, errors, err_sh, q_opt, idx = barrido_base_completa(out)
    figura_barrido(q_vals, errors, err_sh, q_opt, idx)
    verificaciones_base_reducida(out)
    with open(RESULTS_DIR / "verificaciones.json", "w", encoding="utf-8") as fh:
        json.dump(redondear(out), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"\nResultados guardados en: {RESULTS_DIR / 'verificaciones.json'}")


if __name__ == "__main__":
    main()
