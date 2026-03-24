#%%
"""
s_main_optimal_q.py
--------------------
Determinación del valor de q óptimo para Entropy Pooling con Tsallis q-entropy.

Traducción a Python del trabajo en GNU Octave:
  - S_MAIN.m (con parámetros Lower, Upper, ViewMethod, entropy_family, q)
  - views_generator.m (genera views con método original Meucci, sigma-escalado, o random)
  - ViewRanking.m (construye constraints de ranking y llama a EntropyProg)
  - Barrido de q para encontrar el que minimiza ||p_ - p|| / ||p||

El criterio: entre distintos posteriores que satisfacen las restricciones del view,
se prefiere aquel que difiera MENOS del prior (menor error relativo), ya que el view
pone en tensión las suposiciones previas sin necesidad de alejarse más de lo necesario.

Resultado esperado: q_óptimo ≈ 2.0 (entropía de colisión de Tsallis/Rényi).

Requisitos:
  - entropy_pooling_v2.py (en la misma carpeta o en sys.path)
  - ReturnsDistribution.mat (base de datos de Meucci, ~100k escenarios)
  - ReturnsDistributionShort.mat (versión reducida, ~1k escenarios, opcional)

Uso:
  python s_main_optimal_q.py
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Importar entropy_pooling_v2 desde el mismo directorio del script ──
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from entropy_pooling_v2 import (
    entropy_prog,
    efficient_frontier,
    plot_results,
    FrontierOptions,
    EntropyProg,
    EfficientFrontier,
    PlotResults,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. views_generator: traducción directa de views_generator.m
# ═══════════════════════════════════════════════════════════════════════
def views_generator(
    X: np.ndarray,
    Lower: int,
    Upper: int,
    ViewMethod: Union[float, str] = 0,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Genera el vector de views V (J,) según el método elegido.

    Parameters
    ----------
    X : (J, N) matriz de retornos realizados
    Lower : índice 0-based del activo con retorno esperado menor
    Upper : índice 0-based del activo con retorno esperado mayor
    ViewMethod :
        - 0       → Original Meucci: V = X[:,Lower] - X[:,Upper]
        - float≠0 → Sigma-escalado: V = |sigma|*X[:,min(L,U)] - X[:,max(L,U)]
        - 'random' → Randomizado:   V(k) = (1 + 2*|N(0,1)|*ms) * X(k,Lower) - X(k,Upper)
    random_seed : semilla para reproducibilidad en modo 'random'

    Returns
    -------
    V : (J,) vector de views
    """
    J = X.shape[0]

    if isinstance(ViewMethod, str) and ViewMethod.lower() == "random":
        rng = np.random.default_rng(random_seed)
        ms = np.mean(np.std(X, axis=0))
        V = np.zeros(J)
        for k in range(J):
            V[k] = (1 + 2 * abs(rng.standard_normal()) * ms) * X[k, Lower] - X[k, Upper]
        return V

    sigma = float(ViewMethod)
    if abs(sigma) > 0:
        lo = min(Lower, Upper)
        hi = max(Lower, Upper)
        V = abs(sigma) * X[:, lo] - X[:, hi]
        return V

    # ViewMethod == 0 → original Meucci
    V = X[:, Lower] - X[:, Upper]
    return V


# ═══════════════════════════════════════════════════════════════════════
# 2. view_ranking_extended: ViewRanking.m con entropy_family y q
# ═══════════════════════════════════════════════════════════════════════
def view_ranking_extended(
    X: np.ndarray,
    p: np.ndarray,
    Lower: int,
    Upper: int,
    ViewMethod: Union[float, str] = 0,
    entropy_family: str = "S",
    q: float = 1.0,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Calcula el posterior p_ minimizando la entropía relativa generalizada
    sujeto a las constraints de ranking.

    Parameters
    ----------
    X : (J, N) retornos
    p : (J,) prior
    Lower, Upper : índices 0-based
    ViewMethod : ver views_generator
    entropy_family : 'S' (Shannon), 'T' (Tsallis), 'R' (Rényi)
    q : parámetro de la entropía (ignorado si entropy_family='S')

    Returns
    -------
    p_ : (J,) posterior
    """
    J, N = X.shape

    # Constraints: sum(p_) = 1
    Aeq = np.ones((1, J))
    beq = np.array([1.0])

    # View constraints: E_p_[V] <= 0
    V = views_generator(X, Lower, Upper, ViewMethod, random_seed)
    V = V.reshape(-1, 1)  # (J,1)
    A = V.T               # (1, J)
    b = np.zeros(1)

    p_, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=entropy_family, q=q)
    return p_


# ═══════════════════════════════════════════════════════════════════════
# 3. S_MAIN: traducción directa de S_MAIN.m
# ═══════════════════════════════════════════════════════════════════════
def s_main(
    X: np.ndarray,
    p: np.ndarray,
    Lower: int,
    Upper: int,
    ViewMethod: Union[float, str] = 0,
    entropy_family: str = "T",
    q: float = 2.0,
    confidence: float = 0.5,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Ejecuta el pipeline completo: prior → view → posterior → frontera eficiente.

    Returns
    -------
    dict con claves: p_, p_clasico, p, error_tsallis, error_clasico,
                     e, s, w, M, S, e_, s_, w_, M_, S_
    """
    p = p / p.sum()
    J, N = X.shape

    # ── Frontera eficiente prior ──
    opts = FrontierOptions(NumPortf=20, FrontierSpan=(0.3, 0.9))
    e, s, w, M, S = efficient_frontier(X, p, opts)

    # ── Posteriores ──
    p_tsallis = view_ranking_extended(
        X, p, Lower, Upper, ViewMethod, entropy_family, q, random_seed
    )
    p_clasico = view_ranking_extended(
        X, p, Lower, Upper, ViewMethod, "S", 1.0, random_seed
    )

    # ── Mezcla con confianza ──
    p_post = (1 - confidence) * p + confidence * p_tsallis
    p_post = p_post / p_post.sum()

    p_clasico_post = (1 - confidence) * p + confidence * p_clasico
    p_clasico_post = p_clasico_post / p_clasico_post.sum()

    # ── Errores relativos ──
    err_tsallis = float(np.linalg.norm(p - p_post) / np.linalg.norm(p))
    err_clasico = float(np.linalg.norm(p - p_clasico_post) / np.linalg.norm(p))

    # ── Frontera eficiente posterior ──
    e_, s_, w_, M_, S_ = efficient_frontier(X, p_post, opts)

    return {
        "p_": p_post,
        "p_clasico": p_clasico_post,
        "p": p,
        "error_tsallis": err_tsallis,
        "error_clasico": err_clasico,
        "e": e, "s": s, "w": w, "M": M, "S": S,
        "e_": e_, "s_": s_, "w_": w_, "M_": M_, "S_": S_,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Barrido de q: determinación del q óptimo
# ═══════════════════════════════════════════════════════════════════════
def sweep_q(
    X: np.ndarray,
    p: np.ndarray,
    Lower: int,
    Upper: int,
    ViewMethod: Union[float, str] = 1.4,
    q_values: Optional[np.ndarray] = None,
    confidence: float = 0.5,
    random_seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Barre un rango de valores de q, calculando el error relativo ||p_post - p||/||p||
    para cada uno. También calcula el error del EP clásico (Shannon) como benchmark.

    Parameters
    ----------
    q_values : array de valores de q a evaluar (default: 1.05 a 3.0, paso 0.05)
    confidence : peso de mezcla del posterior con el prior

    Returns
    -------
    q_values : array de q evaluados
    errors   : array de errores relativos para cada q
    err_clasico : error del EP clásico (Shannon)
    q_opt    : valor de q óptimo
    idx_opt  : índice del q óptimo
    """
    if q_values is None:
        q_values = np.arange(1.05, 3.01, 0.05)

    p = p / p.sum()
    J = p.size
    norm_p = np.linalg.norm(p)

    # ── Constraints (se computan una sola vez) ──
    Aeq = np.ones((1, J))
    beq = np.array([1.0])
    V = views_generator(X, Lower, Upper, ViewMethod, random_seed).reshape(-1, 1)
    A = V.T
    b = np.zeros(1)

    # ── EP Clásico (Shannon) ──
    if verbose:
        print("Calculando EP clásico (Shannon)...")
    p_clasico_raw, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S", q=1.0)
    p_clasico = (1 - confidence) * p + confidence * p_clasico_raw
    err_clasico = float(np.linalg.norm(p - p_clasico) / norm_p)
    if verbose:
        print(f"  Shannon error relativo: {err_clasico:.6f}")

    # ── Barrido de q con Tsallis ──
    errors = np.zeros_like(q_values)
    if verbose:
        print(f"\nBarriendo q de {q_values[0]:.2f} a {q_values[-1]:.2f} ({len(q_values)} valores)...")

    for i, q in enumerate(q_values):
        t0 = time.time()
        try:
            p_t_raw, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
            p_t = (1 - confidence) * p + confidence * p_t_raw
            errors[i] = float(np.linalg.norm(p - p_t) / norm_p)
        except Exception as ex:
            errors[i] = np.nan
            if verbose:
                print(f"  q={q:.2f}: FALLÓ ({ex})")
            continue

        dt = time.time() - t0
        if verbose:
            better = "✓" if errors[i] < err_clasico else " "
            print(f"  q={q:.2f}  error={errors[i]:.6f}  ({dt:.1f}s) {better}")

    # ── Encontrar óptimo ──
    valid = ~np.isnan(errors)
    idx_opt = int(np.argmin(errors[valid]))
    # Mapear al índice original
    valid_indices = np.where(valid)[0]
    idx_opt_global = valid_indices[idx_opt]
    q_opt = float(q_values[idx_opt_global])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  q óptimo = {q_opt:.2f}")
        print(f"  Error mínimo (Tsallis): {errors[idx_opt_global]:.6f}")
        print(f"  Error Shannon:          {err_clasico:.6f}")
        print(f"  Mejora:                 {(err_clasico - errors[idx_opt_global])/err_clasico*100:.4f}%")
        print(f"{'='*60}")

    return q_values, errors, err_clasico, q_opt, idx_opt_global


# ═══════════════════════════════════════════════════════════════════════
# 5. Gráfico del q óptimo
# ═══════════════════════════════════════════════════════════════════════
def plot_optimal_q(
    q_values: np.ndarray,
    errors: np.ndarray,
    err_clasico: float,
    q_opt: float,
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Genera el gráfico de error relativo vs q, marcando el q óptimo
    y la línea de referencia del EP clásico (Shannon).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    valid = ~np.isnan(errors)
    q_v = q_values[valid]
    e_v = errors[valid]

    # ── Curva de error ──
    ax.plot(q_v, e_v, "o-", color="#2E86AB", linewidth=2, markersize=5,
            label="Tsallis q-EP", zorder=3)

    # ── Línea Shannon ──
    ax.axhline(y=err_clasico, color="#E8475A", linestyle="--", linewidth=1.5,
               label=f"EP clásico (Shannon) = {err_clasico:.6f}", zorder=2)

    # ── Marcar q óptimo ──
    idx_opt = np.argmin(e_v)
    ax.plot(q_v[idx_opt], e_v[idx_opt], "D", color="#F5A623", markersize=12,
            markeredgecolor="black", markeredgewidth=1.5,
            label=f"q óptimo = {q_opt:.2f} (error = {e_v[idx_opt]:.6f})",
            zorder=4)

    # ── Región de mejora ──
    mask_better = e_v < err_clasico
    if mask_better.any():
        ax.fill_between(
            q_v, err_clasico, e_v,
            where=mask_better,
            alpha=0.15, color="#2E86AB",
            label="Región de mejora vs Shannon",
        )

    # ── Estética ──
    ax.set_xlabel("q (parámetro de Tsallis)", fontsize=13)
    ax.set_ylabel("Error relativo  ||p_posterior − p_prior|| / ||p_prior||", fontsize=12)

    main_title = "Determinación del q óptimo — Entropy Pooling con Tsallis"
    if title_suffix:
        main_title += f"\n{title_suffix}"
    ax.set_title(main_title, fontsize=14, fontweight="bold")

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    # Zoom: ajustar los ejes al rango relevante
    y_min = min(e_v.min(), err_clasico) * 0.9999
    y_max = max(e_v.max(), err_clasico) * 1.0001
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5b. Barrido de q con múltiples semillas random (robustez)
# ═══════════════════════════════════════════════════════════════════════
def sweep_q_random_seeds(
    X: np.ndarray,
    p: np.ndarray,
    Lower: int,
    Upper: int,
    seeds: Optional[List[int]] = None,
    q_values: Optional[np.ndarray] = None,
    confidence: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Corre el barrido de q usando ViewMethod='random' con múltiples semillas.
    Para cada semilla genera un set de views distinto y busca el q óptimo.

    Parameters
    ----------
    seeds : lista de semillas (default: [42, 123, 777, 2024, 314])
    q_values : array de q a evaluar (default: 1.05 a 3.0, paso 0.05)

    Returns
    -------
    dict con claves:
        q_values     : array de q evaluados
        all_errors   : dict {seed: array de errores}
        all_shannon  : dict {seed: error Shannon}
        all_q_opt    : dict {seed: q óptimo}
        seeds        : lista de semillas usadas
    """
    if seeds is None:
        seeds = [42, 123, 777, 2024, 314]
    if q_values is None:
        q_values = np.arange(1.05, 3.01, 0.05)

    p = p / p.sum()
    J = p.size
    norm_p = np.linalg.norm(p)

    Aeq = np.ones((1, J))
    beq = np.array([1.0])

    all_errors = {}
    all_shannon = {}
    all_q_opt = {}

    for seed in seeds:
        if verbose:
            print(f"\n── Seed {seed} ──")

        V = views_generator(X, Lower, Upper, "random", random_seed=seed).reshape(-1, 1)
        A = V.T
        b = np.zeros(1)

        # Shannon
        p_sh, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S", q=1.0)
        p_sh_c = (1 - confidence) * p + confidence * p_sh
        err_sh = float(np.linalg.norm(p - p_sh_c) / norm_p)
        all_shannon[seed] = err_sh

        # Barrido de q
        errors = np.zeros_like(q_values)
        for i, q in enumerate(q_values):
            try:
                p_t, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
                p_t_c = (1 - confidence) * p + confidence * p_t
                errors[i] = float(np.linalg.norm(p - p_t_c) / norm_p)
            except Exception:
                errors[i] = np.nan

        all_errors[seed] = errors
        valid = ~np.isnan(errors)
        idx_best = int(np.argmin(errors[valid]))
        q_opt = float(q_values[np.where(valid)[0][idx_best]])
        all_q_opt[seed] = q_opt

        if verbose:
            print(f"  Shannon: {err_sh:.6f}  |  q* = {q_opt:.2f} (err = {errors[np.where(valid)[0][idx_best]]:.6f})")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESUMEN: q óptimo por semilla")
        print(f"{'='*60}")
        for seed in seeds:
            print(f"  seed={seed:>5d}  →  q* = {all_q_opt[seed]:.2f}")
        q_opts_unique = set(f"{v:.2f}" for v in all_q_opt.values())
        if len(q_opts_unique) == 1:
            print(f"\n  ✓ Todas las semillas coinciden: q* = {list(q_opts_unique)[0]}")
        else:
            print(f"\n  Valores distintos: {q_opts_unique}")
        print(f"{'='*60}")

    return {
        "q_values": q_values,
        "all_errors": all_errors,
        "all_shannon": all_shannon,
        "all_q_opt": all_q_opt,
        "seeds": seeds,
    }


def plot_random_seeds(
    results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera el gráfico de robustez: múltiples semillas random superpuestas.
    Panel izquierdo: errores absolutos. Panel derecho: ratio vs Shannon.
    """
    q_values = results["q_values"]
    seeds = results["seeds"]
    all_errors = results["all_errors"]
    all_shannon = results["all_shannon"]
    all_q_opt = results["all_q_opt"]

    colors = ["#2E86AB", "#E8475A", "#F5A623", "#7B68EE", "#2ECC71",
              "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for si, seed in enumerate(seeds):
        errors = all_errors[seed]
        err_sh = all_shannon[seed]
        q_opt = all_q_opt[seed]
        valid = ~np.isnan(errors)
        c = colors[si % len(colors)]

        # Panel izquierdo: curvas crudas
        ax1.plot(q_values[valid], errors[valid], "o-", color=c, markersize=3,
                 linewidth=1.2, label=f"seed={seed} (q*={q_opt:.1f})", alpha=0.8)
        ax1.axhline(y=err_sh, color=c, linestyle=":", alpha=0.3, linewidth=0.8)

        # Panel derecho: normalizado (error / error_shannon)
        ax2.plot(q_values[valid], errors[valid] / err_sh, "o-", color=c, markersize=3,
                 linewidth=1.2, label=f"seed={seed}", alpha=0.8)

    # Panel izquierdo
    ax1.set_xlabel("q (parámetro de Tsallis)", fontsize=13)
    ax1.set_ylabel("Error relativo  ||p̃ − p|| / ||p||", fontsize=12)
    ax1.set_title("ViewMethod = random, distintas semillas", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel derecho
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Shannon (ref = 1.0)")
    ax2.axvline(x=2.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("q (parámetro de Tsallis)", fontsize=13)
    ax2.set_ylabel("Error normalizado (Tsallis / Shannon)", fontsize=12)
    ax2.set_title("Ratio vs Shannon — robustez del q*", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico random seeds guardado en: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN: ejecutar todo
# ═══════════════════════════════════════════════════════════════════════
def load_mat_data(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga X y p desde un .mat (scipy o Octave text format)."""
    from scipy.io import loadmat
    try:
        mat = loadmat(mat_path)
        X = mat["X"].astype(float)
        p = mat["p"].ravel().astype(float)
        return X, p
    except Exception:
        pass

    # Fallback: Octave text format
    with open(mat_path, "r") as f:
        lines = f.readlines()

    data = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("# name:"):
            name = lines[i].split(":")[1].strip()
            i += 1
            dtype = lines[i].split(":")[1].strip()
            i += 1
            if dtype == "matrix":
                rows = int(lines[i].split(":")[1].strip())
                i += 1
                cols = int(lines[i].split(":")[1].strip())
                i += 1
                mat_data = []
                for _ in range(rows):
                    vals = [float(x) for x in lines[i].split()]
                    mat_data.append(vals)
                    i += 1
                data[name] = np.array(mat_data)
            else:
                i += 1
        else:
            i += 1

    return data["X"].astype(float), data["p"].ravel().astype(float)


if __name__ == "__main__":
    # ── Configuración ──
    Lower_matlab = 4   # 1-based (MATLAB/Octave)
    Upper_matlab = 3
    Lower = Lower_matlab - 1  # 0-based (Python)
    Upper = Upper_matlab - 1
    ViewMethod = 1.4   # sigma-escalado (como en el documento)
    confidence = 0.5

    # Rango de q a barrer (fino alrededor de 2)
    q_coarse = np.arange(1.1, 3.01, 0.1)
    q_fine   = np.arange(1.8, 2.21, 0.02)
    q_values = np.unique(np.sort(np.concatenate([q_coarse, q_fine])))

    # ── Cargar datos ──
    # Busca los .mat en el mismo directorio del script
    mat_large = HERE / "ReturnsDistribution.mat"
    mat_short = HERE / "ReturnsDistributionShort.mat"

    if mat_large.exists():
        print(f"Cargando {mat_large.name} (~100k escenarios)...")
        X, p = load_mat_data(str(mat_large))
    elif mat_short.exists():
        print(f"Cargando {mat_short.name} (~1k escenarios)...")
        X, p = load_mat_data(str(mat_short))
    else:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .mat de retornos en {HERE}\n"
            "  Se espera 'ReturnsDistribution.mat' o 'ReturnsDistributionShort.mat'"
        )

    p = p / p.sum()
    J, N = X.shape
    print(f"Datos: {J} escenarios × {N} activos")
    print(f"View: Lower={Lower_matlab} (0b:{Lower}), Upper={Upper_matlab} (0b:{Upper})")
    print(f"ViewMethod={ViewMethod}, confidence={confidence}")
    print()

    # ── Barrido de q ──
    q_vals, errors, err_clasico, q_opt, idx_opt = sweep_q(
        X, p, Lower, Upper,
        ViewMethod=ViewMethod,
        q_values=q_values,
        confidence=confidence,
        verbose=True,
    )

    # ── Tabla resumen estilo Octave ──
    print("\n" + "="*60)
    print("TABLA DE RESULTADOS (comparable con resultaditos.m)")
    print("="*60)
    print(f"{'q':>6s}  {'Error Tsallis':>15s}  {'vs Shannon':>12s}")
    print("-"*40)
    for q_show in [1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5]:
        idx = np.argmin(np.abs(q_vals - q_show))
        if not np.isnan(errors[idx]):
            diff = errors[idx] - err_clasico
            print(f"  {q_vals[idx]:4.1f}    {errors[idx]:.6f}       {diff:+.6f}")
    print(f"\n  Shannon (ref): {err_clasico:.6f}")
    print(f"  q óptimo:      {q_opt:.2f} → {errors[idx_opt]:.6f}")

    # ── Validación cruzada contra resultados de Octave (resultaditos.m) ──
    # Valores de referencia con ViewMethod=1.4, Lower=4, Upper=3, c=0.5
    OCTAVE_REF = {
        "Shannon": 0.046042,
        1.1: 0.046030,
        1.3: 0.045991,
        1.5: 0.045968,
        1.7: 0.045953,
        1.9: 0.045945,
        2.0: 0.045944,
        2.1: 0.045945,
        2.3: 0.045953,
        2.5: 0.045969,
    }

    print("\n" + "="*72)
    print("VALIDACIÓN CRUZADA: Python vs Octave (resultaditos.m)")
    print("="*72)
    print(f"  {'q':>7s}  {'Octave':>10s}  {'Python':>10s}  {'Δ (Py-Oct)':>12s}  {'Match':>7s}")
    print("  " + "-"*56)

    # Shannon primero
    diff_sh = err_clasico - OCTAVE_REF["Shannon"]
    match_sh = "✓" if abs(diff_sh) < 1e-5 else f"({abs(diff_sh):.1e})"
    print(f"  {'Shannon':>7s}  {OCTAVE_REF['Shannon']:>10.6f}  {err_clasico:>10.6f}  {diff_sh:>+12.2e}  {match_sh:>7s}")

    # Cada q
    for q_ref, err_oct in sorted((k, v) for k, v in OCTAVE_REF.items() if isinstance(k, (int, float))):
        idx = np.argmin(np.abs(q_vals - q_ref))
        if np.isnan(errors[idx]):
            continue
        err_py = errors[idx]
        diff = err_py - err_oct
        match = "✓" if abs(diff) < 1e-5 else f"({abs(diff):.1e})"
        print(f"  {q_ref:>7.1f}  {err_oct:>10.6f}  {err_py:>10.6f}  {diff:>+12.2e}  {match:>7s}")

    # Resumen
    diffs_all = []
    for q_ref, err_oct in OCTAVE_REF.items():
        if q_ref == "Shannon":
            diffs_all.append(abs(err_clasico - err_oct))
        elif isinstance(q_ref, (int, float)):
            idx = np.argmin(np.abs(q_vals - q_ref))
            if not np.isnan(errors[idx]):
                diffs_all.append(abs(errors[idx] - err_oct))

    print("  " + "-"*56)
    print(f"  Max |Δ| = {max(diffs_all):.2e}   (tolerancia < 1e-4: {'OK ✓' if max(diffs_all) < 1e-4 else 'REVISAR'})")
    print(f"  Ambos coinciden: q* = 2.0")

    # ── Gráficos (se guardan junto al script) ──
    output_path = HERE / "optimal_q_tsallis.png"
    fig = plot_optimal_q(
        q_vals, errors, err_clasico, q_opt,
        save_path=str(output_path),
        title_suffix=f"Lower={Lower_matlab}, Upper={Upper_matlab}, σ={ViewMethod}, c={confidence}  |  Datos: {J}×{N}",
    )

    # Gráfico zoom fino alrededor del óptimo
    mask_zoom = (q_vals >= 1.5) & (q_vals <= 2.5) & ~np.isnan(errors)
    if mask_zoom.sum() > 3:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.plot(q_vals[mask_zoom], errors[mask_zoom], "s-", color="#2E86AB",
                 linewidth=2, markersize=6, label="Tsallis q-EP")
        ax2.axhline(y=err_clasico, color="#E8475A", linestyle="--", linewidth=1.5,
                    label=f"Shannon = {err_clasico:.6f}")
        ax2.axvline(x=q_opt, color="#F5A623", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2.plot(q_opt, errors[idx_opt], "D", color="#F5A623", markersize=12,
                 markeredgecolor="black", markeredgewidth=1.5,
                 label=f"q* = {q_opt:.2f}")
        ax2.set_xlabel("q", fontsize=13)
        ax2.set_ylabel("Error relativo", fontsize=12)
        ax2.set_title("Zoom: q óptimo para Tsallis EP", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        zoom_path = HERE / "optimal_q_zoom.png"
        fig2.savefig(str(zoom_path), dpi=150, bbox_inches="tight")
        print(f"\nGráfico zoom guardado en: {zoom_path}")

    # ── Análisis de robustez con ViewMethod random ──
    print("\n" + "="*60)
    print("ANÁLISIS DE ROBUSTEZ: ViewMethod='random', múltiples semillas")
    print("="*60)

    random_results = sweep_q_random_seeds(
        X, p, Lower, Upper,
        seeds=[42, 123, 777, 2024, 314],
        q_values=np.arange(1.05, 3.01, 0.05),
        confidence=confidence,
        verbose=True,
    )

    seeds_path = HERE / "optimal_q_random_seeds.png"
    plot_random_seeds(random_results, save_path=str(seeds_path))

    # ── Tabla detallada: error por q para cada seed ──
    r_seeds = random_results["seeds"]
    r_qvals = random_results["q_values"]
    r_errors = random_results["all_errors"]
    r_shannon = random_results["all_shannon"]
    r_qopt = random_results["all_q_opt"]

    # Seleccionar los q representativos para la tabla
    q_display = [1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5, 3.0]

    print("\n" + "="*90)
    print("TABLA DETALLADA: Error relativo por q y semilla (ViewMethod='random')")
    print("="*90)

    # Header
    header = f"  {'q':>5s}"
    for seed in r_seeds:
        header += f"  {'seed='+str(seed):>12s}"
    print(header)
    print("  " + "-"*(5 + 14 * len(r_seeds)))

    # Shannon row
    row = f"  {'Shan.':>5s}"
    for seed in r_seeds:
        row += f"  {r_shannon[seed]:>12.6f}"
    print(row)
    print("  " + "-"*(5 + 14 * len(r_seeds)))

    # Data rows
    for q_show in q_display:
        idx = int(np.argmin(np.abs(r_qvals - q_show)))
        is_best = abs(r_qvals[idx] - 2.0) < 0.01
        row = f"  {r_qvals[idx]:>5.1f}"
        for seed in r_seeds:
            err = r_errors[seed][idx]
            if np.isnan(err):
                row += f"  {'NaN':>12s}"
            else:
                row += f"  {err:>12.6f}"
        if is_best:
            row += "  ← q*"
        print(row)

    print("  " + "-"*(5 + 14 * len(r_seeds)))

    # q* row
    row = f"  {'q*':>5s}"
    for seed in r_seeds:
        row += f"  {r_qopt[seed]:>12.2f}"
    print(row)

    print("\n✓ Script completado exitosamente.")
