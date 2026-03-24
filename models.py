"""
models.py
=========
Implementación de los tres modelos de asignación de cartera con views:

  1. Black-Litterman (BL)
     - Actualización bayesiana paramétrica (normal-normal conjugada)
     - Ref: Black & Litterman (1992), He & Litterman (2002)

  2. Entropy Pooling (EP)
     - Minimización de la divergencia de Kullback-Leibler (Shannon)
     - Ref: Meucci (2008)

  3. q-Tsallis Entropy Pooling (q-Tsallis-EP)
     - Minimización de la q-divergencia de Tsallis con q parametrizable
     - Ref: Tsallis (1988), Corvalán Salguero (2026)

Cada modelo recibe:
  - Datos de mercado (retornos, covarianzas, weights del benchmark)
  - Views traducidos (via views_config.py)

Y devuelve:
  - Cartera óptima posterior
  - Distribución posterior (retornos esperados, covarianzas)
  - Frontera eficiente
  - Medidas de riesgo

Uso:
    from models import run_black_litterman, run_entropy_pooling, run_q_tsallis_ep

Requiere: entropy_pooling_v2.py en el mismo directorio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from views_config import BLViews, EPViews


# ═══════════════════════════════════════════════════════════════════════
# Resultado unificado
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """
    Resultado de cualquiera de los tres modelos.

    Attributes
    ----------
    model_name : str
        Nombre del modelo ("BL", "EP-Shannon", "q-Tsallis-EP").
    w_optimal : (N,) np.ndarray
        Pesos óptimos del portafolio posterior.
    mu_posterior : (N,) np.ndarray
        Retornos esperados posteriores.
    Sigma_posterior : (N, N) np.ndarray
        Covarianzas posteriores (en BL cambia, en EP se recalcula con p̃).
    frontier_e : (NumPortf,) np.ndarray
        Retornos esperados de la frontera eficiente.
    frontier_s : (NumPortf,) np.ndarray
        Volatilidades de la frontera eficiente.
    frontier_w : (NumPortf, N) np.ndarray
        Pesos de las carteras en la frontera eficiente.
    p_posterior : (J,) np.ndarray or None
        Distribución posterior de probabilidades (solo EP/q-EP).
    risk_metrics : dict
        Medidas de riesgo del portafolio óptimo.
    params : dict
        Parámetros del modelo usados.
    """
    model_name: str
    w_optimal: np.ndarray
    mu_posterior: np.ndarray
    Sigma_posterior: np.ndarray
    frontier_e: np.ndarray
    frontier_s: np.ndarray
    frontier_w: np.ndarray
    p_posterior: Optional[np.ndarray]
    risk_metrics: Dict[str, float]
    params: Dict[str, object]


# ═══════════════════════════════════════════════════════════════════════
# Utilidades comunes
# ═══════════════════════════════════════════════════════════════════════

def _compute_risk_metrics(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    X: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    annualization_factor: float = 252.0,
) -> Dict[str, float]:
    """
    Calcula medidas de riesgo para un portafolio dado.

    Parameters
    ----------
    w : (N,) pesos del portafolio
    mu : (N,) retornos esperados (diarios)
    Sigma : (N,N) covarianzas (diarias)
    X : (J,N) escenarios (opcional, para VaR/CVaR empírico)
    p : (J,) probabilidades (opcional)
    annualization_factor : float
        Factor de anualización (252 para datos diarios).

    Returns
    -------
    dict con claves:
        - expected_return_daily, expected_return_annual
        - volatility_daily, volatility_annual
        - sharpe_ratio (asumiendo rf=0 para simplificar)
        - VaR_95, VaR_99 (si X y p disponibles)
        - CVaR_95, CVaR_99 (si X y p disponibles)
    """
    ret_daily = float(w @ mu)
    vol_daily = float(np.sqrt(w @ Sigma @ w))
    sqrt_ann = np.sqrt(annualization_factor)

    metrics = {
        "expected_return_daily": ret_daily,
        "expected_return_annual": ret_daily * annualization_factor,
        "volatility_daily": vol_daily,
        "volatility_annual": vol_daily * sqrt_ann,
        "sharpe_ratio": (ret_daily / vol_daily * sqrt_ann) if vol_daily > 0 else 0.0,
    }

    # VaR y CVaR empíricos si hay escenarios
    if X is not None and p is not None:
        portfolio_returns = X @ w  # (J,)
        # Ordenar por retorno
        sorted_idx = np.argsort(portfolio_returns)
        sorted_returns = portfolio_returns[sorted_idx]
        sorted_probs = p[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)

        for alpha, label in [(0.05, "95"), (0.01, "99")]:
            # VaR: el retorno tal que P(R < VaR) = alpha
            var_idx = np.searchsorted(cum_probs, alpha)
            var_idx = min(var_idx, len(sorted_returns) - 1)
            var_value = sorted_returns[var_idx]
            metrics[f"VaR_{label}"] = float(-var_value)  # convención: VaR positivo

            # CVaR: E[R | R < VaR]
            tail_mask = cum_probs <= alpha
            if tail_mask.any():
                tail_probs = sorted_probs[tail_mask]
                tail_returns = sorted_returns[tail_mask]
                cvar = float(np.sum(tail_probs * tail_returns) / np.sum(tail_probs))
                metrics[f"CVaR_{label}"] = float(-cvar)
            else:
                metrics[f"CVaR_{label}"] = float(-var_value)

    return metrics


def _mean_variance_optimal(
    mu: np.ndarray,
    Sigma: np.ndarray,
    delta: float = 2.5,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> np.ndarray:
    """
    Resuelve el problema de media-varianza:
        max_w  w^T μ - (δ/2) w^T Σ w
        s.t.   1^T w = 1, 0 ≤ w_i ≤ max_weight (si long_only)

    Parameters
    ----------
    mu : (N,) retornos esperados
    Sigma : (N,N) covarianzas
    delta : float
        Coeficiente de aversión al riesgo.
    long_only : bool
        Si True, impone w >= 0.
    max_weight : float
        Peso máximo por activo (ej: 0.30 para 30%). Default 1.0 (sin límite).

    Returns
    -------
    w : (N,) pesos óptimos
    """
    N = len(mu)
    bounds = [(0.0, max_weight)] * N if long_only else [(-max_weight, max_weight)] * N

    def objective(w):
        return -(w @ mu - (delta / 2) * w @ Sigma @ w)

    def grad(w):
        return -(mu - delta * Sigma @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    w0 = np.full(N, 1.0 / N)
    res = minimize(objective, w0, jac=grad, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"maxiter": 10000, "ftol": 1e-12})

    if not res.success:
        import warnings
        warnings.warn(f"Optimización media-varianza no convergió: {res.message}")

    w = np.asarray(res.x)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    return w


def _efficient_frontier_mv(
    mu: np.ndarray,
    Sigma: np.ndarray,
    num_portf: int = 20,
    frontier_span: Tuple[float, float] = (0.3, 0.9),
    long_only: bool = True,
    max_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la frontera eficiente de media-varianza.

    Parameters
    ----------
    max_weight : float
        Peso máximo por activo (ej: 0.30 para 30%).

    Returns
    -------
    e : (num_portf,) retornos esperados
    s : (num_portf,) volatilidades
    w : (num_portf, N) pesos
    """
    N = len(mu)
    bounds = [(0.0, max_weight)] * N if long_only else [(-max_weight, max_weight)] * N

    # Min vol
    def min_vol_obj(w):
        return float(w @ Sigma @ w)

    cons_eq = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.full(N, 1.0 / N)
    res_min = minimize(min_vol_obj, w0, bounds=bounds, constraints=cons_eq,
                       method="SLSQP", options={"ftol": 1e-12})
    min_ret = float(res_min.x @ mu)

    # Max ret
    from scipy.optimize import linprog
    c = -mu
    A_eq = np.ones((1, N))
    b_eq = np.array([1.0])
    lp = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    max_ret = float(lp.x @ mu)

    # Grid
    grid = np.linspace(frontier_span[0], frontier_span[1], num_portf)
    targets = min_ret + grid * (max_ret - min_ret)

    e_list, s_list, w_list = [], [], []
    w_prev = res_min.x.copy()

    for t in targets:
        target_val = float(t)
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tv=target_val: w @ mu - tv},
        ]
        res = minimize(min_vol_obj, w_prev, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"ftol": 1e-12, "maxiter": 10000})
        w_t = np.asarray(res.x)
        w_prev = w_t
        w_list.append(w_t)
        s_list.append(float(np.sqrt(w_t @ Sigma @ w_t)))
        e_list.append(float(w_t @ mu))

    return np.array(e_list), np.array(s_list), np.vstack(w_list)


# ═══════════════════════════════════════════════════════════════════════
# Modelo 1: Black-Litterman
# ═══════════════════════════════════════════════════════════════════════

def run_black_litterman(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    bl_views: BLViews,
    tickers: List[str],
    delta: float = 2.5,
    tau: float = 0.05,
    X: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    num_portf: int = 20,
    max_weight: float = 1.0,
) -> ModelResult:
    """
    Ejecuta el modelo de Black-Litterman.

    Ecuaciones centrales (He & Litterman, 2002; Idzorek, 2007):

        Π = δ Σ w_mkt                                    (reverse optimization)
        μ_BL = (τΣ⁻¹ + P^T Ω⁻¹ P)⁻¹ (τΣ⁻¹ Π + P^T Ω⁻¹ Q)   (posterior mean)
        Σ_BL = (τΣ⁻¹ + P^T Ω⁻¹ P)⁻¹                            (posterior cov)
        w_BL = (1/δ) Σ⁻¹ μ_BL                            (optimal weights)

    Parameters
    ----------
    Sigma : (N,N) covarianzas
    w_mkt : (N,) pesos del benchmark de mercado
    bl_views : BLViews
        Views en formato BL (P, Q, Ω).
    tickers : list[str]
    delta : float
        Coeficiente de aversión al riesgo.
    tau : float
        Incertidumbre del prior.
    X, p : opcionales, para calcular VaR/CVaR.
    num_portf : int
        Número de portafolios en la frontera eficiente.

    Returns
    -------
    ModelResult
    """
    N = len(tickers)
    P, Q, Omega = bl_views.P, bl_views.Q, bl_views.Omega

    # Reverse optimization: retornos implícitos de equilibrio
    Pi = delta * Sigma @ w_mkt  # (N,)

    # Posterior BL
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)  # (N,N)
    Omega_inv = np.linalg.inv(Omega)            # (K,K)

    # Σ_BL = (τΣ⁻¹ + P^T Ω⁻¹ P)⁻¹
    Sigma_BL = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)

    # μ_BL = Σ_BL (τΣ⁻¹ Π + P^T Ω⁻¹ Q)
    mu_BL = Sigma_BL @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

    # Portafolio óptimo
    w_BL = _mean_variance_optimal(mu_BL, Sigma, delta=delta, long_only=True, max_weight=max_weight)

    # Frontera eficiente con la distribución posterior
    frontier_e, frontier_s, frontier_w = _efficient_frontier_mv(
        mu_BL, Sigma + Sigma_BL, num_portf=num_portf, max_weight=max_weight
    )

    # Medidas de riesgo
    risk = _compute_risk_metrics(w_BL, mu_BL, Sigma, X=X, p=p)

    return ModelResult(
        model_name="Black-Litterman",
        w_optimal=w_BL,
        mu_posterior=mu_BL,
        Sigma_posterior=Sigma + Sigma_BL,
        frontier_e=frontier_e,
        frontier_s=frontier_s,
        frontier_w=frontier_w,
        p_posterior=None,
        risk_metrics=risk,
        params={"delta": delta, "tau": tau, "Pi": Pi},
    )


# ═══════════════════════════════════════════════════════════════════════
# Modelo 2: Entropy Pooling (Shannon, q=1)
# ═══════════════════════════════════════════════════════════════════════

def run_entropy_pooling(
    X: np.ndarray,
    p: np.ndarray,
    ep_views: EPViews,
    tickers: List[str],
    w_mkt: np.ndarray,
    delta: float = 2.5,
    confidence: float = 1.0,
    num_portf: int = 20,
    max_weight: float = 1.0,
) -> ModelResult:
    """
    Ejecuta Entropy Pooling clásico (Shannon/Kullback-Leibler).

    Resuelve:
        p̃ = argmin_{x: Ax≤b, Aeq·x=beq}  Σ x_j [ln(x_j) - ln(p_j)]

    Ref: Meucci (2008), "Fully Flexible Views: Theory and Practice".

    Parameters
    ----------
    X : (J, N) realizaciones de retornos
    p : (J,) prior
    ep_views : EPViews
        Views en formato EP.
    tickers : list[str]
    w_mkt : (N,) pesos del benchmark
    delta : float
        Aversión al riesgo para la optimización media-varianza.
    confidence : float
        Confianza global (se mezcla: p_post = (1-c)*p + c*p̃).
        Nota: la confianza individual ya está incorporada en ep_views.
    num_portf : int

    Returns
    -------
    ModelResult
    """
    from entropy_pooling_v2 import entropy_prog, efficient_frontier, FrontierOptions

    J, N = X.shape
    p = p / p.sum()

    # Resolver EP
    p_view, _, _ = entropy_prog(
        p, ep_views.A, ep_views.b, ep_views.Aeq, ep_views.beq,
        entropy_family="S", q=1.0
    )

    # Mezcla con confianza global
    p_post = (1 - confidence) * p + confidence * p_view
    p_post = np.maximum(p_post, 0.0)
    p_post = p_post / p_post.sum()

    # Momentos posteriores
    mu_post = X.T @ p_post
    Scnd = X.T @ (X * p_post[:, None])
    Scnd = 0.5 * (Scnd + Scnd.T)
    Sigma_post = Scnd - np.outer(mu_post, mu_post)

    # Portafolio óptimo
    w_opt = _mean_variance_optimal(mu_post, Sigma_post, delta=delta, long_only=True, max_weight=max_weight)

    # Frontera eficiente usando EP (distribución completa)
    opts = FrontierOptions(NumPortf=num_portf, FrontierSpan=(0.3, 0.9))
    frontier_e, frontier_s, frontier_w, _, _ = efficient_frontier(X, p_post, opts)

    # Medidas de riesgo
    risk = _compute_risk_metrics(w_opt, mu_post, Sigma_post, X=X, p=p_post)

    return ModelResult(
        model_name="EP-Shannon",
        w_optimal=w_opt,
        mu_posterior=mu_post,
        Sigma_posterior=Sigma_post,
        frontier_e=frontier_e,
        frontier_s=frontier_s,
        frontier_w=frontier_w,
        p_posterior=p_post,
        risk_metrics=risk,
        params={"entropy_family": "S", "q": 1.0, "confidence": confidence},
    )


# ═══════════════════════════════════════════════════════════════════════
# Modelo 3: q-Tsallis Entropy Pooling
# ═══════════════════════════════════════════════════════════════════════

def run_q_tsallis_ep(
    X: np.ndarray,
    p: np.ndarray,
    ep_views: EPViews,
    tickers: List[str],
    w_mkt: np.ndarray,
    delta: float = 2.5,
    q: float = 2.0,
    confidence: float = 1.0,
    num_portf: int = 20,
    max_weight: float = 1.0,
) -> ModelResult:
    """
    Ejecuta q-Tsallis Entropy Pooling.

    Resuelve:
        p̃_q = argmin_{x: Ax≤b, Aeq·x=beq}  (1/(q-1)) * [Σ p_j (x_j/p_j)^q - 1]

    Para q=2 corresponde a la entropía de colisión (Rényi collision entropy),
    que se demostró es el valor óptimo para la determinación del posterior
    con mínima distancia al prior.

    Ref: Tsallis (1988), Corvalán Salguero (2026).

    Parameters
    ----------
    X : (J, N) realizaciones de retornos
    p : (J,) prior
    ep_views : EPViews
    tickers : list[str]
    w_mkt : (N,) pesos del benchmark
    delta : float
    q : float
        Parámetro de Tsallis. Default 2.0 (óptimo demostrado).
    confidence : float
    num_portf : int

    Returns
    -------
    ModelResult
    """
    from entropy_pooling_v2 import entropy_prog, efficient_frontier, FrontierOptions

    J, N = X.shape
    p = p / p.sum()

    # Resolver q-Tsallis EP
    p_view, _, _ = entropy_prog(
        p, ep_views.A, ep_views.b, ep_views.Aeq, ep_views.beq,
        entropy_family="T", q=q
    )

    # Mezcla con confianza global
    p_post = (1 - confidence) * p + confidence * p_view
    p_post = np.maximum(p_post, 0.0)
    p_post = p_post / p_post.sum()

    # Momentos posteriores
    mu_post = X.T @ p_post
    Scnd = X.T @ (X * p_post[:, None])
    Scnd = 0.5 * (Scnd + Scnd.T)
    Sigma_post = Scnd - np.outer(mu_post, mu_post)

    # Portafolio óptimo
    w_opt = _mean_variance_optimal(mu_post, Sigma_post, delta=delta, long_only=True, max_weight=max_weight)

    # Frontera eficiente
    opts = FrontierOptions(NumPortf=num_portf, FrontierSpan=(0.3, 0.9))
    frontier_e, frontier_s, frontier_w, _, _ = efficient_frontier(X, p_post, opts)

    # Medidas de riesgo
    risk = _compute_risk_metrics(w_opt, mu_post, Sigma_post, X=X, p=p_post)

    return ModelResult(
        model_name=f"q-Tsallis-EP (q={q})",
        w_optimal=w_opt,
        mu_posterior=mu_post,
        Sigma_posterior=Sigma_post,
        frontier_e=frontier_e,
        frontier_s=frontier_s,
        frontier_w=frontier_w,
        p_posterior=p_post,
        risk_metrics=risk,
        params={"entropy_family": "T", "q": q, "confidence": confidence},
    )


# ═══════════════════════════════════════════════════════════════════════
# Utilidades de comparación y visualización
# ═══════════════════════════════════════════════════════════════════════

def print_model_comparison(
    results: List[ModelResult],
    tickers: List[str],
    w_mkt: np.ndarray,
) -> None:
    """
    Imprime una tabla comparativa de los resultados de los modelos.
    """
    N = len(tickers)
    max_ticker_len = max(len(t) for t in tickers)

    print(f"\n{'='*90}")
    print(f"  COMPARACIÓN DE MODELOS")
    print(f"{'='*90}")

    # ── Pesos óptimos ──
    print(f"\n  PESOS ÓPTIMOS (%):")
    header = f"  {'Ticker':<{max_ticker_len+2}s}  {'Benchmark':>10s}"
    for r in results:
        header += f"  {r.model_name:>16s}"
    print(header)
    print("  " + "-" * (max_ticker_len + 2 + 12 + 18 * len(results)))

    for i, t in enumerate(tickers):
        row = f"  {t:<{max_ticker_len+2}s}  {w_mkt[i]*100:>10.2f}"
        for r in results:
            row += f"  {r.w_optimal[i]*100:>16.2f}"
        print(row)

    # Totales
    row = f"  {'TOTAL':<{max_ticker_len+2}s}  {w_mkt.sum()*100:>10.2f}"
    for r in results:
        row += f"  {r.w_optimal.sum()*100:>16.2f}"
    print(row)

    # ── Retornos esperados ──
    print(f"\n  RETORNOS ESPERADOS POSTERIORES (% diario):")
    header = f"  {'Ticker':<{max_ticker_len+2}s}"
    for r in results:
        header += f"  {r.model_name:>16s}"
    print(header)
    print("  " + "-" * (max_ticker_len + 2 + 18 * len(results)))

    for i, t in enumerate(tickers):
        row = f"  {t:<{max_ticker_len+2}s}"
        for r in results:
            row += f"  {r.mu_posterior[i]*100:>16.4f}"
        print(row)

    # ── Medidas de riesgo del portafolio ──
    print(f"\n  MEDIDAS DE RIESGO DEL PORTAFOLIO:")
    header = f"  {'Métrica':<30s}"
    for r in results:
        header += f"  {r.model_name:>16s}"
    print(header)
    print("  " + "-" * (30 + 18 * len(results)))

    metric_labels = {
        "expected_return_annual": ("E[R] anual (%)", 100),
        "volatility_annual": ("Vol anual (%)", 100),
        "sharpe_ratio": ("Sharpe Ratio", 1),
        "VaR_95": ("VaR 95% diario (%)", 100),
        "CVaR_95": ("CVaR 95% diario (%)", 100),
        "VaR_99": ("VaR 99% diario (%)", 100),
        "CVaR_99": ("CVaR 99% diario (%)", 100),
    }

    for key, (label, mult) in metric_labels.items():
        row = f"  {label:<30s}"
        for r in results:
            val = r.risk_metrics.get(key, np.nan)
            row += f"  {val * mult:>16.4f}"
        print(row)

    print(f"\n{'='*90}")


def plot_model_comparison(
    results: List[ModelResult],
    tickers: List[str],
    w_mkt: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Genera gráficos comparativos de los tres modelos.

    Panel 1: Pesos óptimos (barras agrupadas)
    Panel 2: Fronteras eficientes superpuestas
    Panel 3: Retornos esperados posteriores
    Panel 4: Medidas de riesgo
    """
    import matplotlib.pyplot as plt

    N = len(tickers)
    n_models = len(results)
    colors = ["#2E86AB", "#E8475A", "#F5A623", "#7B68EE"]

    fig = plt.figure(figsize=(18, 12))

    # ── Panel 1: Pesos óptimos ──
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(N)
    width = 0.8 / (n_models + 1)

    ax1.bar(x - width * n_models / 2, w_mkt * 100, width,
            label="Benchmark", color="gray", alpha=0.5, edgecolor="black")
    for i, r in enumerate(results):
        offset = width * (i - n_models / 2 + 1)
        ax1.bar(x + offset, r.w_optimal * 100, width,
                label=r.model_name, color=colors[i % len(colors)], alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Peso (%)")
    ax1.set_title("Pesos óptimos por modelo", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Panel 2: Fronteras eficientes ──
    ax2 = fig.add_subplot(2, 2, 2)
    for i, r in enumerate(results):
        ax2.plot(r.frontier_s * 100, r.frontier_e * 100, "o-",
                 color=colors[i % len(colors)], markersize=3, linewidth=1.5,
                 label=r.model_name)
        # Marcar portafolio óptimo
        vol_opt = float(np.sqrt(r.w_optimal @ r.Sigma_posterior @ r.w_optimal))
        ret_opt = float(r.w_optimal @ r.mu_posterior)
        ax2.plot(vol_opt * 100, ret_opt * 100, "D",
                 color=colors[i % len(colors)], markersize=10, markeredgecolor="black")

    ax2.set_xlabel("Volatilidad (%)")
    ax2.set_ylabel("Retorno esperado (%)")
    ax2.set_title("Fronteras eficientes", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Retornos esperados ──
    ax3 = fig.add_subplot(2, 2, 3)
    for i, r in enumerate(results):
        ax3.barh(x + width * (i - n_models / 2 + 0.5), r.mu_posterior * 100, width,
                 label=r.model_name, color=colors[i % len(colors)], alpha=0.8)

    ax3.set_yticks(x)
    ax3.set_yticklabels(tickers, fontsize=8)
    ax3.set_xlabel("Retorno esperado diario (%)")
    ax3.set_title("Retornos esperados posteriores", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="x")

    # ── Panel 4: Medidas de riesgo ──
    ax4 = fig.add_subplot(2, 2, 4)
    risk_keys = ["expected_return_annual", "volatility_annual", "sharpe_ratio"]
    risk_labels = ["E[R] anual", "Vol anual", "Sharpe"]
    x_risk = np.arange(len(risk_keys))
    width_r = 0.8 / n_models

    for i, r in enumerate(results):
        vals = [r.risk_metrics.get(k, 0) * (100 if "ratio" not in k else 1) for k in risk_keys]
        ax4.bar(x_risk + width_r * (i - n_models / 2 + 0.5), vals, width_r,
                label=r.model_name, color=colors[i % len(colors)], alpha=0.8)

    ax4.set_xticks(x_risk)
    ax4.set_xticklabels(risk_labels)
    ax4.set_title("Métricas de riesgo-retorno", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Comparación: BL vs EP-Shannon vs q-Tsallis-EP", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")

    return fig
