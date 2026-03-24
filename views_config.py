"""
views_config.py
===============
Sistema unificado de especificación de views para los tres modelos:
  - Black-Litterman (BL)
  - Entropy Pooling (EP)
  - q-Tsallis Entropy Pooling (q-Tsallis-EP)

Permite definir views de forma intuitiva mediante diccionarios y los traduce
automáticamente a las estructuras matemáticas que requiere cada modelo:
  - BL:          matrices P, Q, Ω
  - EP/q-EP:     matrices A (desigualdad), b, Aeq (igualdad), beq

Tipos de views soportados:
  1. Absoluto:       "YPFD rinde 5%"
  2. Relativo:       "YPFD rinde 2% más que PAMP"
  3. Ranking:        "YPFD > PAMP > TGS > CEPU" (ordenamiento)
  4. Volatilidad:    "La volatilidad de YPFD será 30%"
  5. Cola (tail):    "P(R_YPFD < -10%) ≤ 5%"

Los views 1-3 son compatibles con BL y EP/q-EP.
Los views 4-5 requieren la distribución completa y solo son compatibles con EP/q-EP.

Referencia teórica:
  - Black & Litterman (1992): views lineales sobre retornos esperados
  - Meucci (2008): Fully Flexible Views, constraints sobre la distribución
  - Idzorek (2007): calibración de confianza vía tau y Omega
  - He & Litterman (2002): intuición del modelo BL

Uso:
    from views_config import ViewSpec, build_views

    views = [
        ViewSpec.absolute("YPFD.BA", expected_return=0.05, confidence=0.8),
        ViewSpec.relative("YPFD.BA", "PAMP.BA", spread=0.02, confidence=0.6),
        ViewSpec.ranking(["YPFD.BA", "PAMP.BA", "TGS.BA", "CEPU.BA"], confidence=0.7),
        ViewSpec.volatility("YPFD.BA", target_vol=0.30, confidence=0.5),
        ViewSpec.tail("YPFD.BA", threshold=-0.10, max_prob=0.05, confidence=0.5),
    ]

    bl_views, ep_views = build_views(views, tickers, Sigma, tau=0.05)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1. Tipos de views
# ═══════════════════════════════════════════════════════════════════════

class ViewType(Enum):
    """Tipos de views soportados por el framework."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    RANKING = "ranking"
    VOLATILITY = "volatility"
    TAIL = "tail"


@dataclass
class ViewSpec:
    """
    Especificación de un view individual.

    Cada view se crea mediante los class methods estáticos (absolute, relative, etc.)
    que proveen una interfaz intuitiva para el usuario.

    Attributes
    ----------
    view_type : ViewType
        Tipo de view.
    tickers : list[str]
        Tickers involucrados en el view.
    params : dict
        Parámetros específicos del tipo de view.
    confidence : float
        Nivel de confianza subjetiva del inversor en este view, entre 0 y 1.
        - 0: confianza nula (el view no tiene efecto)
        - 1: confianza total (el view domina al prior)
    label : str
        Descripción legible del view (generada automáticamente).
    bl_compatible : bool
        Si el view es compatible con Black-Litterman (solo views lineales sobre E[R]).
    """
    view_type: ViewType
    tickers: List[str]
    params: Dict[str, object]
    confidence: float
    label: str
    bl_compatible: bool

    # ── Constructores intuitivos ──────────────────────────────────────

    @staticmethod
    def absolute(
        ticker: str,
        expected_return: float,
        confidence: float = 0.5,
    ) -> "ViewSpec":
        """
        View absoluto: "El activo {ticker} tendrá un retorno esperado de {expected_return}".

        En BL: P_k = e_i^T (vector canónico del activo i), Q_k = expected_return.
        En EP: constraint de igualdad E_p̃[X_i] = expected_return.

        Parameters
        ----------
        ticker : str
            Ticker del activo (ej: "YPFD.BA").
        expected_return : float
            Retorno esperado del view (ej: 0.05 para 5%).
        confidence : float
            Confianza subjetiva, entre 0 y 1.
        """
        return ViewSpec(
            view_type=ViewType.ABSOLUTE,
            tickers=[ticker],
            params={"expected_return": expected_return},
            confidence=confidence,
            label=f"E[R_{ticker}] = {expected_return:.2%}",
            bl_compatible=True,
        )

    @staticmethod
    def relative(
        ticker_long: str,
        ticker_short: str,
        spread: float,
        confidence: float = 0.5,
    ) -> "ViewSpec":
        """
        View relativo: "{ticker_long} rendirá {spread} más que {ticker_short}".

        En BL: P_k = e_long^T - e_short^T, Q_k = spread.
        En EP: constraint de igualdad E_p̃[X_long - X_short] = spread.

        Parameters
        ----------
        ticker_long : str
            Ticker del activo que se espera rinda más.
        ticker_short : str
            Ticker del activo que se espera rinda menos.
        spread : float
            Diferencia de retorno esperada (ej: 0.02 para 2%).
        confidence : float
            Confianza subjetiva, entre 0 y 1.
        """
        return ViewSpec(
            view_type=ViewType.RELATIVE,
            tickers=[ticker_long, ticker_short],
            params={"spread": spread},
            confidence=confidence,
            label=f"E[R_{ticker_long}] - E[R_{ticker_short}] = {spread:.2%}",
            bl_compatible=True,
        )

    @staticmethod
    def ranking(
        tickers_ordered: List[str],
        confidence: float = 0.5,
    ) -> "ViewSpec":
        """
        View de ranking: "{tickers_ordered[0]} > {tickers_ordered[1]} > ... > {tickers_ordered[-1]}".

        Genera K-1 constraints de desigualdad: E_p̃[X_i] >= E_p̃[X_{i+1}] para cada par
        consecutivo en el ranking.

        En BL: se traduce a K-1 views relativos con spread=0 (o un spread mínimo ε>0).
        En EP: constraints de desigualdad sobre las expectativas.

        Parameters
        ----------
        tickers_ordered : list[str]
            Tickers ordenados de mayor a menor retorno esperado.
            Ej: ["YPFD.BA", "PAMP.BA", "TGS.BA", "CEPU.BA"]
            significa E[R_YPFD] >= E[R_PAMP] >= E[R_TGS] >= E[R_CEPU].
        confidence : float
            Confianza subjetiva en el ranking completo.
        """
        ranking_str = " > ".join(tickers_ordered)
        return ViewSpec(
            view_type=ViewType.RANKING,
            tickers=list(tickers_ordered),
            params={},
            confidence=confidence,
            label=f"Ranking: {ranking_str}",
            bl_compatible=True,  # se traduce a views relativos para BL
        )

    @staticmethod
    def volatility(
        ticker: str,
        target_vol: float,
        confidence: float = 0.5,
    ) -> "ViewSpec":
        """
        View sobre volatilidad: "La volatilidad de {ticker} será {target_vol}".

        Este view solo es compatible con EP/q-EP ya que involucra el segundo momento.

        En EP: constraint sobre E_p̃[X_i²] = target_vol² + (E_p̃[X_i])²
               es decir, Var_p̃(X_i) = target_vol².

        Parameters
        ----------
        ticker : str
            Ticker del activo.
        target_vol : float
            Volatilidad target (desvío estándar, ej: 0.30 para 30%).
        confidence : float
            Confianza subjetiva.
        """
        return ViewSpec(
            view_type=ViewType.VOLATILITY,
            tickers=[ticker],
            params={"target_vol": target_vol},
            confidence=confidence,
            label=f"σ({ticker}) = {target_vol:.2%}",
            bl_compatible=False,
        )

    @staticmethod
    def tail(
        ticker: str,
        threshold: float,
        max_prob: float,
        confidence: float = 0.5,
    ) -> "ViewSpec":
        """
        View sobre colas: "P(R_{ticker} < threshold) <= max_prob".

        Solo compatible con EP/q-EP ya que requiere la distribución completa.

        En EP: para cada escenario j, si X_{j,i} < threshold entonces
               la constraint es Σ_{j: X_{j,i}<threshold} p̃_j <= max_prob.

        Parameters
        ----------
        ticker : str
            Ticker del activo.
        threshold : float
            Umbral de retorno (ej: -0.10 para -10%).
        max_prob : float
            Probabilidad máxima en la cola (ej: 0.05 para 5%).
        confidence : float
            Confianza subjetiva.
        """
        return ViewSpec(
            view_type=ViewType.TAIL,
            tickers=[ticker],
            params={"threshold": threshold, "max_prob": max_prob},
            confidence=confidence,
            label=f"P(R_{ticker} < {threshold:.2%}) ≤ {max_prob:.2%}",
            bl_compatible=False,
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Traducción a estructuras de BL: P, Q, Ω
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BLViews:
    """
    Estructura de views para Black-Litterman.

    P : (K, N)  — Pick matrix. Cada fila es un view.
    Q : (K,)    — Vector de retornos esperados de los views.
    Omega : (K, K) — Matriz de incertidumbre de los views.
    labels : list[str] — Descripciones de cada view.

    La ecuación de BL es:
        μ_BL = (τΣ⁻¹ + P^T Ω⁻¹ P)⁻¹ (τΣ⁻¹ Π + P^T Ω⁻¹ Q)
    """
    P: np.ndarray
    Q: np.ndarray
    Omega: np.ndarray
    labels: List[str]


def _ticker_index(ticker: str, tickers: List[str]) -> int:
    """Busca el índice de un ticker en la lista. Lanza error claro si no existe."""
    try:
        return tickers.index(ticker)
    except ValueError:
        raise ValueError(
            f"Ticker '{ticker}' no encontrado en la lista de activos: {tickers}"
        )


def build_bl_views(
    views: List[ViewSpec],
    tickers: List[str],
    Sigma: np.ndarray,
    tau: float = 0.05,
    base_variance: float = 0.02,
) -> BLViews:
    """
    Traduce una lista de ViewSpec a la estructura P, Q, Ω de Black-Litterman.

    Parameters
    ----------
    views : list[ViewSpec]
        Views definidos por el usuario.
    tickers : list[str]
        Lista de tickers en el mismo orden que las columnas de Sigma.
    Sigma : (N, N) np.ndarray
        Matriz de covarianzas de los retornos.
    tau : float
        Escalar de incertidumbre del prior (típicamente 0.02-0.10).
    base_variance : float
        Varianza base para la calibración de Ω según la confianza.
        Se usa la fórmula de Idzorek (2007):
            ω_k = ((1 - c_k) / c_k) * τ * (P_k Σ P_k^T)

    Returns
    -------
    BLViews
        Estructura con P, Q, Ω, labels.
    """
    N = len(tickers)
    P_rows = []
    Q_vals = []
    omega_diag = []
    labels = []

    for v in views:
        if not v.bl_compatible:
            continue

        c = max(min(v.confidence, 0.999), 0.001)  # clamp para evitar div/0

        if v.view_type == ViewType.ABSOLUTE:
            idx = _ticker_index(v.tickers[0], tickers)
            p_row = np.zeros(N)
            p_row[idx] = 1.0
            P_rows.append(p_row)
            Q_vals.append(v.params["expected_return"])
            # Ω_k = ((1-c)/c) * τ * P_k Σ P_k^T  (Idzorek, 2007)
            view_var = float(p_row @ Sigma @ p_row) * tau
            omega_k = ((1 - c) / c) * view_var
            omega_diag.append(omega_k)
            labels.append(v.label)

        elif v.view_type == ViewType.RELATIVE:
            idx_long = _ticker_index(v.tickers[0], tickers)
            idx_short = _ticker_index(v.tickers[1], tickers)
            p_row = np.zeros(N)
            p_row[idx_long] = 1.0
            p_row[idx_short] = -1.0
            P_rows.append(p_row)
            Q_vals.append(v.params["spread"])
            view_var = float(p_row @ Sigma @ p_row) * tau
            omega_k = ((1 - c) / c) * view_var
            omega_diag.append(omega_k)
            labels.append(v.label)

        elif v.view_type == ViewType.RANKING:
            # Ranking se traduce a K-1 views relativos con spread ε ≈ 0
            # E[R_i] - E[R_{i+1}] >= 0 para cada par consecutivo
            # En BL lo modelamos como views relativos con Q_k = ε (spread mínimo)
            epsilon = 1e-6  # spread mínimo para evitar singularidad
            for j in range(len(v.tickers) - 1):
                idx_upper = _ticker_index(v.tickers[j], tickers)
                idx_lower = _ticker_index(v.tickers[j + 1], tickers)
                p_row = np.zeros(N)
                p_row[idx_upper] = 1.0
                p_row[idx_lower] = -1.0
                P_rows.append(p_row)
                Q_vals.append(epsilon)
                view_var = float(p_row @ Sigma @ p_row) * tau
                omega_k = ((1 - c) / c) * view_var
                omega_diag.append(omega_k)
                labels.append(f"E[R_{v.tickers[j]}] ≥ E[R_{v.tickers[j+1]}]")

    if not P_rows:
        raise ValueError("No se encontraron views compatibles con Black-Litterman")

    P = np.array(P_rows)
    Q = np.array(Q_vals)
    Omega = np.diag(omega_diag)

    return BLViews(P=P, Q=Q, Omega=Omega, labels=labels)


# ═══════════════════════════════════════════════════════════════════════
# 3. Traducción a estructuras de EP: A, b, Aeq, beq
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EPViews:
    """
    Estructura de views para Entropy Pooling y q-Tsallis-EP.

    A    : (K_ineq, J) — Constraints de desigualdad: A p̃ ≤ b
    b    : (K_ineq,)
    Aeq  : (K_eq, J)   — Constraints de igualdad: Aeq p̃ = beq
    beq  : (K_eq,)
    labels : list[str]  — Descripciones de cada constraint.

    La constraint fundamental Σ p̃_j = 1 siempre se incluye.
    """
    A: np.ndarray
    b: np.ndarray
    Aeq: np.ndarray
    beq: np.ndarray
    labels: List[str]


def build_ep_views(
    views: List[ViewSpec],
    tickers: List[str],
    X: np.ndarray,
    p: np.ndarray,
) -> EPViews:
    """
    Traduce una lista de ViewSpec a la estructura A, b, Aeq, beq de Entropy Pooling.

    Parameters
    ----------
    views : list[ViewSpec]
        Views definidos por el usuario.
    tickers : list[str]
        Lista de tickers (mismo orden que columnas de X).
    X : (J, N) np.ndarray
        Matriz de realizaciones de retornos (escenarios × activos).
    p : (J,) np.ndarray
        Vector de probabilidades del prior (suman 1).

    Returns
    -------
    EPViews
        Estructura con A, b, Aeq, beq, labels.
    """
    J, N = X.shape
    p = p / p.sum()

    # Siempre: constraint Σ p̃_j = 1
    Aeq_rows = [np.ones(J)]
    beq_vals = [1.0]
    eq_labels = ["Σ p̃_j = 1"]

    A_rows = []
    b_vals = []
    ineq_labels = []

    for v in views:
        c = v.confidence

        if v.view_type == ViewType.ABSOLUTE:
            # Constraint de igualdad: E_p̃[X_i] = expected_return
            # Mezcla con confianza: E_p̃[X_i] = c * expected_return + (1-c) * E_p[X_i]
            idx = _ticker_index(v.tickers[0], tickers)
            row = X[:, idx]  # (J,)
            prior_mean = float(p @ X[:, idx])
            target = c * v.params["expected_return"] + (1 - c) * prior_mean
            Aeq_rows.append(row)
            beq_vals.append(target)
            eq_labels.append(v.label)

        elif v.view_type == ViewType.RELATIVE:
            # Constraint: E_p̃[X_long - X_short] = spread
            idx_long = _ticker_index(v.tickers[0], tickers)
            idx_short = _ticker_index(v.tickers[1], tickers)
            row = X[:, idx_long] - X[:, idx_short]  # (J,)
            prior_spread = float(p @ row)
            target = c * v.params["spread"] + (1 - c) * prior_spread
            Aeq_rows.append(row)
            beq_vals.append(target)
            eq_labels.append(v.label)

        elif v.view_type == ViewType.RANKING:
            # K-1 constraints de desigualdad: E_p̃[X_upper] >= E_p̃[X_lower]
            # Equivalente: E_p̃[X_lower - X_upper] <= 0
            for j in range(len(v.tickers) - 1):
                idx_upper = _ticker_index(v.tickers[j], tickers)
                idx_lower = _ticker_index(v.tickers[j + 1], tickers)
                # V = X_lower - X_upper; A p̃ ≤ 0
                row = X[:, idx_lower] - X[:, idx_upper]  # (J,)
                A_rows.append(row)
                b_vals.append(0.0)
                ineq_labels.append(
                    f"E[R_{v.tickers[j]}] ≥ E[R_{v.tickers[j+1]}]"
                )

        elif v.view_type == ViewType.VOLATILITY:
            # Constraint sobre segundo momento:
            # E_p̃[X_i²] = σ_target² + (E_p̃[X_i])²
            # Esto es un constraint no lineal, pero podemos linealizarlo
            # usando el segundo momento directamente:
            # E_p̃[X_i²] ≈ σ_target² + μ_prior²  (usando media del prior como aprox)
            idx = _ticker_index(v.tickers[0], tickers)
            target_vol = v.params["target_vol"]
            prior_mean = float(p @ X[:, idx])
            prior_var = float(p @ (X[:, idx] ** 2)) - prior_mean**2
            target_second_moment = target_vol**2 + prior_mean**2
            # Mezcla con confianza
            prior_second_moment = prior_var + prior_mean**2
            target = c * target_second_moment + (1 - c) * prior_second_moment
            row = X[:, idx] ** 2  # (J,)
            Aeq_rows.append(row)
            beq_vals.append(target)
            eq_labels.append(v.label)

        elif v.view_type == ViewType.TAIL:
            # Constraint de cola:
            # P(X_i < threshold) ≤ max_prob
            # Equivalente: Σ_{j: X_{j,i} < threshold} p̃_j ≤ max_prob
            idx = _ticker_index(v.tickers[0], tickers)
            threshold = v.params["threshold"]
            max_prob = v.params["max_prob"]
            # Indicadora: 1 si X_{j,i} < threshold, 0 otherwise
            indicator = (X[:, idx] < threshold).astype(float)  # (J,)
            # Mezcla con confianza
            prior_tail_prob = float(p @ indicator)
            target = c * max_prob + (1 - c) * prior_tail_prob
            A_rows.append(indicator)
            b_vals.append(target)
            ineq_labels.append(v.label)

    # Construir matrices
    Aeq = np.array(Aeq_rows)   # (K_eq, J)
    beq = np.array(beq_vals)   # (K_eq,)
    if A_rows:
        A = np.array(A_rows)   # (K_ineq, J)
        b = np.array(b_vals)   # (K_ineq,)
    else:
        A = np.zeros((0, J))
        b = np.zeros(0)

    all_labels = eq_labels + ineq_labels

    return EPViews(A=A, b=b, Aeq=Aeq, beq=beq, labels=all_labels)


# ═══════════════════════════════════════════════════════════════════════
# 4. Función unificada: build_views
# ═══════════════════════════════════════════════════════════════════════

def build_views(
    views: List[ViewSpec],
    tickers: List[str],
    Sigma: np.ndarray,
    X: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    tau: float = 0.05,
) -> Tuple[Optional[BLViews], Optional[EPViews]]:
    """
    Función unificada que traduce views a ambos formatos (BL y EP).

    Parameters
    ----------
    views : list[ViewSpec]
        Lista de views del usuario.
    tickers : list[str]
        Tickers en el orden de las columnas de Sigma y X.
    Sigma : (N, N) np.ndarray
        Matriz de covarianzas.
    X : (J, N) np.ndarray, optional
        Matriz de escenarios (requerida para EP/q-EP).
    p : (J,) np.ndarray, optional
        Prior de probabilidades (requerido para EP/q-EP).
    tau : float
        Escalar de incertidumbre del prior para BL.

    Returns
    -------
    bl_views : BLViews or None
        Views en formato BL (None si no hay views BL-compatibles).
    ep_views : EPViews or None
        Views en formato EP (None si no se proveyeron X y p).
    """
    # BL views
    bl_compatible = [v for v in views if v.bl_compatible]
    bl_views = None
    if bl_compatible:
        try:
            bl_views = build_bl_views(views, tickers, Sigma, tau=tau)
        except Exception as e:
            print(f"Warning: no se pudieron construir views BL: {e}")

    # EP views
    ep_views = None
    if X is not None and p is not None:
        try:
            ep_views = build_ep_views(views, tickers, X, p)
        except Exception as e:
            print(f"Warning: no se pudieron construir views EP: {e}")

    return bl_views, ep_views


# ═══════════════════════════════════════════════════════════════════════
# 5. Utilidades de diagnóstico
# ═══════════════════════════════════════════════════════════════════════

def print_views_summary(views: List[ViewSpec]) -> None:
    """Imprime un resumen legible de los views configurados."""
    print(f"\n{'='*70}")
    print(f"  RESUMEN DE VIEWS ({len(views)} views)")
    print(f"{'='*70}")
    for i, v in enumerate(views, 1):
        compat = "BL+EP" if v.bl_compatible else "EP only"
        print(f"  [{i}] {v.label}")
        print(f"      Tipo: {v.view_type.value}  |  Confianza: {v.confidence:.0%}  |  Compatibilidad: {compat}")
    print(f"{'='*70}\n")


def print_bl_views(bl: BLViews, tickers: List[str]) -> None:
    """Imprime las matrices P, Q, Ω de BL de forma legible."""
    K, N = bl.P.shape
    print(f"\n{'='*70}")
    print(f"  BLACK-LITTERMAN: P ({K}×{N}), Q ({K},), Ω ({K}×{K})")
    print(f"{'='*70}")
    print(f"\n  Pick Matrix P (cada fila = un view):")
    header = "  " + "".join(f"{t:>10s}" for t in tickers)
    print(header)
    for k in range(K):
        row = "  " + "".join(f"{bl.P[k, n]:>10.3f}" for n in range(N))
        print(f"{row}  ← {bl.labels[k]}")
    print(f"\n  Q (retornos esperados de views): {bl.Q}")
    print(f"\n  Ω (diagonal de incertidumbre): {np.diag(bl.Omega)}")
    print(f"{'='*70}\n")


def print_ep_views(ep: EPViews) -> None:
    """Imprime un resumen de las constraints de EP."""
    K_eq = ep.Aeq.shape[0]
    K_ineq = ep.A.shape[0]
    J = ep.Aeq.shape[1]
    print(f"\n{'='*70}")
    print(f"  ENTROPY POOLING: {K_eq} igualdades, {K_ineq} desigualdades, {J} escenarios")
    print(f"{'='*70}")
    for i, label in enumerate(ep.labels):
        tipo = "EQ" if i < K_eq else "INEQ"
        print(f"  [{tipo}] {label}")
    print(f"{'='*70}\n")
