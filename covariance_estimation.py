"""
covariance_estimation.py
========================
Estimación robusta de la matriz de covarianzas para paneles desbalanceados.

Problema:
  Cuando un papel nuevo tiene menos historia que los demás, la matriz de retornos
  tiene NaN en las primeras filas para ese activo. Si se rellena con bfill() se
  fabrican retornos cero que distorsionan la covarianza y las correlaciones.

Solución:
  1. Solo se hace bfill/ffill para huecos cortos (días sin operar intercalados).
  2. Los NaN de inicio de serie (papel nuevo) se dejan como están.
  3. La covarianza se estima por pairwise: cada par usa su máximo solapamiento.
  4. Si la Σ resultante no es PSD, se corrige con el algoritmo de Higham
     (nearest symmetric positive semidefinite matrix en norma de Frobenius).
  5. Si la Σ ya es PSD, no se toca — se devuelve tal cual.

Uso:
    from covariance_estimation import prepare_prices, estimate_covariance

Ref: Higham, N. J. (2002). Computing the nearest correlation matrix—a problem
     from finance. IMA Journal of Numerical Analysis, 22(3), 329–343.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 1. PREPARACIÓN DE PRECIOS — MANEJO INTELIGENTE DE NaN
# ═══════════════════════════════════════════════════════════════

def prepare_prices(
    prices: pd.DataFrame,
    max_gap_fill: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepara la serie de precios rellenando solo huecos cortos (días sin operar)
    y dejando intactos los NaN de inicio de serie (papel nuevo sin historia).

    Parameters
    ----------
    prices : pd.DataFrame
        Precios con DatetimeIndex. Columnas = tickers.
        Puede tener NaN al inicio (papel nuevo) y/o intercalados (días sin operar).
    max_gap_fill : int
        Cantidad máxima de días consecutivos a rellenar con interpolación/ffill.
        Huecos más largos que esto se consideran "papel sin historia" y se dejan
        como NaN. Default: 5 días hábiles (una semana de mercado).
    verbose : bool
        Si True, imprime un resumen de lo que hizo.

    Returns
    -------
    pd.DataFrame
        Precios con huecos cortos rellenados pero NaN de inicio intactos.
    """
    prices_out = prices.copy()

    for col in prices_out.columns:
        serie = prices_out[col]

        # Identificar el primer dato válido de este papel
        first_valid_idx = serie.first_valid_index()
        if first_valid_idx is None:
            # Columna completamente vacía — la dejamos
            if verbose:
                print(f"  ⚠ {col}: sin datos, se deja vacía")
            continue

        # Separar: NaN "de inicio" (antes del primer dato) vs huecos intercalados
        first_valid_pos = serie.index.get_loc(first_valid_idx)
        nan_inicio = first_valid_pos  # cantidad de NaN al arranque

        # Solo trabajar sobre la parte con datos (desde first_valid en adelante)
        serie_con_datos = serie.iloc[first_valid_pos:]

        # Detectar huecos intercalados y rellenar solo los cortos
        serie_rellena = _fill_short_gaps(serie_con_datos, max_gap=max_gap_fill)

        # Recomponer: NaN de inicio + datos rellenados
        prices_out[col] = pd.concat([
            serie.iloc[:first_valid_pos],   # NaN de inicio (intactos)
            serie_rellena                    # datos con huecos cortos rellenados
        ])

        # Reporte
        if verbose:
            nan_restantes = prices_out[col].isna().sum()
            huecos_rellenados = serie_con_datos.isna().sum() - serie_rellena.isna().sum()
            if nan_inicio > 0 or huecos_rellenados > 0:
                print(f"  {col}: {nan_inicio} NaN de inicio (intactos), "
                      f"{huecos_rellenados} huecos cortos rellenados, "
                      f"{nan_restantes} NaN totales restantes")

    return prices_out


def _fill_short_gaps(serie: pd.Series, max_gap: int) -> pd.Series:
    """
    Rellena con interpolación lineal + ffill solo los huecos de hasta `max_gap`
    días consecutivos. Los huecos más largos se dejan como NaN.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal (ya recortada desde el primer dato válido).
    max_gap : int
        Máximo de NaN consecutivos que se rellenan.

    Returns
    -------
    pd.Series
        Serie con huecos cortos rellenados.
    """
    # interpolate con limit rellena como máximo `limit` NaN consecutivos
    rellena = serie.interpolate(method="linear", limit=max_gap)
    # ffill para el caso de que el último dato sea NaN (máx `max_gap` días)
    rellena = rellena.ffill(limit=max_gap)
    return rellena


def get_panel_diagnostics(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con diagnóstico de cobertura por activo.
    Útil para entender el desbalance del panel antes de estimar.

    Parameters
    ----------
    prices : pd.DataFrame
        Precios con DatetimeIndex.

    Returns
    -------
    pd.DataFrame con columnas:
        - ticker, primera_fecha, ultima_fecha, dias_con_dato,
          dias_totales, cobertura_pct, nan_inicio, nan_intercalados
    """
    rows = []
    total_fechas = len(prices)
    for col in prices.columns:
        serie = prices[col]
        first = serie.first_valid_index()
        last = serie.last_valid_index()
        n_datos = serie.notna().sum()
        if first is not None:
            first_pos = serie.index.get_loc(first)
            nan_inicio = first_pos
            nan_intercalados = serie.iloc[first_pos:].isna().sum()
        else:
            nan_inicio = total_fechas
            nan_intercalados = 0
        rows.append({
            "ticker": col,
            "primera_fecha": first,
            "ultima_fecha": last,
            "dias_con_dato": int(n_datos),
            "dias_totales": total_fechas,
            "cobertura_pct": round(n_datos / total_fechas * 100, 1),
            "nan_inicio": int(nan_inicio),
            "nan_intercalados": int(nan_intercalados),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 2. ESTIMACIÓN PAIRWISE DE LA COVARIANZA
# ═══════════════════════════════════════════════════════════════

def estimate_covariance(
    returns: pd.DataFrame,
    min_obs_pairwise: int = 60,
    verbose: bool = True,
) -> np.ndarray:
    """
    Estima la matriz de covarianzas de forma robusta para paneles desbalanceados.

    Estrategia:
      - Si no hay NaN en los retornos → usa np.cov estándar (ddof=1).
      - Si hay NaN → estima pairwise y, si es necesario, corrige con Higham.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos con posibles NaN (papeles con distinta historia).
    min_obs_pairwise : int
        Mínimo de observaciones solapadas requeridas para estimar la covarianza
        de un par de activos. Si un par tiene menos, se usa la varianza
        individual y correlación cero (enfoque conservador).
        Default: 60 (aprox. 3 meses de datos diarios).
    verbose : bool
        Si True, imprime información del proceso.

    Returns
    -------
    np.ndarray (N, N)
        Matriz de covarianzas simétrica y PSD garantizada.
    """
    has_nan = returns.isna().any().any()

    if not has_nan:
        # ── Caso simple: panel balanceado, sin NaN ──
        if verbose:
            print("  Panel balanceado — covarianza estándar (sin corrección)")
        Sigma = np.cov(returns.values.T, ddof=1)
        Sigma = 0.5 * (Sigma + Sigma.T)  # simetría numérica
        return Sigma

    # ── Caso con NaN: estimación pairwise ──
    if verbose:
        n_nan_cols = returns.isna().any().sum()
        print(f"  Panel desbalanceado ({n_nan_cols} activos con NaN) "
              f"— estimación pairwise")

    tickers = returns.columns.tolist()
    N = len(tickers)
    Sigma = np.zeros((N, N))

    # Matriz de conteo de observaciones pairwise (para diagnóstico)
    obs_count = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(i, N):
            # Máscara: fechas donde ambos tienen dato
            ri = returns.iloc[:, i]
            rj = returns.iloc[:, j]
            mask = ri.notna() & rj.notna()
            n_obs = mask.sum()
            obs_count[i, j] = n_obs
            obs_count[j, i] = n_obs

            if n_obs >= min_obs_pairwise:
                # Covarianza del par con datos solapados
                xi = ri[mask].values
                xj = rj[mask].values
                cov_ij = np.cov(xi, xj, ddof=1)[0, 1]
                Sigma[i, j] = cov_ij
                Sigma[j, i] = cov_ij

                # Varianza en la diagonal (usar toda la historia individual)
                if i == j:
                    all_data = ri.dropna().values
                    Sigma[i, i] = np.var(all_data, ddof=1)
            else:
                # Insuficientes observaciones solapadas:
                # varianza individual en la diagonal, covarianza cero fuera
                if verbose and i != j:
                    print(f"  ⚠ {tickers[i]}-{tickers[j]}: solo {n_obs} obs "
                          f"(< {min_obs_pairwise}), covarianza = 0")
                if i == j:
                    all_data = ri.dropna().values
                    Sigma[i, i] = np.var(all_data, ddof=1) if len(all_data) > 1 else 0.0

    # Asegurar varianzas en la diagonal (usar toda la historia de cada activo)
    for i in range(N):
        all_data = returns.iloc[:, i].dropna().values
        if len(all_data) > 1:
            Sigma[i, i] = np.var(all_data, ddof=1)

    # Simetría numérica
    Sigma = 0.5 * (Sigma + Sigma.T)

    # ── Verificar y corregir PSD si es necesario ──
    Sigma = _ensure_psd(Sigma, verbose=verbose)

    if verbose:
        min_overlap = obs_count[np.triu_indices(N, k=1)].min()
        max_overlap = obs_count[np.triu_indices(N, k=1)].max()
        print(f"  Solapamiento pairwise: mín={min_overlap}, máx={max_overlap} obs")

    return Sigma


# ═══════════════════════════════════════════════════════════════
# 3. CORRECCIÓN PSD — ALGORITMO DE HIGHAM
# ═══════════════════════════════════════════════════════════════

def _ensure_psd(
    Sigma: np.ndarray,
    tol: float = 1e-10,
    verbose: bool = True,
) -> np.ndarray:
    """
    Verifica si Σ es PSD. Si no lo es, la corrige con Higham.
    Si ya es PSD, la devuelve sin modificar.

    Parameters
    ----------
    Sigma : np.ndarray (N, N)
        Matriz de covarianzas (simétrica).
    tol : float
        Tolerancia para considerar un eigenvalue como negativo.
    verbose : bool

    Returns
    -------
    np.ndarray (N, N)
        Σ corregida (PSD garantizada) o la original si ya era PSD.
    """
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eig = eigenvalues.min()

    if min_eig >= -tol:
        if verbose:
            print(f"  Σ ya es PSD (λ_min = {min_eig:.2e}) — sin corrección")
        # Corregir eigenvalues marginalmente negativos por error numérico
        if min_eig < 0:
            Sigma = Sigma + (-min_eig + tol) * np.eye(Sigma.shape[0])
        return Sigma

    if verbose:
        n_neg = (eigenvalues < -tol).sum()
        print(f"  Σ NO es PSD (λ_min = {min_eig:.2e}, {n_neg} eigenvalues negativos)")
        print(f"  Aplicando corrección de Higham...")

    Sigma_psd = _nearest_psd_higham(Sigma)

    if verbose:
        eig_new = np.linalg.eigvalsh(Sigma_psd)
        frob_error = np.linalg.norm(Sigma_psd - Sigma, 'fro')
        frob_orig = np.linalg.norm(Sigma, 'fro')
        print(f"  ✓ Corregida: λ_min = {eig_new.min():.2e}, "
              f"error Frobenius = {frob_error:.2e} "
              f"({frob_error/frob_orig*100:.4f}% de ||Σ||)")

    return Sigma_psd


def _nearest_psd_higham(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Calcula la matriz simétrica semidefinida positiva más cercana a A
    en norma de Frobenius, preservando la diagonal (varianzas).

    Implementación del algoritmo alternating projections de Higham (2002):
      Proyección alternada entre el cono PSD y el conjunto de matrices
      con la diagonal original.

    Parameters
    ----------
    A : np.ndarray (N, N)
        Matriz simétrica de entrada.
    max_iter : int
        Máximo de iteraciones.
    tol : float
        Tolerancia de convergencia.

    Returns
    -------
    np.ndarray (N, N)
        Nearest PSD matrix.

    Ref: Higham, N. J. (2002). Computing the nearest correlation matrix.
    """
    N = A.shape[0]
    Y = A.copy()
    delta_S = np.zeros_like(A)

    for k in range(max_iter):
        # Corrección de Dykstra
        R = Y - delta_S

        # Proyección al cono PSD: clampear eigenvalues negativos a 0
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0.0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Actualizar corrección de Dykstra
        delta_S = X - R

        # Proyección para preservar la diagonal original (varianzas)
        Y_new = X.copy()
        np.fill_diagonal(Y_new, np.diag(A))

        # Convergencia
        diff = np.linalg.norm(Y_new - Y, 'fro')
        if diff < tol:
            break

        Y = Y_new

    # Asegurar simetría exacta
    Y = 0.5 * (Y + Y.T)

    # Último clamp: por errores numéricos residuales
    eigvals, eigvecs = np.linalg.eigh(Y)
    eigvals = np.maximum(eigvals, 0.0)
    Y = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Y = 0.5 * (Y + Y.T)

    # Restaurar diagonal original (varianzas exactas)
    np.fill_diagonal(Y, np.diag(A))

    return Y


# ═══════════════════════════════════════════════════════════════
# 4. FUNCIÓN WRAPPER — PARA USO DIRECTO DESDE s_main_merval.py
# ═══════════════════════════════════════════════════════════════

def compute_returns_robust(
    prices: pd.DataFrame,
    method: str = "log",
    max_gap_fill: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calcula retornos preservando NaN de inicio de serie.

    A diferencia de un simple prices.pct_change().dropna(), esta función:
      - Solo rellena huecos cortos (días sin operar) antes de calcular retornos.
      - Deja NaN donde el papel no tiene historia.
      - NO hace dropna() al final, para preservar la historia larga de los
        activos que sí tienen datos.

    Parameters
    ----------
    prices : pd.DataFrame
        Precios ya procesados con prepare_prices().
    method : str
        "log" (logarítmico), "simple" (aritmético), "delta" (diferencia).
    max_gap_fill : int
        Pasado a prepare_prices() para rellenar huecos cortos.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Retornos con NaN donde el activo no tiene historia.
    """
    # Paso 1: preparar precios (rellenar huecos cortos, dejar NaN de inicio)
    prices_clean = prepare_prices(prices, max_gap_fill=max_gap_fill, verbose=verbose)

    # Paso 2: calcular retornos
    if method == "log":
        returns = np.log(prices_clean / prices_clean.shift(1))
    elif method == "simple":
        returns = prices_clean.pct_change()
    elif method == "delta":
        returns = prices_clean.diff()
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usar 'log', 'simple' o 'delta'.")

    # Eliminar la primera fila (siempre es NaN por el shift)
    returns = returns.iloc[1:]

    if verbose:
        n_total = returns.shape[0] * returns.shape[1]
        n_nan = returns.isna().sum().sum()
        if n_nan > 0:
            print(f"  Retornos: {returns.shape[0]} fechas × {returns.shape[1]} activos, "
                  f"{n_nan} NaN ({n_nan/n_total*100:.1f}% del panel)")
        else:
            print(f"  Retornos: {returns.shape[0]} fechas × {returns.shape[1]} activos "
                  f"(panel completo, sin NaN)")

    return returns


def prepare_data_for_ep(
    returns: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara la matriz X y el vector p para Entropy Pooling,
    manejando correctamente el panel desbalanceado.

    Para EP necesitamos una matriz X (J, N) sin NaN y su vector de
    probabilidades p (J,). Cuando el panel es desbalanceado, usamos
    solo las filas donde TODOS los activos tienen dato.

    NOTA: La covarianza Σ para BL se estima aparte con estimate_covariance()
    usando toda la información pairwise. Este recorte es solo para EP
    que necesita escenarios completos.

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos con posibles NaN.
    verbose : bool

    Returns
    -------
    X : np.ndarray (J, N)
        Matriz de retornos sin NaN (filas completas).
    p : np.ndarray (J,)
        Prior uniforme.
    """
    # Filas donde todos los activos tienen dato
    mask_complete = returns.notna().all(axis=1)
    n_complete = mask_complete.sum()
    n_total = len(returns)

    if verbose and n_complete < n_total:
        n_dropped = n_total - n_complete
        print(f"  EP: usando {n_complete}/{n_total} filas completas "
              f"({n_dropped} filas con NaN descartadas para escenarios)")

    X = returns.loc[mask_complete].values
    J = X.shape[0]
    p = np.full(J, 1.0 / J)

    return X, p
