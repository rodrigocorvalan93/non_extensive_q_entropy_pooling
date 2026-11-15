#%%
"""
portfolio_evolution.py
======================
Evaluación out-of-sample de carteras: evolución, métricas y comparación.

Dado un vector de weights y un rango de fechas, calcula:
  - Evolución del valor de la cartera (base 100)
  - Retorno acumulado
  - Volatilidad anualizada
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown y su duración
  - Tracking Error vs benchmark

Admite múltiples carteras para comparar (ej: Benchmark vs BL vs EP vs q-EP).

El benchmark puede ser:
  (a) Una cartera definida por weights (ej: weights del Merval)
  (b) Una serie índice incluida en el archivo de precios (ej: columna "MERVAL")

Inputs:
  - data/dataacciones/input_current_mkt_px.xlsx: precios en el período OOS
  - results/merval_pesos_simple.csv: pesos de todas las carteras, generado
    por s_main_merval.py (no hay pesos copiados a mano)
  - Fecha inicio y fin del backtest

Convenciones:
  - Buy-and-hold (por defecto): se compra al cierre de la fecha inicial y no
    se vuelve a operar; los pesos derivan con los precios.
  - Rebalanceo diario (rebalance="daily"): pesos constantes, se rebalancea
    todos los días sin costo. Se informa como robustez.
  - Sharpe y Sortino con tasa libre de riesgo cero: son cocientes sobre
    retornos nominales en pesos, comparables entre carteras pero no en nivel.

Outputs (carpeta results/ y paper/figs/):
  - Tabla de métricas (consola, CSV y Excel) para las dos convenciones
  - Intervalos de confianza bootstrap para diferencias de Sharpe
  - Gráfico de evolución comparativa (PNG), en español y en inglés

Ref: Bacon (2008), "Practical Portfolio Performance Measurement and Attribution"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "dataacciones"
RESULTS_DIR = HERE / "results"
FIGS_DIR = HERE / "paper" / "figs"


# ═══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════

def load_prices(filepath: str) -> pd.DataFrame:
    """
    Carga precios desde Excel o CSV y limpia los nombres de los tickers.

    No rellena faltantes: interpolar precios en el período de evaluación
    usaría información futura. portfolio_evolution() falla si encuentra un
    precio faltante dentro de la ventana.
    """
    ext = Path(filepath).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ═══════════════════════════════════════════════════════════════
# 2. EVOLUCIÓN DE CARTERA
# ═══════════════════════════════════════════════════════════════

def portfolio_evolution(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    date_start: str,
    date_end: str,
    initial_value: float = 100.0,
    rebalance: str = "none",
) -> pd.Series:
    """
    Calcula la evolución del valor de una cartera.

    Los pesos se aplican al cierre de date_start.

    rebalance="none"  (por defecto) : buy-and-hold. Se compran las cantidades
        que dan los pesos iniciales y no se opera más; el valor es
        V_t = V_0 · Σ_i w_i · P_{i,t} / P_{i,0} y los pesos derivan.
    rebalance="daily" : pesos constantes. La cartera vuelve a los pesos
        iniciales todos los días (sin costos); el retorno diario es
        Σ_i w_i · r_{i,t}.

    Parameters
    ----------
    prices : DataFrame de precios (DatetimeIndex, columnas = tickers)
    weights : dict {ticker: peso}, deben sumar ~1.0
    date_start : fecha de inicio (inclusive)
    date_end : fecha de fin (inclusive)
    initial_value : valor inicial de la cartera (default 100)
    rebalance : "none" o "daily"

    Returns
    -------
    pd.Series con la evolución del valor de la cartera (DatetimeIndex)
    """
    if rebalance not in ("none", "daily"):
        raise ValueError("rebalance debe ser 'none' o 'daily'")
    d0 = pd.Timestamp(date_start)
    d1 = pd.Timestamp(date_end)

    # Validar rango de fechas
    available = prices.index
    if d0 < available[0]:
        raise ValueError(f"Fecha inicio {d0.date()} anterior al primer dato ({available[0].date()})")
    if d1 > available[-1]:
        raise ValueError(f"Fecha fin {d1.date()} posterior al último dato ({available[-1].date()})")

    # Filtrar rango
    mask = (available >= d0) & (available <= d1)
    if mask.sum() < 2:
        raise ValueError(f"Menos de 2 fechas en el rango [{d0.date()}, {d1.date()}]")

    prices_oos = prices.loc[mask].copy()

    # Filtrar tickers con peso > 0
    active_tickers = [t for t, w in weights.items() if abs(w) > 1e-10 and t in prices_oos.columns]
    if not active_tickers:
        raise ValueError("Ningún ticker con peso > 0 encontrado en los precios")
    if prices_oos[active_tickers].isnull().any().any():
        raise ValueError("Hay precios faltantes en el período de evaluación")

    # Normalizar weights a los tickers activos
    total_w = sum(weights.get(t, 0) for t in active_tickers)
    w_norm = pd.Series({t: weights.get(t, 0) / total_w for t in active_tickers})

    if rebalance == "none":
        relativos = prices_oos[active_tickers] / prices_oos[active_tickers].iloc[0]
        portfolio_value = (relativos * w_norm).sum(axis=1) * initial_value
    else:
        returns = prices_oos[active_tickers] / prices_oos[active_tickers].shift(1) - 1.0
        portfolio_ret = (returns * w_norm).sum(axis=1)
        portfolio_ret.iloc[0] = 0.0
        portfolio_value = (1 + portfolio_ret).cumprod() * initial_value
    portfolio_value.name = "Portfolio"

    return portfolio_value


# ═══════════════════════════════════════════════════════════════
# 3. MÉTRICAS DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    portfolio_value: pd.Series,
    benchmark_value: Optional[pd.Series] = None,
    annualization: float = 252.0,
    rf_annual: float = 0.0,
) -> Dict[str, float]:
    """
    Calcula métricas de performance sobre la serie de valor de la cartera.

    Parameters
    ----------
    portfolio_value : Serie con la evolución del valor (base 100)
    benchmark_value : Serie del benchmark (mismas fechas), opcional
    annualization : factor de anualización (252 para datos diarios)
    rf_annual : tasa libre de riesgo anualizada (default 0)

    Returns
    -------
    dict con métricas
    """
    # Retornos diarios
    rets = portfolio_value.pct_change().dropna()
    n_days = len(rets)
    rf_daily = (1 + rf_annual) ** (1 / annualization) - 1

    # Retorno acumulado
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1

    # Retorno anualizado
    n_years = n_days / annualization
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatilidad
    vol_daily = rets.std()
    vol_annual = vol_daily * np.sqrt(annualization)

    # Sharpe
    excess_daily = rets.mean() - rf_daily
    sharpe = (excess_daily / vol_daily * np.sqrt(annualization)) if vol_daily > 0 else 0

    # Sortino: desvío a la baja sobre TODOS los días (los días por encima
    # del umbral aportan cero), que es la definición usual.
    downside = np.minimum(rets - rf_daily, 0.0)
    downside_std = np.sqrt((downside ** 2).mean())
    sortino = (excess_daily / downside_std * np.sqrt(annualization)) if downside_std > 0 else np.nan

    # Max Drawdown
    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_dd = drawdown.min()
    # Duración del max drawdown (días desde el pico hasta el punto más bajo)
    if max_dd < 0:
        dd_end_idx = drawdown.idxmin()
        # Buscar el pico anterior
        peak_before = portfolio_value.loc[:dd_end_idx].idxmax()
        dd_duration = (dd_end_idx - peak_before).days
        # Buscar recuperación (si la hay)
        post_dd = portfolio_value.loc[dd_end_idx:]
        recovery_mask = post_dd >= portfolio_value.loc[peak_before]
        if recovery_mask.any():
            recovery_date = post_dd[recovery_mask].index[0]
            dd_recovery_days = (recovery_date - dd_end_idx).days
        else:
            dd_recovery_days = (portfolio_value.index[-1] - dd_end_idx).days
    else:
        dd_duration = 0
        dd_recovery_days = 0

    metrics = {
        "Retorno acumulado (%)": total_return * 100,
        "Retorno anualizado (%)": annual_return * 100,
        "Volatilidad anualizada (%)": vol_annual * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown (%)": max_dd * 100,
        "Drawdown duración (días)": dd_duration,
        "Drawdown recuperación (días)": dd_recovery_days,
        "Días en el período": n_days,
    }

    # Tracking Error vs benchmark
    if benchmark_value is not None:
        bench_rets = benchmark_value.pct_change().dropna()
        # Alinear fechas
        common_idx = rets.index.intersection(bench_rets.index)
        if len(common_idx) > 1:
            excess = rets.loc[common_idx] - bench_rets.loc[common_idx]
            te = excess.std() * np.sqrt(annualization)
            info_ratio = (excess.mean() * annualization) / te if te > 0 else 0
            metrics["Tracking Error anualizado (%)"] = te * 100
            metrics["Information Ratio"] = info_ratio
            metrics["Exceso retorno anual (%)"] = excess.mean() * annualization * 100

            # Beta y Alfa (regresión OLS: R_p = α + β·R_bench + ε)
            r_p = rets.loc[common_idx].values
            r_b = bench_rets.loc[common_idx].values
            cov_pb = np.cov(r_p, r_b)          # misma convención (ddof=1) arriba y abajo
            beta = cov_pb[0, 1] / cov_pb[1, 1] if cov_pb[1, 1] > 0 else 1.0
            alpha_daily = r_p.mean() - beta * r_b.mean()
            alpha_annual = alpha_daily * annualization
            metrics["Beta"] = beta
            metrics["Alfa anualizado (%)"] = alpha_annual * 100

    return metrics


# ═══════════════════════════════════════════════════════════════
# 4. COMPARACIÓN DE MÚLTIPLES CARTERAS
# ═══════════════════════════════════════════════════════════════

def compare_portfolios(
    prices: pd.DataFrame,
    portfolios: Dict[str, Dict[str, float]],
    date_start: str,
    date_end: str,
    benchmark_name: str = "Benchmark",
    benchmark_index_ticker: Optional[str] = None,
    rebalance: str = "none",
    rf_annual: float = 0.0,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    Compara múltiples carteras en el mismo período.

    Parameters
    ----------
    prices : DataFrame de precios
    portfolios : dict {nombre_cartera: {ticker: peso}}
    date_start, date_end : rango del backtest
    benchmark_name : nombre de la cartera benchmark (para tracking error)
    benchmark_index_ticker : si se provee, usa esta columna de prices como
                             serie índice de benchmark en vez de la cartera
    rebalance : "none" (buy-and-hold) o "daily" (pesos constantes)

    Returns
    -------
    evolutions : dict {nombre: pd.Series} con las evoluciones
    metrics_df : DataFrame con métricas comparativas
    """
    evolutions = {}
    all_metrics = {}

    # Calcular evolución de cada cartera
    for name, weights in portfolios.items():
        ev = portfolio_evolution(prices, weights, date_start, date_end, rebalance=rebalance)
        evolutions[name] = ev

    # Benchmark para tracking error
    if benchmark_index_ticker and benchmark_index_ticker in prices.columns:
        d0, d1 = pd.Timestamp(date_start), pd.Timestamp(date_end)
        mask = (prices.index >= d0) & (prices.index <= d1)
        bench_prices = prices.loc[mask, benchmark_index_ticker]
        bench_ev = bench_prices / bench_prices.iloc[0] * 100
        bench_ev.name = benchmark_index_ticker
    elif benchmark_name in evolutions:
        bench_ev = evolutions[benchmark_name]
    else:
        bench_ev = None

    # Calcular métricas
    for name, ev in evolutions.items():
        bench_for_te = bench_ev if name != benchmark_name else None
        m = compute_metrics(ev, benchmark_value=bench_for_te, rf_annual=rf_annual)
        all_metrics[name] = m

    # Armar DataFrame
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "Cartera"

    return evolutions, metrics_df


def plot_comparison(
    evolutions: Dict[str, pd.Series],
    metrics_df: pd.DataFrame,
    title: str = "Evolución de carteras (base 100)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Genera gráfico de evolución comparativa y tabla de métricas.

    Panel superior: evolución del valor (base 100)
    Panel inferior: tabla de métricas principales
    """
    colors = {
        "Benchmark": "#888888",
        "Black-Litterman": "#2E86AB",
        "EP-Shannon": "#E8475A",
        "q-Tsallis-EP (q=2.0)": "#F5A623",
    }
    default_colors = ["#7B68EE", "#2ECC71", "#FF6B6B", "#4ECDC4"]

    fig = plt.figure(figsize=(14, 10))

    # ── Panel superior: evolución ──
    ax1 = fig.add_axes([0.08, 0.42, 0.88, 0.52])

    for i, (name, ev) in enumerate(evolutions.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        lw = 2.5 if name == "Benchmark" else 1.8
        ls = "--" if name == "Benchmark" else "-"
        alpha = 0.6 if name == "Benchmark" else 0.9
        ax1.plot(ev.index, ev.values, color=color, linewidth=lw,
                 linestyle=ls, alpha=alpha, label=name)

    ax1.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
    ax1.set_ylabel("Valor de la cartera (base 100)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # ── Panel inferior: tabla de métricas ──
    # Seleccionar métricas principales para la tabla visual
    display_metrics = [
        "Retorno acumulado (%)",
        "Retorno anualizado (%)",
        "Volatilidad anualizada (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Drawdown duración (días)",
        "Tracking Error anualizado (%)",
        "Information Ratio",
        "Beta",
        "Alfa anualizado (%)",
    ]

    available = [m for m in display_metrics if m in metrics_df.columns]
    table_data = metrics_df[available].T

    ax2 = fig.add_axes([0.08, 0.02, 0.88, 0.32])
    ax2.axis("off")

    cell_text = []
    for metric in available:
        row = []
        for portfolio in table_data.columns:
            val = table_data.loc[metric, portfolio]
            if pd.isna(val):
                row.append("—")
            elif "Ratio" in metric:
                row.append(f"{val:.3f}")
            elif "días" in metric:
                row.append(f"{int(val)}")
            else:
                row.append(f"{val:.2f}")
        cell_text.append(row)

    table = ax2.table(
        cellText=cell_text,
        rowLabels=available,
        colLabels=list(table_data.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Colorear header
    for j, col in enumerate(table_data.columns):
        color = colors.get(col, "#DDDDDD")
        table[0, j].set_facecolor(color)
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.subplots_adjust(hspace=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════
# 5. SIGNIFICANCIA: bootstrap en bloques para diferencias de Sharpe
# ═══════════════════════════════════════════════════════════════

def sharpe_ratio(rets: np.ndarray, annualization: float = 252.0) -> float:
    """Sharpe anualizado con tasa libre de riesgo cero."""
    sd = np.std(rets, ddof=1)
    return float(np.mean(rets) / sd * np.sqrt(annualization)) if sd > 0 else np.nan


def bootstrap_sharpe_diff(
    rets_a: np.ndarray,
    rets_b: np.ndarray,
    block: int = 5,
    n_boot: int = 10_000,
    seed: int = 12345,
    annualization: float = 252.0,
) -> Dict[str, float]:
    """
    Intervalo de confianza para Sharpe(a) − Sharpe(b) por bootstrap circular
    en bloques (Politis y Romano, 1992), remuestreando los dos retornos con
    los mismos índices para conservar su correlación.

    Devuelve la diferencia observada, el intervalo percentil de 95% y la
    fracción de réplicas con diferencia <= 0 (un p-valor unilateral aproximado).
    """
    a = np.asarray(rets_a, dtype=float)
    b = np.asarray(rets_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Las dos series deben tener el mismo largo")
    n = a.size
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    offsets = np.arange(block)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        diffs[i] = sharpe_ratio(a[idx], annualization) - sharpe_ratio(b[idx], annualization)
    obs = sharpe_ratio(a, annualization) - sharpe_ratio(b, annualization)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_le_0": float(np.mean(diffs <= 0.0))}


def plot_paper_figure(evolutions: Dict[str, pd.Series], labels: Dict[str, str],
                      ylabel: str, save_path: Path) -> None:
    """Figura limpia para el paper: solo la evolución, sin título ni tabla."""
    estilos = {
        "Benchmark": dict(color="#555555", ls="--", lw=2.2),
        "BL":        dict(color="#2E86AB", ls="-",  lw=1.8),
        "EP-eq":     dict(color="#2E86AB", ls=":",  lw=2.2),
        "MV-hist":   dict(color="#E8475A", ls="--", lw=1.8),
        "EP":        dict(color="#E8475A", ls="-",  lw=1.8),
    }
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, st in estilos.items():
        if name in evolutions:
            ev = evolutions[name]
            ax.plot(ev.index, ev.values, label=labels.get(name, name), **st)
    ax.axhline(100, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 6. MAIN — Backtest Merval sep2025 → mar2026
# ═══════════════════════════════════════════════════════════════

DATE_START = "2025-09-22"
DATE_END = "2026-03-20"

# Carteras que se evalúan y nombre con el que aparecen en las tablas.
# Las columnas salen de results/merval_pesos_simple.csv (s_main_merval.py).
PORTFOLIO_ORDER = [
    "Benchmark", "BL", "EP-eq", "qEP-eq",          # prior de equilibrio
    "MV-hist", "BL-hist", "EP", "qEP",             # prior histórico
    "1/N", "MinVar",                               # referencias
]


def run_backtest(weights_csv: Path, prices_path: Path,
                 date_start: str = DATE_START, date_end: str = DATE_END,
                 rebalance: str = "none"):
    """Evalúa todas las carteras del CSV de pesos en la ventana indicada."""
    df_w = pd.read_csv(weights_csv, index_col=0)
    cols = [c for c in PORTFOLIO_ORDER if c in df_w.columns and df_w[c].notna().all()]
    portfolios = {c: df_w[c].to_dict() for c in cols}
    prices = load_prices(str(prices_path))
    evolutions, metrics_df = compare_portfolios(
        prices, portfolios, date_start, date_end,
        benchmark_name="Benchmark", benchmark_index_ticker=None, rebalance=rebalance)
    return df_w[cols], evolutions, metrics_df


if __name__ == "__main__":
    print("=" * 70)
    print("  BACKTEST OUT-OF-SAMPLE: Merval Panel Líder")
    print(f"  {DATE_START} → {DATE_END}")
    print("=" * 70)

    weights_csv = RESULTS_DIR / "merval_pesos_simple.csv"
    px_path = DATA_DIR / "input_current_mkt_px.xlsx"
    for fp in (weights_csv, px_path):
        if not fp.exists():
            print(f"\n⚠ No se encontró: {fp}")
            if fp == weights_csv:
                print("  Correr primero: python s_main_merval.py")
            sys.exit(1)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 250)
    resultados = {}
    for rebalance, etiqueta in (("none", "buy-and-hold"), ("daily", "rebalanceo diario")):
        df_w, evolutions, metrics_df = run_backtest(weights_csv, px_path, rebalance=rebalance)
        resultados[rebalance] = (evolutions, metrics_df)
        print(f"\n{'='*100}")
        print(f"  MÉTRICAS OUT-OF-SAMPLE ({etiqueta}): {DATE_START} → {DATE_END}")
        print(f"{'='*100}")
        print(metrics_df.to_string())

    evolutions, metrics_df = resultados["none"]
    n_days = int(metrics_df["Días en el período"].iloc[0])

    # ── Costo de armar cada cartera partiendo del benchmark ──
    # Volumen operado = ||w − w_bench||₁ (compras más ventas); a c puntos
    # básicos por lado el costo único es c · ||w − w_bench||₁.
    turnover = {c: float(np.abs(df_w[c] - df_w["Benchmark"]).sum()) for c in df_w.columns}
    print("\n  Volumen operado para pasar del benchmark a cada cartera (||Δw||₁) y costo a 50 pb por lado:")
    for c, t in turnover.items():
        print(f"    {c:<10s} {t:6.3f}   costo = {t * 0.5:5.2f}%")

    # ── Contribución de cada papel al retorno (buy-and-hold): w_i · (P_T/P_0 − 1) ──
    prices = load_prices(str(px_path))
    win = prices.loc[(prices.index >= pd.Timestamp(DATE_START)) & (prices.index <= pd.Timestamp(DATE_END))]
    ret_papel = (win.iloc[-1] / win.iloc[0] - 1.0).reindex(df_w.index)
    contrib = df_w.mul(ret_papel, axis=0) * 100
    contrib.insert(0, "retorno_papel_%", ret_papel * 100)
    contrib.to_csv(RESULTS_DIR / "oos_contribuciones.csv", float_format="%.6f")
    if {"MV-hist", "Benchmark"} <= set(df_w.columns):
        dif = (contrib["MV-hist"] - contrib["Benchmark"]).sort_values(ascending=False)
        print("\n  Contribución por papel a la diferencia (sin views, prior histórico) − benchmark, en puntos:")
        print("    " + ", ".join(f"{t} {v:+.1f}" for t, v in dif.head(4).items())
              + f" … total {dif.sum():+.1f}")

    # ── Significancia: diferencias de Sharpe con bootstrap en bloques ──
    rets = {k: v.pct_change().dropna().values for k, v in evolutions.items()}
    pares = [("BL", "Benchmark"), ("EP-eq", "Benchmark"), ("qEP-eq", "Benchmark"),
             ("BL-hist", "MV-hist"), ("EP", "MV-hist"), ("qEP", "MV-hist"),
             ("MV-hist", "Benchmark"), ("EP", "BL"), ("EP-eq", "BL"), ("qEP", "EP")]
    filas = []
    print(f"\n  Diferencias de Sharpe (buy-and-hold), IC 95% por bootstrap en bloques de 5 días, {n_days} días:")
    for a, b in pares:
        if a in rets and b in rets:
            r = bootstrap_sharpe_diff(rets[a], rets[b])
            filas.append({"a": a, "b": b, **r})
            print(f"    Sharpe({a}) − Sharpe({b}) = {r['diff']:+.2f}   IC95% [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
                  f"   P(dif ≤ 0) = {r['p_le_0']:.3f}")
    boot_df = pd.DataFrame(filas)

    # ── Sensibilidad del intervalo de la diferencia entre los dos priors ──
    # El extremo inferior del intervalo de Sharpe(sin views, histórico) −
    # Sharpe(benchmark) queda cerca de cero, así que su signo depende del largo
    # del bloque y de la semilla del remuestreo. Se informa el rango.
    sens = []
    if "MV-hist" in rets and "Benchmark" in rets:
        for block in (1, 3, 5, 10, 20):
            for seed in range(1000, 1010):
                r = bootstrap_sharpe_diff(rets["MV-hist"], rets["Benchmark"], block=block, seed=seed)
                sens.append({"bloque": block, "semilla": seed, "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    sens_df = pd.DataFrame(sens)
    if len(sens_df):
        res_bloque = sens_df.groupby("bloque")["ci_lo"].agg(["min", "max"])
        print("\n  Extremo inferior del IC 95% de Sharpe(sin views, histórico) − Sharpe(benchmark),"
              " 10 semillas por largo de bloque:")
        for b, fila in res_bloque.iterrows():
            print(f"    bloque {b:>2d}: de {fila['min']:+.2f} a {fila['max']:+.2f}")

    # ── Peso de la rueda de mayor movimiento del benchmark ──
    r_bench = evolutions["Benchmark"].pct_change().dropna()
    dia = r_bench.abs().idxmax()
    filas_dia = []
    for k, v in evolutions.items():
        r = v.pct_change().dropna()
        filas_dia.append({"cartera": k, "fecha": pd.Timestamp(dia).date().isoformat(),
                          "retorno_dia_pct": float(r.loc[dia]) * 100,
                          "sharpe": sharpe_ratio(r.values),
                          "sharpe_sin_dia": sharpe_ratio(r.drop(dia).values)})
    dia_df = pd.DataFrame(filas_dia).set_index("cartera")
    print(f"\n  Rueda de mayor movimiento del benchmark: {pd.Timestamp(dia).date()} "
          f"({dia_df.loc['Benchmark', 'retorno_dia_pct']:+.1f}%). Sharpe sin esa rueda:")
    print("    " + ", ".join(f"{k} {v:.2f}" for k, v in dia_df["sharpe_sin_dia"].items()))

    # ── Exportar ──
    RESULTS_DIR.mkdir(exist_ok=True)
    for rebalance, sufijo in (("none", "buyhold"), ("daily", "rebalanceo_diario")):
        resultados[rebalance][1].to_csv(RESULTS_DIR / f"oos_metricas_{sufijo}.csv", float_format="%.6f")
    boot_df.to_csv(RESULTS_DIR / "oos_bootstrap_sharpe.csv", index=False, float_format="%.6f")
    sens_df.to_csv(RESULTS_DIR / "oos_bootstrap_sensibilidad.csv", index=False, float_format="%.6f")
    dia_df.to_csv(RESULTS_DIR / "oos_dia_extremo.csv", float_format="%.6f")
    pd.Series(turnover, name="l1_vs_benchmark").to_csv(RESULTS_DIR / "oos_turnover.csv", float_format="%.6f")
    excel_path = RESULTS_DIR / "resultados_oos.xlsx"
    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
        resultados["none"][1].to_excel(writer, sheet_name="Métricas buy-and-hold")
        resultados["daily"][1].to_excel(writer, sheet_name="Métricas rebalanceo diario")
        pd.DataFrame(evolutions).to_excel(writer, sheet_name="Evolución buy-and-hold")
        boot_df.to_excel(writer, sheet_name="Bootstrap Sharpe", index=False)
    print(f"\n  Resultados exportados a: {RESULTS_DIR}")

    # ── Gráficos ──
    plot_comparison(
        {k: v for k, v in evolutions.items()}, metrics_df,
        title=f"Evolución out-of-sample: {DATE_START} → {DATE_END} (base 100, buy-and-hold)",
        save_path=str(RESULTS_DIR / "portfolio_evolution_oos.png"),
    )
    plt.close("all")
    textos = {
        "es": ({"Benchmark": "Benchmark", "BL": "Black–Litterman (prior de equilibrio)",
                "EP-eq": "EP (prior de equilibrio)", "MV-hist": "Sin views (prior histórico)",
                "EP": "EP (prior histórico)"}, "Valor de la cartera (base 100)"),
        "en": ({"Benchmark": "Benchmark", "BL": "Black–Litterman (equilibrium prior)",
                "EP-eq": "EP (equilibrium prior)", "MV-hist": "No views (historical prior)",
                "EP": "EP (historical prior)"}, "Portfolio value (base 100)"),
    }
    for lang, (labels, ylabel) in textos.items():
        plot_paper_figure(evolutions, labels, ylabel, FIGS_DIR / f"oos_evolution_{lang}.png")

    print("\n✓ Backtest completado.")
