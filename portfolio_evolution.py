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
  - input_current_mkt_px.xlsx: precios de los activos en el período OOS
  - Vectores de weights por modelo (hardcodeados o pasados como argumento)
  - Fecha inicio y fin del backtest

Outputs:
  - Gráfico de evolución comparativa (PNG)
  - Tabla de métricas (consola + Excel)

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


# ═══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════

def load_prices(filepath: str) -> pd.DataFrame:
    """Carga precios desde Excel o CSV. Limpia tickers y nulls."""
    ext = Path(filepath).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, index_col=0, parse_dates=True)
    else:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.isnull().sum().sum() > 0:
        df = df.interpolate(method="linear").ffill().bfill()
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
) -> pd.Series:
    """
    Calcula la evolución del valor de una cartera buy-and-hold.

    Se asume que los weights se aplican al cierre de date_start y la cartera
    se mantiene sin rebalanceo hasta date_end. La evolución refleja el
    cambio de precios de cada activo ponderado por su peso inicial.

    Parameters
    ----------
    prices : DataFrame de precios (DatetimeIndex, columnas = tickers)
    weights : dict {ticker: peso}, deben sumar ~1.0
    date_start : fecha de inicio (inclusive)
    date_end : fecha de fin (inclusive)
    initial_value : valor inicial de la cartera (default 100)

    Returns
    -------
    pd.Series con la evolución del valor de la cartera (DatetimeIndex)
    """
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

    # Normalizar weights a los tickers activos
    total_w = sum(weights.get(t, 0) for t in active_tickers)
    w_norm = {t: weights.get(t, 0) / total_w for t in active_tickers}

    # Retornos simples diarios
    returns = prices_oos[active_tickers].pct_change()

    # Retorno de la cartera = suma ponderada de retornos individuales
    portfolio_ret = pd.Series(0.0, index=returns.index)
    for t in active_tickers:
        portfolio_ret += w_norm[t] * returns[t]

    # Evolución acumulada
    portfolio_value = (1 + portfolio_ret).cumprod() * initial_value
    portfolio_value.iloc[0] = initial_value
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

    # Sortino (solo downside deviation)
    downside = rets[rets < rf_daily] - rf_daily
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else vol_daily
    sortino = (excess_daily / downside_std * np.sqrt(annualization)) if downside_std > 0 else 0

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

    Returns
    -------
    evolutions : dict {nombre: pd.Series} con las evoluciones
    metrics_df : DataFrame con métricas comparativas
    """
    evolutions = {}
    all_metrics = {}

    # Calcular evolución de cada cartera
    for name, weights in portfolios.items():
        ev = portfolio_evolution(prices, weights, date_start, date_end)
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
        m = compute_metrics(ev, benchmark_value=bench_for_te)
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
# 5. MAIN — Backtest Merval sep2025 → mar2026
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  BACKTEST OUT-OF-SAMPLE: Merval Panel Líder")
    print("  22/sep/2025 → 20/mar/2026")
    print("=" * 70)

    # ── Fechas ──
    DATE_START = "2025-09-22"
    DATE_END = "2026-03-20"

    # ── Tickers (orden alfabético, consistente con los modelos) ──
    TICKERS = [
        "ALUA", "BBAR", "BMA", "BYMA", "CEPU", "COME", "CRES", "EDN",
        "GGAL", "LOMA", "METR", "PAMP", "SUPV", "TECO2", "TGNO4",
        "TGSU2", "TRAN", "TXAR", "VALO", "YPFD",
    ]

    # ── Pesos por modelo (resultados de s_main_merval.py con retornos simples) ──
    W_BENCHMARK = dict(zip(TICKERS, [
        2.6768, 5.2048, 8.0094, 4.7222, 6.2323, 1.3602, 2.6799, 1.2280,
        15.0428, 3.4749, 0.8689, 14.6787, 2.7442, 2.1948, 1.1663,
        5.6422, 1.2265, 5.0218, 1.2791, 14.5463,
    ]))

    W_BL = dict(zip(TICKERS, [
        2.9216, 0, 0, 4.4471, 3.4694, 1.2274, 2.7282, 1.0369,
        7.8439, 3.4029, 0.2774, 27.0455, 0, 2.2051, 1.1387,
        10.6769, 1.2823, 4.9934, 1.3876, 23.9158,
    ]))

    W_EP = dict(zip(TICKERS, [
        15.7675, 0, 0, 22.5765, 0, 0, 2.1515, 0,
        0, 0, 0, 6.2464, 0, 0, 0,
        10.0757, 0, 4.0551, 21.7884, 17.3388,
    ]))

    W_QEP = dict(zip(TICKERS, [
        15.7824, 0, 0, 22.5607, 0, 0, 2.1393, 0,
        0, 0, 0, 6.0501, 0, 0, 0,
        10.3137, 0, 4.0663, 21.7640, 17.3234,
    ]))

    # Normalizar (los weights vienen en %, convertir a fracción)
    for d in [W_BENCHMARK, W_BL, W_EP, W_QEP]:
        total = sum(d.values())
        for k in d:
            d[k] = d[k] / total

    PORTFOLIOS = {
        "Benchmark": W_BENCHMARK,
        "Black-Litterman": W_BL,
        "EP-Shannon": W_EP,
        "q-Tsallis-EP (q=2.0)": W_QEP,
    }

    # ── Cargar precios ──
    px_path = HERE / "input_current_mkt_px.xlsx"
    if not px_path.exists():
        print(f"\n⚠ No se encontró: {px_path}")
        sys.exit(1)

    prices = load_prices(str(px_path))
    print(f"\nPrecios cargados: {prices.shape[0]} fechas × {prices.shape[1]} activos")
    print(f"  Rango disponible: {prices.index[0].date()} → {prices.index[-1].date()}")

    # Verificar que el rango OOS está cubierto
    d0, d1 = pd.Timestamp(DATE_START), pd.Timestamp(DATE_END)
    mask = (prices.index >= d0) & (prices.index <= d1)
    print(f"  Fechas en rango OOS [{DATE_START} → {DATE_END}]: {mask.sum()}")
    if mask.sum() < 2:
        print("  ⚠ Datos insuficientes para el período OOS")
        sys.exit(1)

    # ── Comparar carteras ──
    # benchmark_index_ticker: si tuvieras el índice Merval como columna,
    # podrías poner benchmark_index_ticker="MERVAL" acá.
    evolutions, metrics_df = compare_portfolios(
        prices, PORTFOLIOS, DATE_START, DATE_END,
        benchmark_name="Benchmark",
        benchmark_index_ticker=None,  # None = usa la cartera Benchmark
    )

    # ── Imprimir resultados ──
    print(f"\n{'='*90}")
    print(f"  MÉTRICAS OUT-OF-SAMPLE: {DATE_START} → {DATE_END}")
    print(f"{'='*90}")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(metrics_df.to_string())
    print(f"{'='*90}")

    # ── Gráfico ──
    plot_path = HERE / "portfolio_evolution_oos.png"
    plot_comparison(
        evolutions, metrics_df,
        title=f"Evolución out-of-sample: {DATE_START} → {DATE_END} (base 100)",
        save_path=str(plot_path),
    )

    # ── Exportar a Excel ──
    excel_path = HERE / "resultados_oos.xlsx"
    with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Métricas OOS")
        # Evoluciones como serie
        ev_df = pd.DataFrame(evolutions)
        ev_df.to_excel(writer, sheet_name="Evolución diaria")
    print(f"  Resultados exportados a: {excel_path}")

    print("\n✓ Backtest completado.")
