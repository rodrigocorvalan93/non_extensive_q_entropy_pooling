#%%
"""
s_main_merval.py
================
Aplicación empírica: BL vs EP-Shannon vs q-Tsallis-EP con el Merval (Panel Líder).

Período de análisis:
  - Estimación (prior): ~8 años de datos históricos hasta 22/sep/2025
  - Views incorporados: 22/sep/2025 (rebalanceo del Merval)
  - Out-of-sample: 22/sep/2025 → 20/mar/2026

View del PM de renta variable argentino:
  (inspirado en opiniones reales de septiembre 2025 - agradecimiento a Pablo J. Escapa por compartir su view)
  Ranking de overweight: YPFD > PAMP > TGSU2 > CEPU
  Underweight relativo: cada una de las energéticas top (YPFD, PAMP) rendirá más que cada banco (GGAL, BBAR, BMA, SUPV)
  (las demás con underweight implícito vía el ranking)

Modelos comparados:
  1. Benchmark (Merval ponderado)
  2. Black-Litterman
  3. Entropy Pooling (Shannon, q=1)
  4. q-Tsallis-EP (q=2, entropía de colisión)

Tipos de retornos soportados:
  - "log"    : logarítmicos ln(Pt/Pt-1) — mejor propiedades estadísticas
  - "simple" : simples (Pt-Pt-1)/Pt-1 — más intuitivo
  - "delta"  : diferencia Yt - Yt-1 — para yields de bonos (por si se decide aplicar a otro tipo de activos)

Requisitos:
  - entropy_pooling_v2.py, views_config.py, models.py (mismo directorio)
  - input_mkt_px.xlsx (precios históricos)
  - input_mkt_w.xlsx (weights del Merval)

Ref: Black & Litterman (1992), Meucci (2008), Tsallis (1988),
     Corvalán Salguero (2026)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from views_config import (
    ViewSpec, build_views,
    print_views_summary, print_bl_views, print_ep_views,
)
from models import (
    run_black_litterman, run_entropy_pooling, run_q_tsallis_ep,
    print_model_comparison, plot_model_comparison, ModelResult,
)

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════
DELTA = 2.5
TAU = 0.05
Q_TSALLIS = 2.0
CONFIDENCE = 0.5
MAX_WEIGHT = 0.30     # Máximo 30% por activo (límite de concentración común en FCIs argentinos)
RETURN_TYPE = "both"   # "log", "simple", "delta", "both"


# ═══════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════

def load_data(prices_path, weights_path):
    ext = Path(prices_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        prices = pd.read_excel(prices_path, index_col=0, parse_dates=True)
    else:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices.columns = [c.strip() for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    if Path(weights_path).suffix.lower() in (".xlsx", ".xls"):
        df_w = pd.read_excel(weights_path)
    else:
        df_w = pd.read_csv(weights_path)
    df_w["ticker"] = df_w["ticker"].str.strip()
    weights_dict = dict(zip(df_w["ticker"], df_w["weight"].astype(float)))

    common = sorted(set(prices.columns) & set(weights_dict.keys()))
    if not common:
        raise ValueError("No hay tickers en común entre precios y weights")
    prices = prices[common]
    tickers = list(common)
    w_mkt = np.array([weights_dict[t] for t in tickers])
    w_mkt = w_mkt / w_mkt.sum()

    null_count = prices.isnull().sum().sum()
    if null_count > 0:
        print(f"  Interpolando {null_count} valores nulos...")
        prices = prices.interpolate(method="linear").ffill().bfill()

    return prices, tickers, w_mkt


def compute_returns(prices, method="log"):
    """
    Calcula retornos.
      log   : ln(Pt/Pt-1)
      simple: (Pt-Pt-1)/Pt-1
      delta : Yt - Yt-1 (para yields de bonos)
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        return prices.pct_change().dropna()
    elif method == "delta":
        return prices.diff().dropna()
    else:
        raise ValueError(f"Método no reconocido: '{method}'. Usar 'log', 'simple' o 'delta'.")


# ═══════════════════════════════════════════════════════════════
# 3. VIEWS DEL PM
# ═══════════════════════════════════════════════════════════════

def define_pm_views():
    """
    Views del PM de renta variable argentino (septiembre 2025).

    View 1 — Ranking de overweight en energéticas:
        YPFD > PAMP > TGSU2 > CEPU
        Confianza: 70%.

    View 2 — Underweight en bancos respecto de las energéticas favoritas:
        Se expresa como views relativos: cada una de las energéticas top
        (YPFD, PAMP) rendirá ligeramente más que cada banco (GGAL, BBAR, BMA, SUPV).
        Spread: 0 (solo pide que les ganen, sin cuantificar cuánto).
        Confianza: 50% (opinión moderada, no es una apuesta fuerte contra bancos).

    Esto genera:
      - 3 constraints de desigualdad del ranking (YPFD≥PAMP≥TGSU2≥CEPU)
      - 8 constraints de desigualdad de underweight bancario:
            YPFD ≥ GGAL, YPFD ≥ BBAR, YPFD ≥ BMA, YPFD ≥ SUPV
            PAMP ≥ GGAL, PAMP ≥ BBAR, PAMP ≥ BMA, PAMP ≥ SUPV
    """
    BANCOS = ["GGAL", "BBAR", "BMA", "SUPV"]
    ENERGIA_TOP = ["YPFD", "PAMP"]

    views = [
        # ── View 1: ranking energéticas ──
        ViewSpec.ranking(
            tickers_ordered=["YPFD", "PAMP", "TGSU2", "CEPU"],
            confidence=0.70,
        ),
    ]

    # ── View 2: energéticas top le ganan a cada banco ──
    for energia in ENERGIA_TOP:
        for banco in BANCOS:
            views.append(
                ViewSpec.relative(
                    ticker_long=energia,
                    ticker_short=banco,
                    spread=0.0,  # solo pide que le gane, sin cuantificar
                    confidence=0.50,
                )
            )

    return views


# ═══════════════════════════════════════════════════════════════
# 4. PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(prices, tickers, w_mkt, views, return_method="log", max_weight=1.0):
    N = len(tickers)
    returns = compute_returns(prices, method=return_method)
    print(f"\n  Retornos ({return_method}): {returns.shape[0]} obs × {returns.shape[1]} activos")
    print(f"  Período: {returns.index[0].date()} → {returns.index[-1].date()}")

    X = returns.values
    J = X.shape[0]
    p = np.full(J, 1.0 / J)

    Sigma = np.cov(X.T, ddof=1)
    Sigma = 0.5 * (Sigma + Sigma.T)

    mu_prior = X.T @ p
    vol_prior = np.sqrt(np.diag(Sigma))
    print(f"\n  Estadísticas del prior (% diario):")
    print(f"  {'Ticker':<8s}  {'E[R]':>8s}  {'Vol':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*36}")
    for i, t in enumerate(tickers):
        sr = mu_prior[i] / vol_prior[i] * np.sqrt(252) if vol_prior[i] > 0 else 0
        print(f"  {t:<8s}  {mu_prior[i]*100:>8.4f}  {vol_prior[i]*100:>8.4f}  {sr:>8.2f}")

    print_views_summary(views)
    bl_views, ep_views = build_views(views, tickers, Sigma, X=X, p=p, tau=TAU)
    if bl_views:
        print_bl_views(bl_views, tickers)
    if ep_views:
        print_ep_views(ep_views)

    results = {}

    if bl_views:
        t0 = time.time()
        print("\n▶ Ejecutando Black-Litterman...")
        results["BL"] = run_black_litterman(
            Sigma, w_mkt, bl_views, tickers, delta=DELTA, tau=TAU, X=X, p=p,
            max_weight=max_weight)
        print(f"  ✓ BL completado ({time.time()-t0:.1f}s)")

    if ep_views:
        t0 = time.time()
        print("\n▶ Ejecutando Entropy Pooling (Shannon)...")
        results["EP"] = run_entropy_pooling(
            X, p, ep_views, tickers, w_mkt, delta=DELTA, confidence=CONFIDENCE,
            max_weight=max_weight)
        print(f"  ✓ EP-Shannon completado ({time.time()-t0:.1f}s)")

    if ep_views:
        t0 = time.time()
        print(f"\n▶ Ejecutando q-Tsallis-EP (q={Q_TSALLIS})...")
        results["qEP"] = run_q_tsallis_ep(
            X, p, ep_views, tickers, w_mkt,
            delta=DELTA, q=Q_TSALLIS, confidence=CONFIDENCE,
            max_weight=max_weight)
        print(f"  ✓ q-Tsallis-EP completado ({time.time()-t0:.1f}s)")

    print_model_comparison(list(results.values()), tickers, w_mkt)

    plot_path = HERE / f"model_comparison_{return_method}.png"
    plot_model_comparison(list(results.values()), tickers, w_mkt, save_path=str(plot_path))

    # ── Exportar tablas a Excel ──
    excel_path = HERE / f"resultados_{return_method}.xlsx"
    _export_results_to_excel(results, tickers, w_mkt, str(excel_path))

    return results


def _export_results_to_excel(
    results: Dict[str, ModelResult],
    tickers: List[str],
    w_mkt: np.ndarray,
    filepath: str,
) -> None:
    """
    Exporta los resultados de los modelos a un archivo Excel con múltiples hojas:
      - Pesos óptimos (%)
      - Retornos esperados posteriores (% diario)
      - Medidas de riesgo del portafolio
    """
    N = len(tickers)

    # ── Hoja 1: Pesos óptimos ──
    df_w = pd.DataFrame({"Ticker": tickers, "Benchmark (%)": np.round(w_mkt * 100, 4)})
    for name, r in results.items():
        # Limpiar ceros numéricos (artefactos del optimizador tipo 1e-16)
        weights_clean = r.w_optimal * 100
        weights_clean[weights_clean < 0.005] = 0.0
        df_w[f"{r.model_name} (%)"] = np.round(weights_clean, 4)
    # Agregar fila TOTAL
    total_row = {"Ticker": "TOTAL", "Benchmark (%)": round(w_mkt.sum() * 100, 4)}
    for name, r in results.items():
        total_row[f"{r.model_name} (%)"] = round(r.w_optimal.sum() * 100, 4)
    df_w = pd.concat([df_w, pd.DataFrame([total_row])], ignore_index=True)

    # ── Hoja 2: Retornos esperados ──
    df_mu = pd.DataFrame({"Ticker": tickers})
    for name, r in results.items():
        df_mu[f"{r.model_name} (% diario)"] = np.round(r.mu_posterior * 100, 6)
        df_mu[f"{r.model_name} (% anual)"] = np.round(r.mu_posterior * 252 * 100, 4)

    # ── Hoja 3: Medidas de riesgo ──
    metric_labels = {
        "expected_return_annual": "E[R] anual (%)",
        "volatility_annual": "Vol anual (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "VaR_95": "VaR 95% diario (%)",
        "CVaR_95": "CVaR 95% diario (%)",
        "VaR_99": "VaR 99% diario (%)",
        "CVaR_99": "CVaR 99% diario (%)",
    }
    risk_rows = []
    for key, label in metric_labels.items():
        row = {"Métrica": label}
        for name, r in results.items():
            val = r.risk_metrics.get(key, np.nan)
            mult = 100 if "ratio" not in key.lower() else 1
            row[r.model_name] = val * mult
        risk_rows.append(row)
    df_risk = pd.DataFrame(risk_rows)

    # ── Escribir Excel ──
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_w.to_excel(writer, sheet_name="Pesos óptimos", index=False)
        df_mu.to_excel(writer, sheet_name="Retornos esperados", index=False)
        df_risk.to_excel(writer, sheet_name="Medidas de riesgo", index=False)

    print(f"  Resultados exportados a: {filepath}")


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  APLICACIÓN EMPÍRICA: Merval Panel Líder")
    print("  BL vs EP-Shannon vs q-Tsallis-EP (q=2)")
    print("=" * 70)

    prices_path = HERE / "input_mkt_px.xlsx"
    weights_path = HERE / "input_mkt_w.xlsx"

    for fp, desc in [(prices_path, "precios"), (weights_path, "weights")]:
        if not fp.exists():
            print(f"\n⚠ No se encontró: {fp}")
            print(f"  Colocar '{fp.name}' en el mismo directorio del script.")
            sys.exit(1)

    prices, tickers, w_mkt = load_data(str(prices_path), str(weights_path))

    print(f"\nDatos cargados:")
    print(f"  Precios: {prices.shape[0]} fechas × {prices.shape[1]} activos")
    print(f"  Rango: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Activos ({len(tickers)}): {tickers}")
    print(f"\n  Weights del Merval (top 5):")
    for t, w in sorted(zip(tickers, w_mkt), key=lambda x: -x[1])[:5]:
        print(f"    {t}: {w:.4%}")

    views = define_pm_views()

    if RETURN_TYPE == "both":
        all_results = {}
        for method in ["log", "simple"]:
            print(f"\n{'#'*70}")
            print(f"  RETORNOS: {method.upper()}")
            print(f"{'#'*70}")
            all_results[method] = run_pipeline(
                prices, tickers, w_mkt, views, return_method=method,
                max_weight=MAX_WEIGHT)

        print(f"\n{'='*70}")
        print(f"  COMPARACIÓN LOG vs SIMPLE: pesos óptimos (%)")
        print(f"{'='*70}")
        print(f"  {'Ticker':<8s}  {'Bench':>7s}", end="")
        for method in ["log", "simple"]:
            for name in all_results[method]:
                print(f"  {name+'_'+method[:3]:>12s}", end="")
        print()
        print(f"  {'-'*75}")
        for i, t in enumerate(tickers):
            row = f"  {t:<8s}  {w_mkt[i]*100:>7.2f}"
            for method in ["log", "simple"]:
                for name, r in all_results[method].items():
                    row += f"  {r.w_optimal[i]*100:>12.2f}"
            print(row)
    else:
        results = run_pipeline(prices, tickers, w_mkt, views, return_method=RETURN_TYPE,
                               max_weight=MAX_WEIGHT)

    print("\n✓ Pipeline completado exitosamente.")
