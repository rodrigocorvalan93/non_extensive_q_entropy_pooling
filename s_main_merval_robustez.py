#%%
"""
s_main_merval_robustez.py
=========================
Robustez de la aplicación al Merval (Sección 6.5 y Apéndice C del paper).

Repite el diseño {prior de equilibrio, prior histórico} × {sin views, BL, EP,
q-EP} de s_main_merval.py cambiando de a un supuesto por vez, y evalúa cada
cartera fuera de muestra con portfolio_evolution.py (buy-and-hold).

Variantes
---------
  base                 retornos simples, views como desigualdades, confianza
                       global 0,5, q = 2, tope 30%, δ = 2,5
  igualdades mezcladas codificación original de los views energética-vs-banco
                       (solo es factible con el prior histórico)
  confianza plena      c = 1 (sin mezcla con el prior)
  q = 1,5 / q = 3      otros miembros de la familia
  tope 20% / 40% / sin tope
  δ = 1 / δ = 5
  retornos log
  ventana de 5 / 3 años   escenarios de los últimos 5 o 3 años en lugar de toda
                       la muestra
  views invertidos     placebo: los views CONTRARIOS a los del PM
  rebalanceo diario    misma cartera base, otra convención de evaluación
  tasa libre 30%       misma cartera base, Sharpe con tasa libre de riesgo de
                       30% anual en lugar de cero

Para cada variante se informa:
  - el Sharpe fuera de muestra del control sin views y de cada modelo, en
    cada prior (el del benchmark no cambia entre variantes);
  - ||w − w_control||₁, cuánto mueven los views la cartera;
  - ||w_EP − w_qEP||₁, cuánto importa q.

Salida: results/merval_robustez.csv
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import s_main_merval as M
import portfolio_evolution as PE
from covariance_estimation import compute_returns_robust, estimate_covariance, prepare_data_for_ep

RESULTS_DIR = HERE / "results"

VARIANTES = [
    # nombre,                 kwargs
    ("base",                 {}),
    ("igualdades mezcladas", {"ep_relative": "blended_equality"}),
    ("confianza plena",      {"confidence": 1.0}),
    ("q = 1,5",              {"q": 1.5}),
    ("q = 3",                {"q": 3.0}),
    ("tope 20%",             {"max_weight": 0.20}),
    ("tope 40%",             {"max_weight": 0.40}),
    ("sin tope",             {"max_weight": 1.00}),
    ("delta = 1",            {"delta": 1.0}),
    ("delta = 5",            {"delta": 5.0}),
    ("retornos log",         {"return_method": "log"}),
    ("ventana 5 anios",      {"anios": 5}),
    ("ventana 3 anios",      {"anios": 3}),
    ("views invertidos",     {"reverse": True}),
    ("rebalanceo diario",    {"rebalance": "daily"}),
    ("tasa libre 30%",       {"rf_annual": 0.30}),
]


def pesos_variante(prices, tickers, w_mkt, *, return_method="simple",
                   ep_relative="inequality", reverse=False, anios=None, **kw):
    """Pesos del diseño completo para una variante (sin imprimir nada)."""
    if anios is not None:
        desde = prices.index[-1] - pd.DateOffset(years=anios)
        prices = prices.loc[prices.index >= desde]
    with contextlib.redirect_stdout(io.StringIO()):
        returns = compute_returns_robust(prices, method=return_method,
                                         max_gap_fill=M.MAX_GAP_FILL, verbose=False)
        Sigma = estimate_covariance(returns, min_obs_pairwise=M.MIN_OBS_PAIRWISE, verbose=False)
        Sigma = 0.5 * (Sigma + Sigma.T)
        X, p = prepare_data_for_ep(returns, verbose=False)
        views_bl, views_ep = M.define_pm_views(ep_relative=ep_relative, reverse=reverse)
        df_w = M.compute_design(X, p, Sigma, tickers, w_mkt, views_bl, views_ep, **kw)
    return df_w


def evaluar(df_w, prices_oos, rebalance="none", rf_annual=0.0):
    """Sharpe y retorno acumulado fuera de muestra de cada cartera con pesos válidos."""
    cols = [c for c in df_w.columns if df_w[c].notna().all()]
    portfolios = {c: df_w[c].to_dict() for c in cols}
    _, met = PE.compare_portfolios(prices_oos, portfolios, PE.DATE_START, PE.DATE_END,
                                   benchmark_name="Benchmark", rebalance=rebalance,
                                   rf_annual=rf_annual)
    return met["Sharpe Ratio"], met["Retorno acumulado (%)"]


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        prices, tickers, w_mkt = M.load_data(str(M.DATA_DIR / "input_mkt_px.xlsx"),
                                             str(M.DATA_DIR / "input_mkt_w.xlsx"))
    prices_oos = PE.load_prices(str(M.DATA_DIR / "input_current_mkt_px.xlsx"))

    def l1(df, a, b):
        if a in df.columns and b in df.columns and df[a].notna().all() and df[b].notna().all():
            return float(np.abs(df[a] - df[b]).sum())
        return np.nan

    filas = []
    for nombre, kw in VARIANTES:
        kw = dict(kw)
        rebalance = kw.pop("rebalance", "none")
        rf_annual = kw.pop("rf_annual", 0.0)
        df_w = pesos_variante(prices, tickers, w_mkt, **kw)
        sharpe, cum = evaluar(df_w, prices_oos, rebalance=rebalance, rf_annual=rf_annual)
        fila = {"variante": nombre}
        for c in ("Benchmark", "BL", "EP-eq", "qEP-eq", "MV-hist", "BL-hist", "EP", "qEP"):
            fila[f"sharpe_{c}"] = float(sharpe.get(c, np.nan))
            fila[f"ret_{c}"] = float(cum.get(c, np.nan))
        fila["l1_BL_vs_bench"] = l1(df_w, "BL", "Benchmark")
        fila["l1_EPeq_vs_bench"] = l1(df_w, "EP-eq", "Benchmark")
        fila["l1_EP_vs_MVhist"] = l1(df_w, "EP", "MV-hist")
        fila["l1_BLhist_vs_MVhist"] = l1(df_w, "BL-hist", "MV-hist")
        fila["l1_EP_vs_qEP"] = l1(df_w, "EP", "qEP")
        fila["l1_EPeq_vs_qEPeq"] = l1(df_w, "EP-eq", "qEP-eq")
        fila["l1_MVhist_vs_bench"] = l1(df_w, "MV-hist", "Benchmark")
        filas.append(fila)
        print(f"  {nombre:<22s} Sharpe: bench {fila['sharpe_Benchmark']:.2f} | BL {fila['sharpe_BL']:.2f} "
              f"| EP-eq {fila['sharpe_EP-eq']:.2f} || sin views {fila['sharpe_MV-hist']:.2f} "
              f"| BL-hist {fila['sharpe_BL-hist']:.2f} | EP {fila['sharpe_EP']:.2f} | qEP {fila['sharpe_qEP']:.2f}"
              f"   ||EP−ctrl||₁ {fila['l1_EP_vs_MVhist']:.3f}  ||EP−qEP||₁ {fila['l1_EP_vs_qEP']:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "merval_robustez.csv"
    pd.DataFrame(filas).to_csv(out, index=False, float_format="%.6f")
    print(f"\nTabla guardada en: {out}")


if __name__ == "__main__":
    main()
