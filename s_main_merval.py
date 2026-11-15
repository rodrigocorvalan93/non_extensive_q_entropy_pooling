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
  4. q-Tsallis-EP (q=2, divergencia chi-cuadrado de Pearson)

Controles (ver run_controls):
  BL parte del equilibrio (Π = δ Σ w_mkt) y EP de los escenarios históricos,
  así que comparar BL contra EP mezcla dos cosas: el prior y el método de
  incorporar los views. Para separarlas se calculan además:
    - la cartera media-varianza con el prior histórico SIN views;
    - BL aplicado al prior histórico;
    - EP y q-EP sobre escenarios centrados en el equilibrio (misma media
      del prior que BL);
    - las carteras 1/N y de mínima varianza como referencia.
  Todos los pesos se exportan a results/merval_pesos_<retornos>.csv, que es
  la entrada de portfolio_evolution.py.

Tipos de retornos soportados:
  - "log"    : logarítmicos ln(Pt/Pt-1) — mejor propiedades estadísticas
  - "simple" : simples (Pt-Pt-1)/Pt-1 — más intuitivo
  - "delta"  : diferencia Yt - Yt-1 — para yields de bonos (por si se decide aplicar a otro tipo de activos)

Requisitos:
  - entropy_pooling_v2.py, views_config.py, models.py (mismo directorio)
  - covariance_estimation.py (estimación robusta para paneles desbalanceados)
  - data/dataacciones/input_mkt_px.xlsx (precios históricos)
  - data/dataacciones/input_mkt_w.xlsx (weights del Merval)

Salidas (carpeta results/):
  - merval_pesos_<retornos>.csv, resultados_<retornos>.xlsx,
    model_comparison_<retornos>.png

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
DATA_DIR = HERE / "data" / "dataacciones"
RESULTS_DIR = HERE / "results"

from views_config import (
    ViewSpec, build_views,
    print_views_summary, print_bl_views, print_ep_views,
)
from models import (
    run_black_litterman, run_entropy_pooling, run_q_tsallis_ep,
    print_model_comparison, plot_model_comparison, ModelResult,
    _mean_variance_optimal,
)
from views_config import build_ep_views
from covariance_estimation import (
    prepare_prices,
    estimate_covariance,
    compute_returns_robust,
    prepare_data_for_ep,
    get_panel_diagnostics,
)

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════
DELTA = 2.5
TAU = 0.05
Q_TSALLIS = 2.0
CONFIDENCE = 0.5
MAX_WEIGHT = 0.30     # Máximo 30% por activo (tope de concentración elegido para el ejercicio)
RETURN_TYPE = "both"   # "log", "simple", "delta", "both"

# Cómo entran en EP los 8 views "energética le gana a banco":
#   "inequality"       : E[R_energía] >= E[R_banco]  (lectura literal; especificación principal)
#   "blended_equality" : E[R_energía - R_banco] = (1 - c)·spread_prior  (primera versión del
#                        ejercicio; solo es factible si el prior ordena YPFD por encima de PAMP)
EP_RELATIVE_ENCODING = "inequality"

# ── Parámetros de estimación robusta ──
MAX_GAP_FILL = 5       # Máximo de días consecutivos sin operar que se rellenan
MIN_OBS_PAIRWISE = 60  # Mínimo de observaciones solapadas para estimar covarianza pairwise


# ═══════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════

def load_data(prices_path, weights_path):
    """
    Carga precios y weights del benchmark.

    A diferencia de la versión anterior, NO rellena NaN de inicio de serie.
    Solo limpia huecos cortos (días sin operar). Los papeles nuevos con
    menos historia mantienen sus NaN al inicio.

    El manejo robusto de NaN se delega a covariance_estimation.py:
      - prepare_prices() rellena solo huecos cortos
      - estimate_covariance() estima pairwise si hay NaN
      - prepare_data_for_ep() recorta filas para EP
    """
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

    return prices, tickers, w_mkt


def compute_returns(prices, method="log"):
    """
    Calcula retornos (versión legacy, para el caso sin NaN).
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

def define_pm_views(ep_relative: str = EP_RELATIVE_ENCODING, reverse: bool = False):
    """
    Views del PM de renta variable argentino (septiembre 2025).

    View 1 — Ranking de overweight en energéticas:
        YPFD > PAMP > TGSU2 > CEPU.  Confianza: 70%.

    View 2 — Underweight en bancos respecto de las energéticas favoritas:
        cada una de las energéticas top (YPFD, PAMP) le gana a cada banco
        (GGAL, BBAR, BMA, SUPV).  Confianza: 50%.

    Los dos views son ORDINALES. Cada modelo los recibe en la forma que
    puede representar, por eso la función devuelve dos listas:

    views_bl  (Black-Litterman)
        BL solo admite igualdades sobre retornos esperados. El ranking entra
        como cadena de views relativos con spread ~0 y los 8 views
        energética-vs-banco como views relativos con spread 0. La confianza
        de cada view (70% y 50%) entra en Ω (Idzorek).

    views_ep  (EP y q-EP)
        El ranking entra como 3 desigualdades E[R_i] >= E[R_k].
        Los 8 views energética-vs-banco entran según ep_relative:
          "inequality" (principal): 8 desigualdades E[R_energía] >= E[R_banco],
              impuestas tal cual. La confianza parcial entra una sola vez,
              con la mezcla global (1 - c)·p + c·p̃ (CONFIDENCE).
          "blended_equality" (primera versión): 8 igualdades
              E[R_energía - R_banco] = (1 - 0,5)·spread_prior, es decir,
              acercan los retornos esperados a mitad de camino en lugar de
              imponer "energía >= banco". Como las 8 igualdades fijan
              E[R_YPFD - R_PAMP] en la mitad de su valor en el prior, esta
              forma contradice al ranking (y el problema queda infactible)
              cuando el prior ordena PAMP por encima de YPFD, como pasa con
              el prior de equilibrio.

    reverse=True devuelve los views CONTRARIOS a los del PM (ranking invertido
    y bancos por encima de las energéticas). Se usa como placebo en
    s_main_merval_robustez.py: si el resultado no cambia al invertir los
    views, no es a los views a los que hay que atribuirlo.

    Returns
    -------
    views_bl, views_ep : list[ViewSpec], list[ViewSpec]
    """
    if ep_relative not in ("inequality", "blended_equality"):
        raise ValueError("ep_relative debe ser 'inequality' o 'blended_equality'")

    BANCOS = ["GGAL", "BBAR", "BMA", "SUPV"]
    ENERGIA_TOP = ["YPFD", "PAMP"]
    orden = ["YPFD", "PAMP", "TGSU2", "CEPU"]
    if reverse:
        orden = orden[::-1]
    # (largo, corto) de cada view relativo
    pares = [(bco, e) if reverse else (e, bco) for e in ENERGIA_TOP for bco in BANCOS]

    ranking = ViewSpec.ranking(tickers_ordered=orden, confidence=0.70)
    relativos_igualdad = [
        ViewSpec.relative(ticker_long=a, ticker_short=b, spread=0.0, confidence=0.50)
        for a, b in pares
    ]
    relativos_desigualdad = [
        ViewSpec.relative_ineq(ticker_long=a, ticker_short=b, bound=0.0,
                               direction="geq", confidence=1.0)
        for a, b in pares
    ]

    views_bl = [ranking] + relativos_igualdad
    if ep_relative == "inequality":
        views_ep = [ranking] + relativos_desigualdad
    else:
        views_ep = [ranking] + relativos_igualdad
    return views_bl, views_ep


# ═══════════════════════════════════════════════════════════════
# 4. PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(prices, tickers, w_mkt, views, return_method="log", max_weight=1.0,
                 views_ep=None, with_controls=True):
    """
    Pipeline principal: estima modelos BL, EP-Shannon y q-Tsallis-EP.

    views    : lista de ViewSpec para Black-Litterman.
    views_ep : lista de ViewSpec para EP y q-EP (por defecto, la misma).
               Ver define_pm_views(): cada modelo recibe los views ordinales
               del PM en la forma que puede representar.

    Manejo de paneles desbalanceados:
      - Σ para BL se estima con toda la información pairwise disponible
        (no se pierde historia de los activos viejos).
      - X y p para EP usan solo las filas completas (EP necesita escenarios
        sin NaN), pero Σ sigue usando la estimación pairwise completa.
    """
    N = len(tickers)

    # ── Calcular retornos (preservando NaN de inicio) ──
    returns = compute_returns_robust(
        prices, method=return_method,
        max_gap_fill=MAX_GAP_FILL, verbose=True,
    )
    print(f"\n  Retornos ({return_method}): {returns.shape[0]} obs × {returns.shape[1]} activos")
    print(f"  Período: {returns.index[0].date()} → {returns.index[-1].date()}")

    # ── Estimar Σ (pairwise si hay NaN, estándar si no) ──
    Sigma = estimate_covariance(
        returns, min_obs_pairwise=MIN_OBS_PAIRWISE, verbose=True,
    )
    Sigma = 0.5 * (Sigma + Sigma.T)  # simetría numérica extra

    # ── Preparar X y p para Entropy Pooling ──
    # EP requiere escenarios completos (sin NaN), así que usamos solo
    # las filas donde todos los activos tienen dato.
    X, p = prepare_data_for_ep(returns, verbose=True)
    J = X.shape[0]

    # ── Prior: media usando toda la historia individual de cada activo ──
    mu_prior = np.array([
        returns.iloc[:, i].dropna().mean() for i in range(N)
    ])
    vol_prior = np.sqrt(np.diag(Sigma))

    print(f"\n  Estadísticas del prior (% diario):")
    print(f"  {'Ticker':<8s}  {'E[R]':>8s}  {'Vol':>8s}  {'Sharpe':>8s}  {'Obs':>6s}")
    print(f"  {'-'*44}")
    for i, t in enumerate(tickers):
        n_obs_i = returns.iloc[:, i].dropna().shape[0]
        sr = mu_prior[i] / vol_prior[i] * np.sqrt(252) if vol_prior[i] > 0 else 0
        print(f"  {t:<8s}  {mu_prior[i]*100:>8.4f}  {vol_prior[i]*100:>8.4f}  {sr:>8.2f}  {n_obs_i:>6d}")

    if views_ep is None:
        views_ep = views
    print_views_summary(views)
    if views_ep is not views:
        print("\n  Views en la forma que recibe EP / q-EP:")
        print_views_summary(views_ep)
    bl_views, _ = build_views(views, tickers, Sigma, tau=TAU)
    ep_views = build_ep_views(views_ep, tickers, X, p)
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

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_path = RESULTS_DIR / f"model_comparison_{return_method}.png"
    plot_model_comparison(list(results.values()), tickers, w_mkt, save_path=str(plot_path))

    # ── Exportar tablas a Excel ──
    excel_path = RESULTS_DIR / f"resultados_{return_method}.xlsx"
    _export_results_to_excel(results, tickers, w_mkt, str(excel_path))

    # ── Controles: prior × método ──
    if ep_views and with_controls:
        print("\n▶ Calculando carteras de control (prior × método)...")
        df_w = compute_design(X, p, Sigma, tickers, w_mkt, views, views_ep,
                              max_weight=max_weight)
        # Los tres modelos del pipeline tienen que coincidir con el diseño.
        for key in ("BL", "EP", "qEP"):
            if key in results:
                assert np.allclose(results[key].w_optimal, df_w[key].values, atol=1e-8), key
        out_csv = RESULTS_DIR / f"merval_pesos_{return_method}.csv"
        df_w.to_csv(out_csv, float_format="%.10f")
        print(f"  Pesos de todas las carteras exportados a: {out_csv}")
        print_controls(df_w)

        diag = view_diagnostics(X, p, Sigma, tickers, w_mkt, views_ep,
                                dates=returns.dropna().index)
        diag["retornos"] = return_method
        # Proxy de iliquidez: porcentaje de ruedas con retorno cero de cada papel
        diag["dias_retorno_cero_pct"] = {t: float((returns[t] == 0).mean() * 100) for t in tickers}
        # Precios faltantes en el archivo de insumos (se completan por interpolación lineal)
        faltantes = prices.isna().sum()
        diag["precios_faltantes"] = {t: int(n) for t, n in faltantes.items() if n > 0}
        diag["peso_maximo"] = {c: float(df_w[c].max()) for c in df_w.columns
                               if df_w[c].notna().all()}
        import json

        def redondear(x, cifras=10):
            # Flotantes a 10 cifras significativas: el archivo no cambia de una
            # corrida a otra por diferencias en el último dígito.
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

        out_json = RESULTS_DIR / f"merval_diagnosticos_{return_method}.json"
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(redondear(diag), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"  Diagnósticos de los views exportados a: {out_json}")

    return results


def compute_design(X, p, Sigma, tickers, w_mkt, views_bl, views_ep, *,
                   delta=None, tau=None, q=None, confidence=None, max_weight=None):
    """
    Diseño completo {prior de equilibrio, prior histórico} × {sin views, BL, EP, q-EP}.

    Black-Litterman parte del equilibrio (Π = δ Σ w_mkt) y EP de los
    escenarios históricos. Si se comparan directamente, la diferencia mezcla
    el prior con el método de incorporar los views. Este diseño los separa:
    dentro de cada prior, la cartera "sin views" es el control, y lo que se
    aleja de ella es el efecto de los views.

    Devuelve un DataFrame de pesos (fracción), una columna por cartera:

      prior de equilibrio
        "Benchmark" : pesos del índice (es el control: sin views, la
                      media-varianza con Π devuelve el benchmark)
        "MV-eq"     : esa media-varianza, calculada (chequeo numérico)
        "BL"        : Black-Litterman
        "EP-eq"     : EP-Shannon sobre escenarios centrados en el equilibrio
        "qEP-eq"    : q-Tsallis-EP sobre los mismos escenarios
      prior histórico
        "MV-hist"   : media-varianza con el prior histórico y SIN views (el
                      mismo pipeline de EP con el conjunto de views vacío)
        "BL-hist"   : la fórmula de BL aplicada al prior de medias históricas
        "EP"        : EP-Shannon sobre los escenarios históricos
        "qEP"       : q-Tsallis-EP sobre los escenarios históricos
      referencias
        "1/N"       : equiponderada
        "MinVar"    : mínima varianza (long-only, mismo tope)

    Si un problema de EP resulta infactible (ver define_pm_views), la columna
    correspondiente queda en NaN y se avisa por consola.
    """
    delta = DELTA if delta is None else delta
    tau = TAU if tau is None else tau
    q = Q_TSALLIS if q is None else q
    confidence = CONFIDENCE if confidence is None else confidence
    max_weight = MAX_WEIGHT if max_weight is None else max_weight

    J, N = X.shape
    p = p / p.sum()
    kw = dict(delta=delta, confidence=confidence, max_weight=max_weight)
    out = {"Benchmark": np.asarray(w_mkt, dtype=float)}

    def _ep(Xs, views, family):
        ep_v = build_ep_views(views, tickers, Xs, p)
        try:
            if family == "S":
                return run_entropy_pooling(Xs, p, ep_v, tickers, w_mkt, **kw).w_optimal
            return run_q_tsallis_ep(Xs, p, ep_v, tickers, w_mkt, q=q, **kw).w_optimal
        except RuntimeError as exc:
            print(f"    ⚠ EP ({family}) sin solución: {exc}")
            return np.full(N, np.nan)

    bl_views, _ = build_views(views_bl, tickers, Sigma, tau=tau)
    mu_hist = X.T @ p

    # ── Prior de equilibrio: escenarios con la misma forma y media Π ──
    Pi = delta * Sigma @ w_mkt
    X_eq = X - mu_hist + Pi
    out["MV-eq"] = _ep(X_eq, [], "S")
    if bl_views:
        out["BL"] = run_black_litterman(
            Sigma, w_mkt, bl_views, tickers, delta=delta, tau=tau, X=X, p=p,
            max_weight=max_weight).w_optimal
    out["EP-eq"] = _ep(X_eq, views_ep, "S")
    out["qEP-eq"] = _ep(X_eq, views_ep, "T")

    # ── Prior histórico ──
    out["MV-hist"] = _ep(X, [], "S")
    if bl_views:
        out["BL-hist"] = run_black_litterman(
            Sigma, w_mkt, bl_views, tickers, delta=delta, tau=tau, X=X, p=p,
            max_weight=max_weight, prior_mean=mu_hist).w_optimal
    out["EP"] = _ep(X, views_ep, "S")
    out["qEP"] = _ep(X, views_ep, "T")

    # ── Referencias sin views ni prior de retornos ──
    out["1/N"] = np.full(N, 1.0 / N)
    Scnd = X.T @ (X * p[:, None])
    S_hist = 0.5 * (Scnd + Scnd.T) - np.outer(mu_hist, mu_hist)
    out["MinVar"] = _mean_variance_optimal(np.zeros(N), S_hist, delta=1.0,
                                           long_only=True, max_weight=max_weight)

    return pd.DataFrame(out, index=pd.Index(tickers, name="ticker"))


def view_diagnostics(X, p, Sigma, tickers, w_mkt, views_ep, dates=None):
    """
    Cuánto mueven los views al posterior, con cada prior y cada familia.

    Para el posterior SIN mezclar con el prior (confianza plena) informa:
      - fracción de escenarios efectivos de orden 2 (Kish) y de orden 1
        (entropía; Meucci, 2012);
      - cuántas de las desigualdades quedan activas;
      - el mayor cambio de un retorno esperado, en puntos porcentuales por año.
    """
    from entropy_pooling_v2 import entropy_prog
    J, N = X.shape
    p = p / p.sum()
    mu_hist = X.T @ p
    Pi = DELTA * Sigma @ w_mkt
    out = {"J": int(J), "N": int(N), "delta": DELTA, "tau": TAU, "q": Q_TSALLIS,
           "confianza_global": CONFIDENCE, "tope": MAX_WEIGHT,
           "codificacion_relativos": EP_RELATIVE_ENCODING}
    if dates is not None:
        out["primera_fecha"] = str(pd.Timestamp(dates[0]).date())
        out["ultima_fecha"] = str(pd.Timestamp(dates[-1]).date())
    out["media_anual_pct"] = {
        "historico": {t: float(mu_hist[i] * 252 * 100) for i, t in enumerate(tickers)},
        "equilibrio": {t: float(Pi[i] * 252 * 100) for i, t in enumerate(tickers)},
    }
    for nombre, Xs in (("historico", X), ("equilibrio", X - mu_hist + Pi)):
        ep_v = build_ep_views(views_ep, tickers, Xs, p)
        for fam, q in (("S", 1.0), ("T", Q_TSALLIS)):
            clave = f"{nombre}_{'shannon' if fam == 'S' else 'tsallis'}"
            try:
                x, _, lv = entropy_prog(p, ep_v.A, ep_v.b, ep_v.Aeq, ep_v.beq,
                                        entropy_family=fam, q=q)
            except RuntimeError:
                out[clave] = None
                continue
            y = x[x > 0]
            n_in = ep_v.A.shape[0]
            out[clave] = {
                "ess_kish_sobre_J": float(1.0 / (J * np.sum(x ** 2))),
                "ens_entropia_sobre_J": float(np.exp(-np.sum(y * np.log(y))) / J),
                "desigualdades": int(n_in),
                "desigualdades_activas": int(np.sum(lv[:n_in] > 1e-10)),
                "max_cambio_media_pp_anual": float(np.max(np.abs(Xs.T @ x - Xs.T @ p)) * 252 * 100),
            }
    return out


def print_controls(df_w):
    """Tabla de pesos (%) del diseño prior × método y distancias L1."""
    orden = ["Benchmark", "BL", "EP-eq", "qEP-eq", "MV-hist", "BL-hist", "EP", "qEP"]
    df_w = df_w.dropna(axis=1, how="all")
    cols = [c for c in orden if c in df_w.columns]
    print("\n  Diseño prior × método: pesos (%)")
    print("  [equilibrio: Benchmark, BL, EP-eq, qEP-eq]  [histórico: MV-hist, BL-hist, EP, qEP]")
    print((df_w[cols] * 100).round(2).to_string())
    def l1(a, b):
        return float(np.abs(df_w[a] - df_w[b]).sum())
    pares = [("EP", "MV-hist"), ("qEP", "MV-hist"), ("BL-hist", "MV-hist"),
             ("BL", "Benchmark"), ("EP-eq", "Benchmark"), ("qEP-eq", "Benchmark"),
             ("MV-eq", "Benchmark"), ("EP", "qEP"), ("EP-eq", "qEP-eq"),
             ("MV-hist", "Benchmark")]
    print("\n  Distancia L1 entre carteras (0 = idénticas, 2 = disjuntas):")
    for a, b in pares:
        if a in df_w.columns and b in df_w.columns:
            print(f"    ||{a} − {b}||₁ = {l1(a, b):.3f}")


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

    prices_path = DATA_DIR / "input_mkt_px.xlsx"
    weights_path = DATA_DIR / "input_mkt_w.xlsx"

    for fp, desc in [(prices_path, "precios"), (weights_path, "weights")]:
        if not fp.exists():
            print(f"\n⚠ No se encontró: {fp}")
            print(f"  Se espera '{fp.name}' en {DATA_DIR}.")
            sys.exit(1)

    prices, tickers, w_mkt = load_data(str(prices_path), str(weights_path))

    print(f"\nDatos cargados:")
    print(f"  Precios: {prices.shape[0]} fechas × {prices.shape[1]} activos")
    print(f"  Rango: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Activos ({len(tickers)}): {tickers}")

    # ── Diagnóstico del panel ──
    diag = get_panel_diagnostics(prices)
    min_coverage = diag["cobertura_pct"].min()
    if min_coverage < 100:
        print(f"\n  ⚠ Panel desbalanceado detectado:")
        for _, row in diag[diag["cobertura_pct"] < 100].iterrows():
            print(f"    {row['ticker']}: {row['cobertura_pct']}% cobertura, "
                  f"desde {row['primera_fecha'].date()}, "
                  f"{row['nan_inicio']} fechas sin historia")
    else:
        print(f"\n  ✓ Panel balanceado: todos los activos tienen historia completa")

    print(f"\n  Weights del Merval (top 5):")
    for t, w in sorted(zip(tickers, w_mkt), key=lambda x: -x[1])[:5]:
        print(f"    {t}: {w:.4%}")

    views_bl, views_ep = define_pm_views()

    if RETURN_TYPE == "both":
        all_results = {}
        for method in ["log", "simple"]:
            print(f"\n{'#'*70}")
            print(f"  RETORNOS: {method.upper()}")
            print(f"{'#'*70}")
            all_results[method] = run_pipeline(
                prices, tickers, w_mkt, views_bl, return_method=method,
                max_weight=MAX_WEIGHT, views_ep=views_ep)

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
        results = run_pipeline(prices, tickers, w_mkt, views_bl, return_method=RETURN_TYPE,
                               max_weight=MAX_WEIGHT, views_ep=views_ep)

    print("\n✓ Pipeline completado exitosamente.")
