#%%
"""
s_main_bonds.py
===============
Aplicación de q-Entropy Pooling a bonos (renta fija).

Extiende el pipeline de s_main_merval.py para trabajar con yields de bonos
como factor invariante, incorporando el mapeo Δy → ΔP/P vía modified
duration y (opcionalmente) convexity.

Concepto clave:
  En renta fija, el factor invariante es el cambio en yield (Δy), no el
  retorno de precio. Sin embargo, la optimización media-varianza y el
  backtesting requieren retornos de precio. El puente entre ambos es:

      ΔP/P ≈ -D_mod · Δy + ½ · C · (Δy)²

  donde D_mod es la modified duration y C la convexity del bono.

Modos de duration (DURATION_MODE):
  - "constant"        : duration fija (un valor por bono, fecha de rebalanceo)
  - "rolling"         : duration histórica rolling (serie temporal)
  - "rolling_convex"  : rolling duration + convexity (aproximación de 2do orden)

Inputs:
  - input_mkt_px.xlsx       : yields históricas (NO precios)
  - input_mkt_w.xlsx        : weights del portfolio benchmark
  - input_bond_stats.xlsx   : hoja "duration" y hoja "convexity"
    * En modo "constant": hoja "duration" con 2 columnas (ticker, duration)
    * En modo "rolling": hoja "duration" con series temporales (fechas × tickers)
    * En modo "rolling_convex": hojas "duration" + "convexity" con series temporales

Views:
  Los views se expresan sobre Δy (el factor invariante). Por ejemplo:
    - Ranking: "GD29 baja más de yield que GD30" (= GD29 sube más de precio)
    - Relativo: "GD35 tendrá un Δy más negativo que GT10"
  Recordar: Δy negativo = suba de precio. Los views de ranking sobre Δy
  se traducen correctamente a P&L gracias al mapeo con duration.

Requisitos:
  - entropy_pooling_v2.py, views_config.py, models.py (mismo directorio)

Ref: Black & Litterman (1992), Meucci (2008), Tsallis (1988),
     Corvalán Salguero (2026)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
MAX_WEIGHT = 0.30

# ── Modo de duration ──
# "constant"        → un escalar por bono (fecha del rebalanceo)
# "rolling"         → serie temporal de mod duration (ΔP/P ≈ -D·Δy)
# "rolling_convex"  → serie temporal de duration + convexity (ΔP/P ≈ -D·Δy + ½C·Δy²)
DURATION_MODE = "rolling_convex"


# ═══════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════

def load_yields_and_weights(yields_path: str, weights_path: str):
    """
    Carga yields históricas y weights del benchmark.
    Idéntico a load_data() de s_main_merval.py pero renombrado
    para claridad semántica (son yields, no precios).
    """
    ext = Path(yields_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        yields = pd.read_excel(yields_path, index_col=0, parse_dates=True)
    else:
        yields = pd.read_csv(yields_path, index_col=0, parse_dates=True)
    yields.columns = [c.strip() for c in yields.columns]
    yields.index = pd.to_datetime(yields.index)
    yields = yields.sort_index()

    if Path(weights_path).suffix.lower() in (".xlsx", ".xls"):
        df_w = pd.read_excel(weights_path)
    else:
        df_w = pd.read_csv(weights_path)
    df_w["ticker"] = df_w["ticker"].str.strip()
    weights_dict = dict(zip(df_w["ticker"], df_w["weight"].astype(float)))

    common = sorted(set(yields.columns) & set(weights_dict.keys()))
    if not common:
        raise ValueError("No hay tickers en común entre yields y weights")
    yields = yields[common]
    tickers = list(common)
    w_mkt = np.array([weights_dict[t] for t in tickers])
    w_mkt = w_mkt / w_mkt.sum()

    null_count = yields.isnull().sum().sum()
    if null_count > 0:
        print(f"  Interpolando {null_count} valores nulos en yields...")
        yields = yields.interpolate(method="linear").ffill().bfill()

    return yields, tickers, w_mkt


def load_bond_stats(
    stats_path: str,
    tickers: List[str],
    yields_index: pd.DatetimeIndex,
    mode: str = "rolling_convex",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Carga duration y convexity desde input_bond_stats.xlsx.

    Modos:
      "constant"       → hoja "duration" con 2 columnas (ticker, duration).
                          Se expande a un DataFrame constante con el mismo
                          DatetimeIndex que los yields.
      "rolling"        → hoja "duration" con series temporales.
      "rolling_convex" → hojas "duration" + "convexity" con series temporales.

    Returns
    -------
    duration_df : DataFrame (fechas × tickers) de modified duration
    convexity_df : DataFrame (fechas × tickers) de convexity, o None si no aplica
    """
    xls = pd.ExcelFile(stats_path)
    sheets = [s.lower().strip() for s in xls.sheet_names]

    if mode == "constant":
        # ── Modo constante: leer como tabla ticker/duration ──
        df_raw = pd.read_excel(stats_path, sheet_name="duration")
        # Detectar si es formato tabla (ticker, duration) o serie temporal
        cols_lower = [c.lower().strip() for c in df_raw.columns]
        if "ticker" in cols_lower or "instrumento" in cols_lower:
            # Formato columnar: ticker | duration
            col_ticker = df_raw.columns[0]
            col_dur = df_raw.columns[1]
            df_raw[col_ticker] = df_raw[col_ticker].astype(str).str.strip()
            dur_dict = dict(zip(df_raw[col_ticker], df_raw[col_dur].astype(float)))
            dur_values = [dur_dict.get(t, np.nan) for t in tickers]
            if any(np.isnan(v) for v in dur_values):
                missing = [t for t, v in zip(tickers, dur_values) if np.isnan(v)]
                raise ValueError(f"Durations faltantes para: {missing}")
            # Expandir a DataFrame constante
            duration_df = pd.DataFrame(
                np.tile(dur_values, (len(yields_index), 1)),
                index=yields_index,
                columns=tickers,
            )
            print(f"  Duration (constante): {dict(zip(tickers, dur_values))}")
        else:
            # Si pusieron serie temporal en modo constant, tomar el último valor
            df_ts = pd.read_excel(stats_path, sheet_name="duration",
                                  index_col=0, parse_dates=True)
            df_ts.columns = [c.strip() for c in df_ts.columns]
            last_dur = df_ts[tickers].iloc[-1].values
            duration_df = pd.DataFrame(
                np.tile(last_dur, (len(yields_index), 1)),
                index=yields_index,
                columns=tickers,
            )
            print(f"  Duration (constante, tomada del último dato):")
            for t, d in zip(tickers, last_dur):
                print(f"    {t}: {d:.4f}")

        return duration_df, None

    elif mode in ("rolling", "rolling_convex"):
        # ── Duration rolling ──
        dur_df = pd.read_excel(stats_path, sheet_name="duration",
                               index_col=0, parse_dates=True)
        dur_df.columns = [c.strip() for c in dur_df.columns]
        dur_df.index = pd.to_datetime(dur_df.index)
        dur_df = dur_df.sort_index()

        # Verificar tickers
        missing_dur = [t for t in tickers if t not in dur_df.columns]
        if missing_dur:
            raise ValueError(f"Tickers sin duration en bond_stats: {missing_dur}")
        dur_df = dur_df[tickers]

        # Alinear fechas con yields (reindex + ffill para cubrir gaps)
        dur_df = dur_df.reindex(yields_index, method="ffill").bfill()
        null_dur = dur_df.isnull().sum().sum()
        if null_dur > 0:
            print(f"  ⚠ Interpolando {null_dur} nulls en duration...")
            dur_df = dur_df.interpolate(method="linear").ffill().bfill()

        print(f"  Duration (rolling): {dur_df.shape[0]} fechas")
        print(f"    Rango duration al inicio: "
              f"{dict(zip(tickers, dur_df.iloc[0].round(3).values))}")
        print(f"    Rango duration al final:  "
              f"{dict(zip(tickers, dur_df.iloc[-1].round(3).values))}")

        convex_df = None
        if mode == "rolling_convex":
            if "convexity" not in sheets:
                raise ValueError(
                    "Modo 'rolling_convex' requiere hoja 'convexity' en bond_stats"
                )
            convex_df = pd.read_excel(stats_path, sheet_name="convexity",
                                      index_col=0, parse_dates=True)
            convex_df.columns = [c.strip() for c in convex_df.columns]
            convex_df.index = pd.to_datetime(convex_df.index)
            convex_df = convex_df.sort_index()

            missing_cvx = [t for t in tickers if t not in convex_df.columns]
            if missing_cvx:
                raise ValueError(f"Tickers sin convexity en bond_stats: {missing_cvx}")
            convex_df = convex_df[tickers]
            convex_df = convex_df.reindex(yields_index, method="ffill").bfill()
            null_cvx = convex_df.isnull().sum().sum()
            if null_cvx > 0:
                print(f"  ⚠ Interpolando {null_cvx} nulls en convexity...")
                convex_df = convex_df.interpolate(method="linear").ffill().bfill()

            print(f"  Convexity (rolling): {convex_df.shape[0]} fechas")

        return dur_df, convex_df

    else:
        raise ValueError(
            f"DURATION_MODE no reconocido: '{mode}'. "
            "Usar 'constant', 'rolling' o 'rolling_convex'."
        )


# ═══════════════════════════════════════════════════════════════
# 3. MAPEO Δy → ΔP/P
# ═══════════════════════════════════════════════════════════════

def compute_delta_yields(yields: pd.DataFrame) -> pd.DataFrame:
    """Calcula Δy = Y(t) - Y(t-1) para cada bono."""
    return yields.diff().dropna()


def map_dy_to_price_returns(
    delta_y: pd.DataFrame,
    duration: pd.DataFrame,
    convexity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Mapea cambios de yield a retornos aproximados de precio.

    Fórmula de Taylor del precio de un bono:
        ΔP/P ≈ -D_mod · Δy + ½ · C · (Δy)²

    donde:
        D_mod = modified duration (positiva)
        C     = convexity (positiva)
        Δy    = cambio en yield (en formato decimal, ej: +0.0050 = +50 bps)

    Si convexity es None, usa solo el término de primer orden:
        ΔP/P ≈ -D_mod · Δy

    Parameters
    ----------
    delta_y   : DataFrame (T×N) de cambios en yield
    duration  : DataFrame (T×N) de modified duration (alineado por fecha)
    convexity : DataFrame (T×N) de convexity (opcional)

    Returns
    -------
    DataFrame (T×N) de retornos de precio aproximados

    Notas
    -----
    - Los yields en input_mkt_px.xlsx deben estar en formato decimal
      (ej: 0.05 para 5%, NO 5.0). Si están en porcentaje, dividir por 100
      antes de aplicar esta función.
    - La duration y convexity deben estar alineadas en fechas con delta_y.
      La función load_bond_stats() ya se encarga de esto.
    """
    # Alinear fechas (por seguridad)
    common_idx = delta_y.index.intersection(duration.index)
    dy = delta_y.loc[common_idx]
    dur = duration.loc[common_idx]

    # Término de primer orden: -D · Δy
    price_returns = -dur * dy

    # Término de segundo orden: + ½ · C · (Δy)²
    if convexity is not None:
        cvx = convexity.loc[common_idx]
        price_returns = price_returns + 0.5 * cvx * (dy ** 2)

    return price_returns


def check_yield_scale(yields: pd.DataFrame) -> str:
    """
    Detecta si los yields están en formato decimal (0.05) o porcentaje (5.0).
    Retorna "decimal" o "percent".
    """
    median_yield = yields.median().median()
    if median_yield > 1.0:
        return "percent"
    else:
        return "decimal"


# ═══════════════════════════════════════════════════════════════
# 4. VIEWS PARA BONOS
# ═══════════════════════════════════════════════════════════════

def define_bond_views():
    """
    Views del PM de renta fija para Globales argentinos + Treasuries.

    Escenarios del PM (horizonte: hoy → 31/dic/2026, ~190 días hábiles):
      - Bueno (70% prob): convergencia a investment grade, TIREA → 8.8%
        GD41: +20.5%  GD35: +18.7%  GD38: +17.9%  GD46: +17.6%
        GD30: +8.9%   GD29: +5.7%
      - Malo (30% prob): blow-up, TIREA → ~16%
        GD30: -0.6%   GD29: -0.8%   GD38: -8.7%   GD35: -12.9%
        GD41: -13.6%  GD46: -15.3%

    Total Return esperado ponderado (70/30):
      GD41: 10.3%  GD38: 9.9%  GD35: 9.2%  GD46: 7.7%
      GD30: 6.0%   GD29: 3.8%

    Views de cola del PM:
      - P(Δy > +300 bps en un día) ≤ 20%  (blow-up)
      - P(Δy < -300 bps en un día) ≤ 5%   (compresión extrema)

    Sin views sobre Treasuries (GT2, GT10): se dejan al prior.
    """
    globales = ["GD29", "GD30", "GD35", "GD38", "GD41", "GD46"]

    views = []
    for t in globales:
        # Cola derecha: blow-up de ≥300 bps en un día → prob ≤ 20%
        views.append(
            ViewSpec.tail_upper(t, threshold=3.0, max_prob=0.20, confidence=0.70)
        )
        # Cola izquierda: compresión de ≥300 bps en un día → prob ≤ 5%
        views.append(
            ViewSpec.tail(t, threshold=-3.0, max_prob=0.05, confidence=0.70)
        )

    return views


# ═══════════════════════════════════════════════════════════════
# 5. PIPELINE PARA BONOS
# ═══════════════════════════════════════════════════════════════

def run_bond_pipeline(
    yields: pd.DataFrame,
    tickers: List[str],
    w_mkt: np.ndarray,
    views: List[ViewSpec],
    duration_df: pd.DataFrame,
    convexity_df: Optional[pd.DataFrame],
    duration_mode: str = "rolling_convex",
    max_weight: float = 1.0,
) -> Dict[str, ModelResult]:
    """
    Pipeline completo para bonos:

    1. Calcular Δy (factor invariante) → se usa para EP/q-EP
    2. Mapear Δy → ΔP/P via duration (+convexity) → se usa para MV optimization
    3. Correr los 3 modelos (BL, EP, q-EP)

    El flujo es:
      - EP recibe la matriz X de Δy y actualiza las probabilidades
      - Con las probabilidades posteriores, calcula μ y Σ sobre los
        retornos de precio mapeados (no sobre Δy)
      - La optimización MV se hace sobre retornos de precio
    """
    N = len(tickers)

    # ── Paso 1: Δy (factor invariante) ──
    delta_y = compute_delta_yields(yields)
    print(f"\n  Delta yields: {delta_y.shape[0]} obs × {delta_y.shape[1]} bonos")
    print(f"  Período: {delta_y.index[0].date()} → {delta_y.index[-1].date()}")

    # ── Detectar escala de yields ──
    scale = check_yield_scale(yields)
    if scale == "percent":
        print(f"\n  ⚠ Yields detectados en formato PORCENTAJE (mediana={yields.median().median():.2f})")
        print(f"    Convirtiendo Δy a decimal (÷100) para el mapeo con duration...")
        delta_y_decimal = delta_y / 100.0
    else:
        print(f"  Yields en formato decimal (mediana={yields.median().median():.4f})")
        delta_y_decimal = delta_y.copy()

    # ── Paso 2: Mapear Δy → ΔP/P ──
    cvx = convexity_df if duration_mode == "rolling_convex" else None
    price_returns = map_dy_to_price_returns(delta_y_decimal, duration_df, cvx)

    # Alinear fechas (delta_y pierde la primera fila por el diff)
    common_idx = delta_y.index.intersection(price_returns.index)
    delta_y = delta_y.loc[common_idx]
    delta_y_decimal = delta_y_decimal.loc[common_idx]
    price_returns = price_returns.loc[common_idx]

    print(f"\n  Retornos de precio mapeados (ΔP/P): {price_returns.shape[0]} obs")
    print(f"  Modo duration: {duration_mode}")

    # ── Stats del factor invariante (Δy) ──
    X_dy = delta_y.values  # Para EP: views sobre Δy
    X_pr = price_returns.values  # Para MV: optimización sobre ΔP/P
    J = X_dy.shape[0]
    p = np.full(J, 1.0 / J)

    mu_dy = X_dy.T @ p
    vol_dy = np.sqrt(np.diag(np.cov(X_dy.T, ddof=1)))
    mu_pr = X_pr.T @ p
    vol_pr = np.sqrt(np.diag(np.cov(X_pr.T, ddof=1)))

    print(f"\n  Estadísticas del prior — Factor invariante (Δy, escala original):")
    print(f"  {'Ticker':<8s}  {'E[Δy]':>10s}  {'Vol(Δy)':>10s}")
    print(f"  {'-'*30}")
    for i, t in enumerate(tickers):
        print(f"  {t:<8s}  {mu_dy[i]:>10.6f}  {vol_dy[i]:>10.6f}")

    print(f"\n  Estadísticas del prior — Retornos de precio mapeados (ΔP/P, % diario):")
    print(f"  {'Ticker':<8s}  {'E[R]':>8s}  {'Vol':>8s}  {'Sharpe':>8s}  {'Dur(last)':>10s}")
    print(f"  {'-'*50}")
    for i, t in enumerate(tickers):
        sr = mu_pr[i] / vol_pr[i] * np.sqrt(252) if vol_pr[i] > 0 else 0
        dur_last = duration_df[t].iloc[-1]
        print(f"  {t:<8s}  {mu_pr[i]*100:>8.4f}  {vol_pr[i]*100:>8.4f}  {sr:>8.2f}  {dur_last:>10.3f}")

    # ── Paso 3: Construir views ──
    # Los views se aplican sobre Δy (factor invariante)
    print_views_summary(views)

    Sigma_dy = np.cov(X_dy.T, ddof=1)
    Sigma_dy = 0.5 * (Sigma_dy + Sigma_dy.T)

    Sigma_pr = np.cov(X_pr.T, ddof=1)
    Sigma_pr = 0.5 * (Sigma_pr + Sigma_pr.T)

    bl_views, ep_views = build_views(
        views, tickers, Sigma_pr, X=X_dy, p=p, tau=TAU
    )
    if bl_views:
        print_bl_views(bl_views, tickers)
    if ep_views:
        print_ep_views(ep_views)

    results = {}

    # ── BL: usa retornos de precio directamente ──
    if bl_views:
        t0 = time.time()
        print("\n▶ Ejecutando Black-Litterman (sobre retornos de precio)...")
        results["BL"] = run_black_litterman(
            Sigma_pr, w_mkt, bl_views, tickers,
            delta=DELTA, tau=TAU, X=X_pr, p=p,
            max_weight=max_weight,
        )
        print(f"  ✓ BL completado ({time.time()-t0:.1f}s)")

    # ── EP Shannon: views sobre Δy, optimización sobre ΔP/P ──
    if ep_views:
        t0 = time.time()
        print("\n▶ Ejecutando Entropy Pooling (Shannon)...")
        results["EP"] = _run_ep_bonds(
            X_dy, X_pr, p, ep_views, tickers, w_mkt,
            delta=DELTA, confidence=CONFIDENCE, max_weight=max_weight,
            entropy_family="S", q=1.0,
            model_name="EP-Shannon",
        )
        print(f"  ✓ EP-Shannon completado ({time.time()-t0:.1f}s)")

    # ── q-Tsallis EP: views sobre Δy, optimización sobre ΔP/P ──
    if ep_views:
        t0 = time.time()
        print(f"\n▶ Ejecutando q-Tsallis-EP (q={Q_TSALLIS})...")
        results["qEP"] = _run_ep_bonds(
            X_dy, X_pr, p, ep_views, tickers, w_mkt,
            delta=DELTA, confidence=CONFIDENCE, max_weight=max_weight,
            entropy_family="T", q=Q_TSALLIS,
            model_name=f"q-Tsallis-EP (q={Q_TSALLIS})",
        )
        print(f"  ✓ q-Tsallis-EP completado ({time.time()-t0:.1f}s)")

    # ── Métricas del Benchmark (prior) ──
    from models import _compute_risk_metrics
    mu_bench = X_pr.T @ p  # retornos esperados del prior
    Sigma_bench = Sigma_pr  # covarianza del prior
    risk_bench = _compute_risk_metrics(w_mkt, mu_bench, Sigma_bench, X=X_pr, p=p)

    bench_result = ModelResult(
        model_name="Benchmark",
        w_optimal=w_mkt,
        mu_posterior=mu_bench,
        Sigma_posterior=Sigma_bench,
        frontier_e=np.array([]),
        frontier_s=np.array([]),
        frontier_w=np.array([[]]),
        p_posterior=p,
        risk_metrics=risk_bench,
        params={"note": "Prior equiprobable, weights de mercado"},
    )

    # Lista completa para comparación (Benchmark + todos los modelos)
    all_results_with_bench = [bench_result] + list(results.values())
    print_model_comparison(all_results_with_bench, tickers, w_mkt)

    plot_path = HERE / "model_comparison_bonds.png"
    plot_model_comparison(list(results.values()), tickers, w_mkt,
                          save_path=str(plot_path), bench_result=bench_result)

    # ── Exportar a Excel ──
    excel_path = HERE / "resultados_bonds.xlsx"
    _export_bond_results(results, tickers, w_mkt, duration_df, str(excel_path),
                         bench_risk=risk_bench)

    return results


def _run_ep_bonds(
    X_dy: np.ndarray,
    X_pr: np.ndarray,
    p: np.ndarray,
    ep_views: "EPViews",
    tickers: List[str],
    w_mkt: np.ndarray,
    delta: float = 2.5,
    confidence: float = 1.0,
    max_weight: float = 1.0,
    entropy_family: str = "S",
    q: float = 1.0,
    model_name: str = "EP",
    num_portf: int = 20,
) -> ModelResult:
    """
    EP para bonos: actualiza probabilidades sobre Δy, optimiza sobre ΔP/P.

    Flujo:
      1. Resolver EP sobre X_dy (Δy) → p̃ (probabilidades posteriores)
      2. Aplicar p̃ sobre X_pr (ΔP/P) → μ_post, Σ_post en espacio de precios
      3. Optimizar MV sobre μ_post, Σ_post → weights
    """
    from entropy_pooling_v2 import entropy_prog, efficient_frontier, FrontierOptions
    from models import _mean_variance_optimal, _compute_risk_metrics

    J, N = X_dy.shape
    p = p / p.sum()

    # ── Paso 1: EP sobre Δy ──
    if entropy_family == "S":
        p_view, _, _ = entropy_prog(
            p, ep_views.A, ep_views.b, ep_views.Aeq, ep_views.beq,
            entropy_family="S",
        )
    else:
        p_view, _, _ = entropy_prog(
            p, ep_views.A, ep_views.b, ep_views.Aeq, ep_views.beq,
            entropy_family="T", q=q,
        )

    # Mezcla con confianza global
    p_post = (1 - confidence) * p + confidence * p_view
    p_post = np.maximum(p_post, 0.0)
    p_post = p_post / p_post.sum()

    # ── Paso 2: Momentos posteriores sobre retornos de PRECIO ──
    mu_post = X_pr.T @ p_post
    Scnd = X_pr.T @ (X_pr * p_post[:, None])
    Scnd = 0.5 * (Scnd + Scnd.T)
    Sigma_post = Scnd - np.outer(mu_post, mu_post)

    # ── Paso 3: Optimización MV sobre retornos de precio ──
    w_opt = _mean_variance_optimal(
        mu_post, Sigma_post, delta=delta, long_only=True, max_weight=max_weight
    )

    # Frontera eficiente (sobre retornos de precio)
    opts = FrontierOptions(NumPortf=num_portf, FrontierSpan=(0.3, 0.9))
    frontier_e, frontier_s, frontier_w, _, _ = efficient_frontier(X_pr, p_post, opts)

    # Medidas de riesgo
    risk = _compute_risk_metrics(w_opt, mu_post, Sigma_post, X=X_pr, p=p_post)

    return ModelResult(
        model_name=model_name,
        w_optimal=w_opt,
        mu_posterior=mu_post,
        Sigma_posterior=Sigma_post,
        frontier_e=frontier_e,
        frontier_s=frontier_s,
        frontier_w=frontier_w,
        p_posterior=p_post,
        risk_metrics=risk,
        params={
            "entropy_family": entropy_family,
            "q": q,
            "confidence": confidence,
            "note": "Views sobre Δy, optimización sobre ΔP/P",
        },
    )


def _export_bond_results(
    results: Dict[str, ModelResult],
    tickers: List[str],
    w_mkt: np.ndarray,
    duration_df: pd.DataFrame,
    filepath: str,
    bench_risk: Optional[Dict[str, float]] = None,
) -> None:
    """Exporta resultados a Excel con hojas adicionales de duration info."""
    N = len(tickers)

    # ── Hoja 1: Pesos óptimos ──
    df_w = pd.DataFrame({
        "Ticker": tickers,
        "Benchmark (%)": np.round(w_mkt * 100, 4),
        "Dur (last)": [round(duration_df[t].iloc[-1], 3) for t in tickers],
    })
    for name, r in results.items():
        weights_clean = r.w_optimal * 100
        weights_clean[weights_clean < 0.005] = 0.0
        df_w[f"{r.model_name} (%)"] = np.round(weights_clean, 4)
    # Fila total
    total_row = {"Ticker": "TOTAL", "Benchmark (%)": round(w_mkt.sum() * 100, 4), "Dur (last)": ""}
    for name, r in results.items():
        total_row[f"{r.model_name} (%)"] = round(r.w_optimal.sum() * 100, 4)
    df_w = pd.concat([df_w, pd.DataFrame([total_row])], ignore_index=True)

    # ── Hoja 2: Retornos esperados (de precio, mapeados) ──
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
        mult = 100 if "ratio" not in key.lower() else 1
        # Benchmark primero
        if bench_risk is not None:
            row["Benchmark"] = bench_risk.get(key, np.nan) * mult
        for name, r in results.items():
            val = r.risk_metrics.get(key, np.nan)
            row[r.model_name] = val * mult
        risk_rows.append(row)
    df_risk = pd.DataFrame(risk_rows)

    # ── Escribir ──
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_w.to_excel(writer, sheet_name="Pesos óptimos", index=False)
        df_mu.to_excel(writer, sheet_name="Retornos esperados", index=False)
        df_risk.to_excel(writer, sheet_name="Medidas de riesgo", index=False)

    print(f"  Resultados exportados a: {filepath}")


# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  APLICACIÓN EMPÍRICA: Bonos (Globales ARG + Treasuries)")
    print("  BL vs EP-Shannon vs q-Tsallis-EP (q=2)")
    print(f"  Duration mode: {DURATION_MODE}")
    print("=" * 70)

    yields_path = HERE / "input_mkt_px.xlsx"
    weights_path = HERE / "input_mkt_w.xlsx"
    stats_path = HERE / "input_bond_stats.xlsx"

    for fp, desc in [
        (yields_path, "yields"),
        (weights_path, "weights"),
        (stats_path, "bond stats (duration/convexity)"),
    ]:
        if not fp.exists():
            print(f"\n⚠ No se encontró: {fp}")
            print(f"  Colocar '{fp.name}' en el mismo directorio del script.")
            sys.exit(1)

    # ── Cargar datos ──
    yields, tickers, w_mkt = load_yields_and_weights(
        str(yields_path), str(weights_path)
    )

    print(f"\nDatos cargados:")
    print(f"  Yields: {yields.shape[0]} fechas × {yields.shape[1]} bonos")
    print(f"  Rango: {yields.index[0].date()} → {yields.index[-1].date()}")
    print(f"  Bonos ({len(tickers)}): {tickers}")
    print(f"\n  Weights del benchmark:")
    for t, w in sorted(zip(tickers, w_mkt), key=lambda x: -x[1])[:8]:
        print(f"    {t}: {w:.4%}")

    # ── Cargar duration y convexity ──
    print(f"\n  Cargando bond stats (modo: {DURATION_MODE})...")
    duration_df, convexity_df = load_bond_stats(
        str(stats_path), tickers, yields.index, mode=DURATION_MODE
    )

    # ── Views ──
    views = define_bond_views()

    # ── Pipeline ──
    results = run_bond_pipeline(
        yields, tickers, w_mkt, views,
        duration_df, convexity_df,
        duration_mode=DURATION_MODE,
        max_weight=MAX_WEIGHT,
    )

    print("\n✓ Pipeline de bonos completado exitosamente.")
