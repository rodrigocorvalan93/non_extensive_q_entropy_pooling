#%%
"""
s_main_qep_vs_ep.py
===================
¿Cuándo se separa q-EP de EP? Experimento de stress sobre el panel líder.

Con views moderados, los posteriors de Shannon (EP) y de Tsallis con q=2
(q-EP) quedan muy cerca (ver s_main_merval.py). Este script mide dónde se
separan y qué se gana y qué se pierde con cada uno. Usa los mismos datos y
la misma preparación que el pipeline principal (J escenarios diarios, prior
uniforme) y confianza plena (sin mezcla con el prior).

Configuraciones
---------------
1. DUREZA DE UN VIEW SOBRE LA MEDIA (experimento principal)
       E[R_i] >= mu_i + k · se_i,      se_i = sigma_i / sqrt(J),
   con k de 0 a 50 errores estándar. La tabla traduce k a retorno anual
   exigido por encima de la media histórica (k · se_i · 252).

2. RANKING INVERTIDO: ranking completo de los N papeles en el orden
   CONTRARIO al de sus medias muestrales (N−1 desigualdades simultáneas).

3. VIEW DE VOLATILIDAD: sigma posterior = m · sigma del prior, media fija.

4. VIEW DE PROBABILIDAD DE COLA: P(R_i < cuantil 5% del prior) >= pi.
   Para views sobre la probabilidad de un evento, el posterior no depende de
   q (es el condicionamiento de Jeffrey): el script lo comprueba.

5. VARIABILIDAD MUESTRAL (bootstrap): se remuestrean los escenarios con
   reposición, se resuelven los dos posteriors para el mismo view y se mide
   cuánto varían entre remuestras la media y la volatilidad posteriores de
   la cartera equiponderada. Es la contrapartida empírica del argumento de
   Kish: a mayor número efectivo de orden 2, menor varianza muestral.

Diagnósticos por posterior
--------------------------
  - ESS/J   = 1 / (J · Σ p̃²)        número efectivo de orden 2 (Kish)
  - ENS/J   = exp(−Σ p̃ ln p̃) / J    número efectivo de orden 1 (Meucci, 2012)
  - máx     = J · máx_j p̃_j          peso del escenario más cargado, en
                                     múltiplos del peso uniforme
  - ceros   = escenarios con p̃_j = 0 (q=2 trunca; Shannon nunca)
  - CVaR    = CVaR al 5% del papel del view bajo el posterior (% diario)
  - ||Δw||₁ = distancia entre las carteras media-varianza de EP y q-EP
              (long-only, tope 30%, δ = 2,5)

Con prior uniforme, EP maximiza ENS y q-EP (q=2) maximiza ESS: cada posterior
gana en su propia medida. El script informa las dos.

Salidas
-------
  results/stress_dureza.csv, results/stress_otros.csv,
  results/stress_robustez.csv, results/stress_bootstrap.csv
  paper/figs/qep_vs_ep_stress_{es,en}.png

Requisitos: entropy_pooling_v2.py, models.py, covariance_estimation.py y
data/dataacciones/ en el repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from entropy_pooling_v2 import entropy_prog
from models import _mean_variance_optimal
from covariance_estimation import compute_returns_robust, prepare_data_for_ep

DATA_DIR = HERE / "data" / "dataacciones"
RESULTS_DIR = HERE / "results"
FIGS_DIR = HERE / "paper" / "figs"

MAX_WEIGHT = 0.30
DELTA = 2.5
MAX_GAP_FILL = 5            # igual que s_main_merval.py
K_GRID = [0, 1, 2, 5, 10, 20, 35, 50]
STRESS_TICKER = "CEPU"
ROBUST_TICKERS = ["GGAL", "YPFD", "ALUA", "VALO"]
ROBUST_Q = [1.5, 3.0]
ALPHA_CVAR = 0.05
ANNUALIZATION = 252
BOOT_REPS = 500
BOOT_SEED = 20261114
BOOT_K = [0, 5, 10, 20, 35, 50]


# ═══════════════════════════════════════════════════════════════
# Datos y diagnósticos
# ═══════════════════════════════════════════════════════════════

def load_scenarios():
    """Escenarios del pipeline principal: retornos simples, filas completas."""
    px = pd.read_excel(DATA_DIR / "input_mkt_px.xlsx", index_col=0, parse_dates=True).sort_index()
    px.columns = [c.strip() for c in px.columns]
    px = px[sorted(px.columns)]
    rets = compute_returns_robust(px, method="simple", max_gap_fill=MAX_GAP_FILL, verbose=False)
    X, p = prepare_data_for_ep(rets, verbose=False)
    return X, p, list(rets.columns)


def ess_frac(pp: np.ndarray) -> float:
    """Número efectivo de orden 2 (Kish) como fracción de J."""
    return float(1.0 / (pp.size * np.sum(pp ** 2)))


def ens_frac(pp: np.ndarray) -> float:
    """Número efectivo de orden 1 (exp de la entropía; Meucci, 2012) como fracción de J."""
    y = pp[pp > 0]
    return float(np.exp(-np.sum(y * np.log(y))) / pp.size)


def cvar(pp: np.ndarray, r: np.ndarray, alpha: float = ALPHA_CVAR) -> float:
    """CVaR_alpha de r bajo las probabilidades pp: media del peor alpha de masa."""
    order = np.argsort(r)
    w, rs = pp[order], r[order]
    cum = np.cumsum(w)
    k = min(int(np.searchsorted(cum, alpha)), cum.size - 1)
    w_tail = w[: k + 1].copy()
    w_tail[-1] -= cum[k] - alpha          # el último átomo entra en forma parcial
    return float(np.sum(w_tail * rs[: k + 1]) / alpha)


def solve(p, A, b, Aeq, beq, family, q=2.0):
    p_post, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=family, q=q)
    return p_post


def cartera(X, pp):
    mu = X.T @ pp
    second = X.T @ (X * pp[:, None])
    Sigma = 0.5 * (second + second.T) - np.outer(mu, mu)
    return _mean_variance_optimal(mu, Sigma, delta=DELTA, long_only=True,
                                  max_weight=MAX_WEIGHT)


def diagnosticos(X, p, pS, pT, r):
    J = p.size
    wS, wT = cartera(X, pS), cartera(X, pT)
    return {
        "ess_S": ess_frac(pS), "ess_T": ess_frac(pT),
        "ens_S": ens_frac(pS), "ens_T": ens_frac(pT),
        "maxw_S": float(pS.max() * J), "maxw_T": float(pT.max() * J),
        "ceros_T": int(np.sum(pT == 0.0)),
        "cvar_S": cvar(pS, r) * 100, "cvar_T": cvar(pT, r) * 100,
        "dist_S": float(np.linalg.norm(pS - p) / np.linalg.norm(p)),
        "dist_T": float(np.linalg.norm(pT - p) / np.linalg.norm(p)),
        "dw1": float(np.abs(wS - wT).sum()),
    }


# ═══════════════════════════════════════════════════════════════
# Experimentos
# ═══════════════════════════════════════════════════════════════

def barrido_dureza(X, p, tickers, ticker, k_grid=K_GRID, q=2.0):
    """View E[R_i] >= mu_i + k·se_i para cada k."""
    J = p.size
    Aeq, beq = np.ones((1, J)), np.array([1.0])
    r = X[:, tickers.index(ticker)]
    mu0 = float(p @ r)
    sd = float(np.sqrt(p @ (r - mu0) ** 2) * np.sqrt(J / (J - 1)))
    se = sd / np.sqrt(J)
    filas = []
    for k in k_grid:
        A = -r.reshape(1, -1)                 # E[R] >= cota  →  −E[R] <= −cota
        b = np.array([-(mu0 + k * se)])
        pS = solve(p, A, b, Aeq, beq, "S")
        pT = solve(p, A, b, Aeq, beq, "T", q=q)
        fila = {"ticker": ticker, "q": q, "k": k,
                "exceso_anual_pp": k * se * ANNUALIZATION * 100}
        fila.update(diagnosticos(X, p, pS, pT, r))
        filas.append(fila)
    info = {"mu_diaria_pct": mu0 * 100, "sd_diaria_pct": sd * 100, "se_diaria_pct": se * 100,
            "mu_anual_pct": mu0 * ANNUALIZATION * 100, "cvar_prior_pct": cvar(p, r) * 100}
    return pd.DataFrame(filas), info


def ranking_invertido(X, p):
    """Ranking completo en el orden contrario al de las medias muestrales."""
    J, N = X.shape
    Aeq, beq = np.ones((1, J)), np.array([1.0])
    orden = list(np.argsort(X.T @ p))                 # peor → mejor por media
    A = np.vstack([X[:, mejor] - X[:, peor]           # E[peor] >= E[mejor]
                   for peor, mejor in zip(orden[:-1], orden[1:])])
    b = np.zeros(A.shape[0])
    pS = solve(p, A, b, Aeq, beq, "S")
    pT = solve(p, A, b, Aeq, beq, "T", q=2.0)
    d = diagnosticos(X, p, pS, pT, X[:, orden[0]])
    d["n_views"] = A.shape[0]
    return d


def views_volatilidad(X, p, tickers, ticker, multiplos=(1.25, 1.5, 2.0, 3.0)):
    """Volatilidad posterior = m · volatilidad del prior, con la media fija."""
    J = p.size
    r = X[:, tickers.index(ticker)]
    mu0 = float(p @ r)
    var0 = float(p @ (r - mu0) ** 2)
    filas = []
    for m in multiplos:
        Aeq = np.vstack([np.ones(J), r, r ** 2])
        beq = np.array([1.0, mu0, m ** 2 * var0 + mu0 ** 2])
        A, b = np.zeros((0, J)), np.zeros(0)
        pS = solve(p, A, b, Aeq, beq, "S")
        pT = solve(p, A, b, Aeq, beq, "T", q=2.0)
        fila = {"view": "volatilidad", "parametro": m}
        fila.update(diagnosticos(X, p, pS, pT, r))
        fila["max_dif_posteriors_xJ"] = float(np.abs(pS - pT).max() * J)
        filas.append(fila)
    return filas


def views_cola(X, p, tickers, ticker, probs=(0.10, 0.20, 0.30)):
    """P(R_i < cuantil 5% del prior) >= pi: el posterior no depende de q."""
    J = p.size
    Aeq, beq = np.ones((1, J)), np.array([1.0])
    r = X[:, tickers.index(ticker)]
    umbral = np.quantile(r, 0.05)
    ind = (r < umbral).astype(float)
    filas = []
    for pi in probs:
        A, b = -ind.reshape(1, -1), np.array([-pi])
        pS = solve(p, A, b, Aeq, beq, "S")
        pT = solve(p, A, b, Aeq, beq, "T", q=2.0)
        fila = {"view": "cola", "parametro": pi}
        fila.update(diagnosticos(X, p, pS, pT, r))
        fila["max_dif_posteriors_xJ"] = float(np.abs(pS - pT).max() * J)
        # Condicionamiento de Jeffrey: masa pi repartida pareja dentro del evento
        jeffrey = np.where(ind > 0, pi / ind.sum(), (1 - pi) / (J - ind.sum()))
        fila["max_dif_vs_jeffrey_xJ"] = float(max(np.abs(pS - jeffrey).max(),
                                                  np.abs(pT - jeffrey).max()) * J)
        filas.append(fila)
    return filas


def variabilidad_bootstrap(X, p, tickers, ticker, ks=BOOT_K, reps=BOOT_REPS, seed=BOOT_SEED):
    """
    Variabilidad muestral de los momentos del posterior.

    El view es un número fijo (la cota calculada sobre la muestra original).
    En cada réplica se remuestrean las filas de X con reposición, se
    resuelven EP y q-EP (q=2) con prior uniforme y se calculan la media y la
    volatilidad posteriores de la cartera equiponderada. Se informa el
    desvío estándar entre réplicas de la media (en puntos básicos por día) y
    el coeficiente de variación de la volatilidad.
    """
    J, N = X.shape
    rng = np.random.default_rng(seed)
    i = tickers.index(ticker)
    r0 = X[:, i]
    mu0 = float(p @ r0)
    se = float(np.sqrt(p @ (r0 - mu0) ** 2) * np.sqrt(J / (J - 1)) / np.sqrt(J))
    Aeq, beq = np.ones((1, J)), np.array([1.0])
    pu = np.full(J, 1.0 / J)
    w_eq = np.full(N, 1.0 / N)
    muestras = [rng.integers(0, J, size=J) for _ in range(reps)]
    filas = []
    for k in ks:
        cota = mu0 + k * se
        acc = {"S": [], "T": []}
        for idx in muestras:
            Xb = X[idx]
            rb = Xb @ w_eq
            A, b = -Xb[:, i].reshape(1, -1), np.array([-cota])
            for fam in ("S", "T"):
                try:
                    pp = solve(pu, A, b, Aeq, beq, fam, q=2.0)
                except RuntimeError:
                    continue
                m = float(pp @ rb)
                acc[fam].append((m, float(np.sqrt(pp @ (rb - m) ** 2))))
        fila = {"k": k, "replicas": reps}
        for fam in ("S", "T"):
            arr = np.array(acc[fam])
            fila[f"n_{fam}"] = int(arr.shape[0])
            fila[f"sd_media_bp_{fam}"] = float(arr[:, 0].std(ddof=1) * 1e4)
            fila[f"cv_vol_{fam}"] = float(arr[:, 1].std(ddof=1) / arr[:, 1].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def equivalencia_local(X, p, tickers, ticker, ks=(0.25, 0.5, 1.0, 2.0)):
    """La diferencia entre posteriors es de segundo orden en el tamaño del view."""
    J = p.size
    r = X[:, tickers.index(ticker)]
    mu0 = float(p @ r)
    se = float(np.sqrt(p @ (r - mu0) ** 2) * np.sqrt(J / (J - 1)) / np.sqrt(J))
    filas = []
    for k in ks:
        Aeq = np.vstack([np.ones(J), r])
        beq = np.array([1.0, mu0 + k * se])
        A, b = np.zeros((0, J)), np.zeros(0)
        pS = solve(p, A, b, Aeq, beq, "S")
        pT = solve(p, A, b, Aeq, beq, "T", q=2.0)
        filas.append({"k": k,
                      "desplazamiento_xJ": float(np.abs(pT - p).max() * J),
                      "dif_S_T_xJ": float(np.abs(pS - pT).max() * J),
                      "dif_sobre_k2": float(np.abs(pS - pT).max() * J / k ** 2)})
    return pd.DataFrame(filas)


# ═══════════════════════════════════════════════════════════════
# Figura
# ═══════════════════════════════════════════════════════════════

TEXTOS = {
    "es": {
        "x": "dureza del view  k  (errores estándar)",
        "y1": "fracción de escenarios efectivos",
        "t1": "Escenarios efectivos",
        "y2": "CVaR 5% del papel estresado (% diario)",
        "t2": "Cola izquierda del posterior",
        "y3": r"$\|w_{q=2} - w_{\mathrm{EP}}\|_1$",
        "t3": "Distancia entre carteras",
        "kish": "orden 2 (Kish)", "ens": "orden 1 (entropía)",
        "ep": "EP (Shannon)", "qep": "q = 2", "prior": "prior",
        "rank": "ranking invertido",
    },
    "en": {
        "x": "view hardness  k  (standard errors)",
        "y1": "effective fraction of scenarios",
        "t1": "Effective scenarios",
        "y2": "5% CVaR of the stressed stock (% per day)",
        "t2": "Left tail of the posterior",
        "y3": r"$\|w_{q=2} - w_{\mathrm{EP}}\|_1$",
        "t3": "Distance between portfolios",
        "kish": "order 2 (Kish)", "ens": "order 1 (entropy)",
        "ep": "EP (Shannon)", "qep": "q = 2", "prior": "prior",
        "rank": "reversed ranking",
    },
}
C_EP, C_QEP = "#E8475A", "#2E86AB"


def figura(df, info, rank, lang, out):
    tx = TEXTOS[lang]
    ks = df["k"].values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax1.plot(ks, df["ess_S"], "o-", color=C_EP, label=f'{tx["ep"]}, {tx["kish"]}')
    ax1.plot(ks, df["ess_T"], "s-", color=C_QEP, label=f'{tx["qep"]}, {tx["kish"]}')
    ax1.plot(ks, df["ens_S"], "o--", color=C_EP, mfc="white", label=f'{tx["ep"]}, {tx["ens"]}')
    ax1.plot(ks, df["ens_T"], "s--", color=C_QEP, mfc="white", label=f'{tx["qep"]}, {tx["ens"]}')
    ax1.set_xlabel(tx["x"]); ax1.set_ylabel(tx["y1"]); ax1.set_title(tx["t1"])
    ax1.set_ylim(0, 1.05); ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    ax2.axhline(info["cvar_prior_pct"], color="gray", ls=":", lw=1.2, label=tx["prior"])
    ax2.plot(ks, df["cvar_S"], "o-", color=C_EP, label=tx["ep"])
    ax2.plot(ks, df["cvar_T"], "s-", color=C_QEP, label=tx["qep"])
    ax2.set_xlabel(tx["x"]); ax2.set_ylabel(tx["y2"]); ax2.set_title(tx["t2"])
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    ax3.plot(ks, df["dw1"], "D-", color="#4A6A3A")
    ax3.axhline(rank["dw1"], color="#4A6A3A", ls=":", lw=1.2, label=tx["rank"])
    ax3.set_xlabel(tx["x"]); ax3.set_ylabel(tx["y3"]); ax3.set_title(tx["t3"])
    ax3.legend(fontsize=8); ax3.grid(alpha=.3)

    if lang == "es":                          # coma decimal en los ejes
        coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        for ax in (ax1, ax2, ax3):
            ax.yaxis.set_major_formatter(coma)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {out}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    X, p, tickers = load_scenarios()
    J, N = X.shape
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    df, info = barrido_dureza(X, p, tickers, STRESS_TICKER)
    print(f"Panel: {J} escenarios × {N} activos  |  stress sobre {STRESS_TICKER}: "
          f"media {info['mu_diaria_pct']:.4f}%/día ({info['mu_anual_pct']:.1f}%/año), "
          f"desvío {info['sd_diaria_pct']:.3f}%, error estándar {info['se_diaria_pct']:.4f}%, "
          f"CVaR 5% del prior {info['cvar_prior_pct']:.2f}%")
    cols = ["k", "exceso_anual_pp", "ess_S", "ess_T", "ens_S", "ens_T", "maxw_S", "maxw_T",
            "ceros_T", "cvar_S", "cvar_T", "dw1"]
    print("\n1) Dureza de un view sobre la media")
    print(df[cols].to_string(index=False))

    rank = ranking_invertido(X, p)
    print(f"\n2) Ranking invertido ({rank['n_views']} desigualdades contra el prior): "
          f"ESS/J S={rank['ess_S']:.3f} T={rank['ess_T']:.3f} | "
          f"ENS/J S={rank['ens_S']:.3f} T={rank['ens_T']:.3f} | "
          f"distorsión S={rank['dist_S']:.3f} T={rank['dist_T']:.3f} | "
          f"ceros (T): {rank['ceros_T']}/{J} | ||Δw||₁ = {rank['dw1']:.3f}")

    otros = pd.DataFrame(views_volatilidad(X, p, tickers, STRESS_TICKER)
                         + views_cola(X, p, tickers, STRESS_TICKER))
    print("\n3) y 4) Views de volatilidad y de probabilidad de cola")
    print(otros[["view", "parametro", "ess_S", "ess_T", "ens_S", "ens_T", "ceros_T",
                 "max_dif_posteriors_xJ", "max_dif_vs_jeffrey_xJ"]].to_string(index=False))

    loc = equivalencia_local(X, p, tickers, STRESS_TICKER)
    print("\n5) Equivalencia local: la diferencia entre posteriors escala con k²")
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        print(loc.to_string(index=False))

    # Robustez: otros papeles y otros q, en k = 20 y k = 50
    rob = []
    for t in [STRESS_TICKER] + ROBUST_TICKERS:
        d, _ = barrido_dureza(X, p, tickers, t, k_grid=[20, 50])
        rob.append(d)
    for q in ROBUST_Q:
        d, _ = barrido_dureza(X, p, tickers, STRESS_TICKER, k_grid=[20, 50], q=q)
        rob.append(d)
    rob = pd.concat(rob, ignore_index=True)
    print("\n6) Robustez (otros papeles con q=2; otros q con el papel base)")
    print(rob[["ticker", "q", "k", "ess_S", "ess_T", "ens_S", "ens_T", "ceros_T",
               "cvar_S", "cvar_T", "dw1"]].to_string(index=False))

    boot = variabilidad_bootstrap(X, p, tickers, STRESS_TICKER)
    print(f"\n7) Variabilidad muestral de los momentos posteriores de la cartera equiponderada "
          f"({BOOT_REPS} remuestras bootstrap)")
    print(boot.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    boot.to_csv(RESULTS_DIR / "stress_bootstrap.csv", index=False, float_format="%.6f")
    df.to_csv(RESULTS_DIR / "stress_dureza.csv", index=False, float_format="%.6f")
    pd.DataFrame([{**rank, **{"J": J, "N": N}, **info}]).to_csv(
        RESULTS_DIR / "stress_ranking_e_info.csv", index=False, float_format="%.6f")
    otros.to_csv(RESULTS_DIR / "stress_otros.csv", index=False, float_format="%.6g")
    loc.to_csv(RESULTS_DIR / "stress_equivalencia_local.csv", index=False, float_format="%.6g")
    rob.to_csv(RESULTS_DIR / "stress_robustez.csv", index=False, float_format="%.6f")
    print(f"\nTablas guardadas en: {RESULTS_DIR}")

    for lang in ("es", "en"):
        figura(df, info, rank, lang, FIGS_DIR / f"qep_vs_ep_stress_{lang}.png")


if __name__ == "__main__":
    main()
