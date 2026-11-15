"""
Pruebas de propiedades del solver de Entropy Pooling generalizado.

Cada prueba corresponde a una afirmación del paper:
  - el posterior es una distribución y cumple los views;
  - con prior uniforme, q = 2 es la proyección euclídea del prior
    (se compara con un QP independiente si cvxpy está instalado);
  - cada q maximiza el número efectivo de escenarios de su propio orden;
  - Rényi y Tsallis dan el mismo posterior (primal de Rényi con SLSQP);
  - Tsallis tiende a Shannon cuando q → 1;
  - a primer orden en el tamaño del view todos los q coinciden;
  - un view sobre la probabilidad de un evento da el condicionamiento de
    Jeffrey para cualquier q;
  - un conjunto de views infactible produce un error claro.

Correr con:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entropy_pooling_v2 import entropy_prog  # noqa: E402


def problema(J=400, N=3, seed=0, shift=0.5):
    """Escenarios con colas pesadas, prior uniforme y un view E[x1] >= media + shift·se."""
    rng = np.random.default_rng(seed)
    X = 0.02 * rng.standard_t(df=4, size=(J, N))
    p = np.full(J, 1.0 / J)
    r = X[:, 0]
    cota = r.mean() + shift * r.std(ddof=1) / np.sqrt(J)
    A, b = -r.reshape(1, -1), np.array([-cota])
    Aeq, beq = np.ones((1, J)), np.array([1.0])
    return X, p, A, b, Aeq, beq


def n_efectivo(x, orden):
    y = x[x > 0]
    return np.exp(-np.sum(y * np.log(y))) if orden == 1 else 1.0 / np.sum(y ** 2)


@pytest.mark.parametrize("familia,q", [("S", 1.0), ("T", 1.5), ("T", 2.0), ("T", 3.0)])
@pytest.mark.parametrize("shift", [0.5, 5.0, 30.0])
def test_posterior_factible(familia, q, shift):
    X, p, A, b, Aeq, beq = problema(shift=shift)
    x, _, lv = entropy_prog(p, A, b, Aeq, beq, entropy_family=familia, q=q)
    assert np.all(x >= 0.0)
    assert abs(x.sum() - 1.0) < 1e-12
    assert np.max(A @ x - b) < 1e-10          # el view se cumple
    assert lv[0] >= 0.0                       # multiplicador de la desigualdad
    assert abs(lv[0] * (b - A @ x)[0]) < 1e-9  # complementariedad


def test_view_no_activo_deja_el_prior():
    X, p, A, b, Aeq, beq = problema(shift=-1.0)   # el prior ya cumple el view
    for familia, q in (("S", 1.0), ("T", 2.0)):
        x, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=familia, q=q)
        assert np.abs(x - p).max() < 1e-12


@pytest.mark.parametrize("shift", [0.5, 5.0, 30.0])
def test_q2_es_la_proyeccion_euclidea(shift):
    cp = pytest.importorskip("cvxpy")
    X, p, A, b, Aeq, beq = problema(shift=shift)
    x2, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=2.0)
    v = cp.Variable(p.size, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(v - p)),
               [A @ v <= b, cp.sum(v) == 1]).solve(solver=cp.OSQP, eps_abs=1e-12,
                                                   eps_rel=1e-12, max_iter=200_000)
    assert np.abs(x2 - np.maximum(v.value, 0.0)).max() < 1e-8


@pytest.mark.parametrize("shift", [2.0, 20.0])
def test_cada_q_maximiza_su_numero_efectivo(shift):
    X, p, A, b, Aeq, beq = problema(shift=shift)
    xS, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S")
    x2, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=2.0)
    x3, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=3.0)
    # orden 1 (entropía): gana Shannon; orden 2 (Kish): gana q = 2
    assert n_efectivo(xS, 1) >= max(n_efectivo(x2, 1), n_efectivo(x3, 1))
    assert n_efectivo(x2, 2) >= max(n_efectivo(xS, 2), n_efectivo(x3, 2))
    # y la distancia euclídea al prior es mínima en q = 2
    d = lambda x: np.linalg.norm(x - p)
    assert d(x2) <= min(d(xS), d(x3)) + 1e-15


@pytest.mark.parametrize("q", [1.5, 2.0, 2.5])
def test_renyi_y_tsallis_dan_el_mismo_posterior(q):
    X, p, A, b, Aeq, beq = problema(J=50, shift=3.0, seed=1)
    xT, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)

    def renyi(x):
        x = np.maximum(x, 1e-300)
        return np.log(np.sum(p * (x / p) ** q)) / (q - 1.0)

    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
            {"type": "ineq", "fun": lambda x: b - A @ x}]
    sol = minimize(renyi, p.copy(), method="SLSQP", bounds=[(0.0, 1.0)] * p.size,
                   constraints=cons, options={"ftol": 1e-15, "maxiter": 2000})
    assert np.abs(sol.x - xT).max() < 1e-6
    # y la rama Rényi del propio solver coincide con Tsallis
    xR, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="R", q=q)
    assert np.abs(xR - xT).max() < 1e-9


def test_limite_q_a_1_es_shannon():
    X, p, A, b, Aeq, beq = problema(shift=3.0)
    xS, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S")
    difs = []
    for q in (1.1, 1.01, 1.001):
        xq, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
        difs.append(np.abs(xq - xS).max())
    assert difs[0] > difs[1] > difs[2]
    assert difs[2] < 1e-2 * np.abs(xS - p).max()


def test_equivalencia_de_primer_orden():
    """La diferencia entre Shannon y q = 2 escala con el cuadrado del tamaño del view."""
    rng = np.random.default_rng(3)
    J = 500
    r = 0.02 * rng.standard_t(df=5, size=J)
    p = np.full(J, 1.0 / J)
    se = r.std(ddof=1) / np.sqrt(J)
    Aeq = np.vstack([np.ones(J), r])
    A, b = np.zeros((0, J)), np.zeros(0)
    ratios = []
    for k in (0.25, 0.5, 1.0):
        beq = np.array([1.0, r.mean() + k * se])
        xS, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S")
        x2, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=2.0)
        assert np.abs(xS - x2).max() < 0.2 * np.abs(x2 - p).max()
        ratios.append(np.abs(xS - x2).max() / k ** 2)
    assert max(ratios) / min(ratios) < 1.3


@pytest.mark.parametrize("pi", [0.10, 0.30])
def test_view_de_probabilidad_es_jeffrey_para_todo_q(pi):
    X, p, _, _, Aeq, beq = problema(J=400)
    r = X[:, 0]
    ind = (r < np.quantile(r, 0.05)).astype(float)
    A, b = -ind.reshape(1, -1), np.array([-pi])          # P(evento) >= pi
    jeffrey = np.where(ind > 0, pi / ind.sum(), (1 - pi) / (p.size - ind.sum()))
    for familia, q in (("S", 1.0), ("T", 1.5), ("T", 2.0), ("T", 3.0)):
        x, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=familia, q=q)
        assert np.abs(x - jeffrey).max() < 1e-10


def test_invariancia_al_reordenar_escenarios():
    X, p, A, b, Aeq, beq = problema(shift=4.0)
    perm = np.random.default_rng(7).permutation(p.size)
    for familia, q in (("S", 1.0), ("T", 2.0)):
        x, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=familia, q=q)
        xp, _, _ = entropy_prog(p[perm], A[:, perm], b, Aeq[:, perm], beq,
                                entropy_family=familia, q=q)
        assert np.abs(x[perm] - xp).max() < 1e-10


def test_views_infactibles_dan_error_claro():
    X, p, A, b, Aeq, beq = problema()
    r = X[:, 0]
    A2 = np.vstack([-r, r])                              # E[r] >= 1 y E[r] <= -1
    b2 = np.array([-1.0, -1.0])
    for familia, q in (("S", 1.0), ("T", 2.0)):
        with pytest.raises(RuntimeError, match="no convergi"):
            entropy_prog(p, A2, b2, Aeq, beq, entropy_family=familia, q=q)


def test_prior_no_uniforme_q2_es_chi_cuadrado():
    """Con prior general, q = 2 minimiza Σ (x_j − p_j)² / p_j."""
    cp = pytest.importorskip("cvxpy")
    X, _, A, b, Aeq, beq = problema(J=300, shift=6.0)
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.full(300, 5.0))
    x2, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=2.0)
    v = cp.Variable(p.size, nonneg=True)
    obj = cp.sum(cp.multiply(cp.square(v - p), 1.0 / p))
    cp.Problem(cp.Minimize(obj), [A @ v <= b, cp.sum(v) == 1]).solve(
        solver=cp.CLARABEL)
    assert np.abs(x2 - np.maximum(v.value, 0.0)).max() < 1e-6


@pytest.mark.parametrize("q", [4.0, 6.0, 10.0])
def test_orden_alto_cumple_el_view_y_coincide_con_la_forma_cerrada(q):
    """
    Para q grande los escenarios truncados tienen que valer exactamente cero
    también dentro del solver: con un piso positivo su masa deja de ser
    despreciable y el posterior devuelto no cumple el view. Se compara con la
    forma cerrada x_j ∝ max(a + r_j, 0)^(1/(q−1)), resuelta por bisección.
    """
    from scipy.optimize import brentq

    X, p, A, b, Aeq, beq = problema(J=400, shift=25.0)
    r, cota = X[:, 0], -b[0]
    x, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
    assert abs(x.sum() - 1.0) < 1e-12
    assert abs(x @ r - cota) < 1e-10 * abs(cota)

    def forma_cerrada(a):
        y = np.maximum(a + r, 0.0) ** (1.0 / (q - 1.0))
        return y / y.sum()

    a = brentq(lambda t: forma_cerrada(t) @ r - cota, -r.max() + 1e-12, 10.0 * np.abs(r).max(),
               xtol=1e-16, rtol=1e-15)
    assert np.abs(x - forma_cerrada(a)).max() < 1e-9
    assert np.array_equal(x == 0.0, forma_cerrada(a) == 0.0)


def test_solo_shannon_conserva_la_independencia():
    """
    Independencia de sistemas (Shore y Johnson): con un prior producto y un
    view sobre cada marginal, el posterior de Shannon es el producto de sus
    marginales; el de q = 2 no.
    """
    rng = np.random.default_rng(5)
    a, c = rng.normal(size=12), rng.normal(size=10)
    pa, pc = rng.dirichlet(np.full(12, 4.0)), rng.dirichlet(np.full(10, 4.0))
    p = np.outer(pa, pc).ravel()
    fa, fc = np.repeat(a, 10), np.tile(c, 12)
    Aeq = np.vstack([np.ones(p.size), fa, fc])
    beq = np.array([1.0, pa @ a + 0.4 * a.std(), pc @ c - 0.4 * c.std()])
    A, b = np.zeros((0, p.size)), np.zeros(0)

    def desvio_del_producto(familia, q):
        x, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=familia, q=q)
        x = x.reshape(12, 10)
        return np.abs(x - np.outer(x.sum(axis=1), x.sum(axis=0))).max()

    assert desvio_del_producto("S", 1.0) < 1e-12
    assert desvio_del_producto("T", 2.0) > 1e-4


def test_media_varianza_coincide_con_un_qp_exacto():
    """Los pesos de media-varianza no dependen de la tolerancia del optimizador."""
    cp = pytest.importorskip("cvxpy")
    from models import _mean_variance_optimal

    rng = np.random.default_rng(3)
    N = 12
    R = 0.02 * rng.standard_t(df=5, size=(900, N))
    mu, S = R.mean(axis=0) + 0.002, np.cov(R.T)
    w = _mean_variance_optimal(mu, S, delta=2.5, long_only=True, max_weight=0.30)
    v = cp.Variable(N)
    cp.Problem(cp.Maximize(mu @ v - 1.25 * cp.quad_form(v, cp.psd_wrap(S))),
               [cp.sum(v) == 1, v >= 0, v <= 0.30]).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13)
    assert np.abs(w - np.asarray(v.value)).sum() < 1e-5


def test_q_menor_que_uno_rechaza_desigualdades():
    """La rama q < 1 no impone la complementariedad: con desigualdades tiene que avisar."""
    X, p, A, b, Aeq, beq = problema(J=6)
    with pytest.raises(NotImplementedError, match="igualdad"):
        entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=0.5)
