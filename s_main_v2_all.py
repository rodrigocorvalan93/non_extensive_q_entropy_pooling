"""
s_main_v2_all.py
----------------

"Main" estilo Jupyter con DOS bloques:

1) Ranking Information (como S_MAIN.m)
2) Test "Datos chicos" (como DatosChicos.m) una versión toy para testear que
 el solver de EntropyProg da lo mismo que Newton-KKT (con gradiente numérico) en un caso pequeño.

Requisitos:
- entropy_pooling_v2.py en la misma carpeta
- ReturnsDistribution.mat en la misma carpeta (solo para el bloque 1)

Uso:
- Terminal:  python s_main_v2_all.py
- Jupyter:   %run s_main_v2_all.py
"""

#%% Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from entropy_pooling_v2 import EfficientFrontier, PlotResults, EntropyProg, FrontierOptions


#%% Switches (prende/apaga bloques)
RUN_RANKING_DEMO = True
RUN_DATOS_CHICOS = True


#%% Bloque 1: Ranking Information (S_MAIN.m)
if RUN_RANKING_DEMO:
    # Elegí la familia de entropía acá
    entropy_family = "T"   # "S", "T", "R"
    q = 1.5

    # confianza de la view
    c = 0.5

    # view: Lower tiene retorno esperado menor que Upper
    Lower = [4]  # 1-based estilo MATLAB
    Upper = [3]

    def matlab_to_python(idxs, N):
        idxs = np.asarray(idxs, dtype=int).ravel()
        if idxs.size == 0:
            return idxs
        if np.any(idxs == 0):
            out = idxs
        else:
            out = idxs - 1
        if np.any(out < 0) or np.any(out >= N):
            raise IndexError(f"Índices fuera de rango: {idxs} con N={N}")
        return out

    HERE = Path(__file__).resolve().parent
    mat_path = HERE / "ReturnsDistribution.mat"
    mat = loadmat(mat_path)

    X = np.asarray(mat["X"], dtype=float)
    p = np.asarray(mat["p"], dtype=float).reshape(-1)
    p = p / p.sum()

    J, N = X.shape

    # frontera prior
    Options = FrontierOptions(NumPortf=20, FrontierSpan=(0.3, 0.9))
    e, s, w, M, S = EfficientFrontier(X, p, Options)
    PlotResults(e, s, w, M)
    plt.gcf().suptitle("PRIOR", y=0.98)

    # constraints de ranking
    lower_idx = matlab_to_python(Lower, N)
    upper_idx = matlab_to_python(Upper, N)

    V = X[:, lower_idx] - X[:, upper_idx]
    A = V.T
    b = np.zeros(A.shape[0])

    Aeq = np.ones((1, J))
    beq = np.array([1.0])

    # posterior view
    p_view, _, lv = EntropyProg(p, A, b, Aeq, beq, entropy_family=entropy_family, q=q)

    # mezcla con confianza
    p_post = (1 - c) * p + c * p_view
    p_post = p_post / p_post.sum()

    # frontera posterior
    e2, s2, w2, M2, S2 = EfficientFrontier(X, p_post, Options)
    PlotResults(e2, s2, w2, M2, Lower, Upper)
    plt.gcf().suptitle(f"POSTERIOR | entropy_family={entropy_family}, q={q}, c={c}", y=0.98)

    # frontera riesgo-retorno clásica
    plt.figure()
    plt.plot(s * 100, e * 100, marker="o", label="prior")
    plt.plot(s2 * 100, e2 * 100, marker="o", label="posterior")
    plt.grid(True)
    plt.xlabel("Volatilidad (%)")
    plt.ylabel("Retorno esperado (%)")
    plt.title("Frontera eficiente (riesgo-retorno)")
    plt.legend()

    plt.show()


#%% Bloque 2: Datos chicos (DatosChicos.m)
if RUN_DATOS_CHICOS:
    # ---- Helpers numéricos (equivalentes a JACOB_APROX / NR_MULTI)
    def jacob_scalar(L, X, epsi=None):
        X = np.asarray(X, dtype=float)
        if epsi is None:
            epsi = float(np.sqrt(np.sqrt(np.finfo(float).eps * (1.0 + np.linalg.norm(X)))))
        grad = np.zeros_like(X, dtype=float)
        for k in range(X.size):
            e = np.zeros_like(X)
            e[k] = 1.0
            grad[k] = (L(X + epsi * e) - L(X - epsi * e)) / (2.0 * epsi)
        return grad

    def jacob_vec(H, X, epsi=None):
        X = np.asarray(X, dtype=float)
        if epsi is None:
            epsi = float(np.sqrt(np.sqrt(np.finfo(float).eps * (1.0 + np.linalg.norm(X)))))
        H0 = np.asarray(H(X), dtype=float).reshape(-1)
        m = H0.size
        n = X.size
        J = np.zeros((m, n), dtype=float)
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1.0
            J[:, k] = (
                np.asarray(H(X + epsi * e), dtype=float).reshape(-1)
                - np.asarray(H(X - epsi * e), dtype=float).reshape(-1)
            ) / (2.0 * epsi)
        return J

    def newton_multi(H, x0, niter=25):
        x = np.asarray(x0, dtype=float).copy()
        for i in range(niter):
            fx = np.asarray(H(x), dtype=float).reshape(-1)
            J = jacob_vec(H, x)
            # FIX: si el Jacobiano es singular (frecuente cerca de puntos silla
            # o en constraints activas), np.linalg.solve levanta LinAlgError.
            # Fallback a mínimos cuadrados para continuar iterando.
            try:
                step = np.linalg.solve(J, fx)
            except np.linalg.LinAlgError:
                step, _, _, _ = np.linalg.lstsq(J, fx, rcond=None)
            x = x - step
        return x

    # ---- Datos
    p = np.array([0.3, 0.2, 0.1, 0.4], dtype=float)

    A = np.array([[2.0, 3.0, 0.5, 1.0]], dtype=float)
    b = np.array([1.3], dtype=float)

    Aeq = np.array([[1.5, 0.5, 0.8, 0.7],
                    [1.0, 1.0, 1.0, 1.0]], dtype=float)
    beq = np.array([0.78, 1.0], dtype=float)

    q_T = 1.5
    q_R = 1.5

    # ---- Lagrangianos
    def L_shannon(X):
        x = np.maximum(X[:4], 1e-32)
        v = X[4:6]
        l = X[6:7]
        return float(
            x.dot(np.log(x) - np.log(p))
            + v.dot(Aeq.dot(x) - beq)
            + l.dot(A.dot(x) - b)
        )

    def L_tsallis(X):
        x = np.maximum(X[:4], 1e-32)
        v = X[4:6]
        l = X[6:7]
        t = float(((x / p) ** q_T).dot(p))
        return float(
            (1.0 / (q_T - 1.0)) * (t - 1.0)
            + v.dot(Aeq.dot(x) - beq)
            + l.dot(A.dot(x) - b)
        )

    def L_renyi(X):
        x = np.maximum(X[:4], 1e-32)
        v = X[4:6]
        l = X[6:7]
        t = float(((x / p) ** q_R).dot(p))
        return float(
            (1.0 / (q_R - 1.0)) * np.log(t)
            + v.dot(Aeq.dot(x) - beq)
            + l.dot(A.dot(x) - b)
        )

    GRAD_shannon = lambda X: jacob_scalar(L_shannon, X)
    GRAD_tsallis = lambda X: jacob_scalar(L_tsallis, X)
    GRAD_renyi = lambda X: jacob_scalar(L_renyi, X)

    # ---- Newton-KKT
    x0 = np.concatenate([p, [5.0, 4.0, -2.0]])
    PC = newton_multi(GRAD_shannon, x0, niter=25)
    PCT = newton_multi(GRAD_tsallis, x0, niter=25)
    PCR = newton_multi(GRAD_renyi, x0, niter=25)

    p_sh_newton = PC[:4]
    p_t_newton = PCT[:4]
    p_r_newton = PCR[:4]

    # ---- EntropyProg
    p_sh, _, _ = EntropyProg(p, A, b, Aeq, beq, entropy_family="S", q=1.0)
    p_t, _, _  = EntropyProg(p, A, b, Aeq, beq, entropy_family="T", q=q_T)
    p_r, _, _  = EntropyProg(p, A, b, Aeq, beq, entropy_family="R", q=q_R)

    # ---- Print comparaciones
    np.set_printoptions(precision=10, suppress=True)
    print("\n=== Datos chicos: comparación EntropyProg vs Newton-KKT ===")

    def show_compare(name, p_entropy, p_newton):
        print(f"\n{name}")
        print("EntropyProg:", p_entropy)
        print("Newton-KKT :", p_newton)
        print("||diff||_2 :", float(np.linalg.norm(p_entropy - p_newton)))

    show_compare("Shannon (KL)", p_sh, p_sh_newton)
    show_compare(f"Tsallis (q={q_T})", p_t, p_t_newton)
    show_compare(f"Rényi (q={q_R})", p_r, p_r_newton)

    print("\nChequeo constraints (con posterior EntropyProg Rényi):")
    print("A x - b (<=0):", float(A.dot(p_r) - b))
    print("Aeq x - beq (=0):", Aeq.dot(p_r) - beq)