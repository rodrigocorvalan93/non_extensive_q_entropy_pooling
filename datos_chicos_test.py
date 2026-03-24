#%%
"""
datos_chicos_test.py
--------------------

Traducción a Python de DatosChicos.m para "testear" EntropyProg contra
una solución obtenida resolviendo las ecuaciones KKT por Newton-Raphson
(con Jacobiano aproximado, como en el script Octave).

Qué compara:
- "EntropyProg" (dual, como en el paper/código)
vs
- "Newton-KKT" (resolver gradiente del Lagrangiano = 0 en todas las variables)

Uso:
- Terminal: python datos_chicos_test.py
- Jupyter:  %run datos_chicos_test.py
"""

import numpy as np

from entropy_pooling_v2 import EntropyProg


# ============================================================
# Helpers numéricos (equivalentes a JACOB_APROX / NR_MULTI)
# ============================================================

def jacob_scalar(L, X, epsi=None):
    """
    Aproxima el gradiente de un escalar L(X) vía diferencias finitas centradas.
    Devuelve un vector del mismo tamaño que X.
    """
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
    """
    Aproxima el Jacobiano de un vector H(X) vía diferencias finitas centradas.
    Devuelve matriz (m,n).
    """
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
    """
    Newton-Raphson multivariado para resolver H(X)=0, usando Jacobiano aproximado.
    """
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(niter):
        fx = np.asarray(H(x), dtype=float).reshape(-1)
        J = jacob_vec(H, x)
        step = np.linalg.solve(J, fx)
        x = x - step
    return x


# ============================================================
# Datos chicos (igual que DatosChicos.m)
# ============================================================

p = np.array([0.3, 0.2, 0.1, 0.4], dtype=float)

# desigualdad A x <= b
A = np.array([[2.0, 3.0, 0.5, 1.0]], dtype=float)
b = np.array([1.3], dtype=float)

# igualdades Aeq x = beq
Aeq = np.array([[1.5, 0.5, 0.8, 0.7],
                [1.0, 1.0, 1.0, 1.0]], dtype=float)
beq = np.array([0.78, 1.0], dtype=float)

# Parámetros q
q_T = 1.5
q_R = 1.5  #  0.8 si se quiere replicar exactamente el LRenyi del .m


# ============================================================
# Lagrangianos (para Newton-KKT)
# Nota: el vector X que resuelve Newton es:
#   X = [x1..x4, v1..vK, l1..lK_]
# En este ejemplo: K=2, K_=1  ->  tamaño 7
# ============================================================

def L_shannon(X):
    x = np.maximum(X[:4], 1e-32)     # evita log(<=0)
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


# Gradientes para Newton-KKT (como GRADL = JACOB_APROX(L, X)')
GRAD_shannon = lambda X: jacob_scalar(L_shannon, X)
GRAD_tsallis = lambda X: jacob_scalar(L_tsallis, X)
GRAD_renyi = lambda X: jacob_scalar(L_renyi, X)


# ============================================================
# Newton-KKT ("a mano")
# ============================================================

x0 = np.concatenate([p, [5.0, 4.0, -2.0]])  # mismo arranque que en DatosChicos.m

PC = newton_multi(GRAD_shannon, x0, niter=25)
PCT = newton_multi(GRAD_tsallis, x0, niter=25)
PCR = newton_multi(GRAD_renyi, x0, niter=25)

p_sh_newton = PC[:4]
p_t_newton = PCT[:4]
p_r_newton = PCR[:4]


# ============================================================
# EntropyProg
# ============================================================

p_sh, _, _ = EntropyProg(p, A, b, Aeq, beq, entropy_family="S", q=1.0)
p_t, _, _  = EntropyProg(p, A, b, Aeq, beq, entropy_family="T", q=q_T)
p_r, _, _  = EntropyProg(p, A, b, Aeq, beq, entropy_family="R", q=q_R)


# ============================================================
# Comparación
# ============================================================

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

print("\nChequeo constraints (con posterior EntropyProg):")
print("A x - b (debe ser <=0):", float((A @ p_r - b).item()))
print("Aeq x - beq (debe ser 0):", Aeq.dot(p_r) - beq)

