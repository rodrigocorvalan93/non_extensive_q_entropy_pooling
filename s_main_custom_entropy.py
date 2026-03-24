#%%
"""
s_main_custom_entropy.py
------------------------

Versión de s_main.py donde la *familia de entropía* (Shannon/Tsallis/Rényi)
se elige desde el main (sin tocar entropy_pooling.py).

- Carga ReturnsDistribution.mat (X, p)
- Calcula frontera eficiente prior
- Impone una view de ranking: E[R_lower] <= E[R_upper]
- Obtiene p_view vía EntropyProg con la familia elegida
- Mezcla con confianza c y recalcula frontera eficiente

Uso:
- Terminal:  python s_main_custom_entropy.py
- Jupyter:   %run s_main_custom_entropy.py
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from entropy_pooling_v2 import EfficientFrontier, PlotResults, EntropyProg, FrontierOptions


# =====================================================
# 1) Acá se puede elegir la familia de entropía y sus parámetros:
# =====================================================
# 'S' = Shannon (KL)
# 'T' = Tsallis (requiere q != 1)
# 'R' = Rényi   (requiere q != 1)
# 'G' = General (requiere q != 1 y funciones g, dg)
entropy_family = "S"
q = 1

# confianza de la view (0 = no view, 1 = todo view)
c = 0.5

# view: Lower tiene retorno esperado menor que Upper
Lower = [4]  # índices estilo MATLAB (1-based)
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


# -----------------------------
# 2) Load X y p
# -----------------------------
HERE = Path(__file__).resolve().parent
mat_path = HERE / "ReturnsDistribution.mat"
mat = loadmat(mat_path)

X = np.asarray(mat["X"], dtype=float)
p = np.asarray(mat["p"], dtype=float).reshape(-1)
p = p / p.sum()

J, N = X.shape

# -----------------------------
# 3) Frontera prior
# -----------------------------
Options = FrontierOptions(NumPortf=20, FrontierSpan=(0.3, 0.9))

e, s, w, M, S = EfficientFrontier(X, p, Options)
PlotResults(e, s, w, M)
plt.gcf().suptitle("PRIOR", y=0.98)

# -----------------------------
# 4) Ranking view -> constraints
# -----------------------------
lower_idx = matlab_to_python(Lower, N)
upper_idx = matlab_to_python(Upper, N)

V = X[:, lower_idx] - X[:, upper_idx]  # (J,K)
A = V.T                                # (K,J)
b = np.zeros(A.shape[0])

Aeq = np.ones((1, J))
beq = np.array([1.0])

# posterior de la view según la familia elegida
# EntropyProg devuelve: p_view, Lx, lv
p_view, _, lv = EntropyProg(p, A, b, Aeq, beq, entropy_family=entropy_family, q=q)

# mezcla con confianza
p_post = (1 - c) * p + c * p_view
p_post = p_post / p_post.sum()

# -----------------------------
# 5) Frontera posterior
# -----------------------------
e2, s2, w2, M2, S2 = EfficientFrontier(X, p_post, Options)
PlotResults(e2, s2, w2, M2, Lower, Upper)
plt.gcf().suptitle(f"POSTERIOR | entropy_family={entropy_family}, q={q}, c={c}", y=0.98)

# -----------------------------
# 6) Extra: curva riesgo-retorno (la "frontera" clásica)
# -----------------------------
plt.figure()
plt.plot(s * 100, e * 100, marker="o", label="prior")
plt.plot(s2 * 100, e2 * 100, marker="o", label="posterior")
plt.grid(True)
plt.xlabel("Volatilidad (%)")
plt.ylabel("Retorno esperado (%)")
plt.title("Frontera eficiente (riesgo-retorno)")
plt.legend()

# -----------------------------
# 7) Extra: comparar p prior vs posterior
# -----------------------------
plt.figure()
plt.plot(p, label="p (prior)")
plt.plot(p_post, label="p (posterior)")
plt.title("Probabilidades por escenario")
plt.grid(True)
plt.legend()

plt.show()
