#%%
"""
Entropy Pooling + Efficient Frontier
------------------------------------
Traducción a Python (NumPy/SciPy/Matplotlib) del set de funciones MATLAB:

- EfficientFrontier.m
- EntropyProg.m
- ImpliedExpRets.m
- PlotFrontier.m
- PlotResults.m
- RobustEfficientFrontier.m
- ViewRanking.m

Pensado para ejecutarse desde un s_main.py estilo "notebook" (o en Jupyter con %run).

Notas:
- Se preserva el criterio del código original: EntropyProg resuelve en el dual
  (optimiza multiplicadores), con posterior p_ de dimensión J.
- Índices Lower/Upper: por compatibilidad con MATLAB, se aceptan 1-based (ej. [4])
  y se convierten internamente a 0-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog, minimize


Array = np.ndarray


@dataclass
class FrontierOptions:
    NumPortf: int = 20
    FrontierSpan: Tuple[float, float] = (0.3, 0.9)


def _as_1d(a: ArrayLike, dtype=float) -> Array:
    x = np.asarray(a, dtype=dtype)
    return np.ravel(x)


def _as_2d(a: ArrayLike, dtype=float) -> Array:
    x = np.asarray(a, dtype=dtype)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def implied_exp_rets(S: ArrayLike, w: ArrayLike) -> Array:
    """
    Traducción directa de ImpliedExpRets.m

    MATLAB:
        M_=S*w;
        s=sqrt(mean(diag(S)));
        M=M_/mean(M_)*.5*s;

    En Python:
        S: (N,N), w: (N,) o (N,1) -> M: (N,)
    """
    S = np.asarray(S, dtype=float)
    w = _as_1d(w)
    M_ = S @ w
    s = np.sqrt(np.mean(np.diag(S)))
    M = M_ / np.mean(M_) * 0.5 * s
    return M


def efficient_frontier(
    X: ArrayLike,
    p: ArrayLike,
    options: Union[FrontierOptions, Dict[str, object], None] = None,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Traducción de EfficientFrontier.m

    Parameters
    ----------
    X : array (J,N) realizaciones de retornos
    p : array (J,) probabilidades (suman 1)
    options : FrontierOptions o dict con claves:
        - NumPortf
        - FrontierSpan (tuple/list de largo 2)

    Returns
    -------
    e : (NumPortf,) expected returns de carteras eficientes
    s : (NumPortf,) volatilities (stdev) de carteras eficientes
    w : (NumPortf,N) pesos de carteras eficientes
    Exps : (N,) esperanzas de activos
    Covs : (N,N) covarianza de activos
    """
    X = np.asarray(X, dtype=float)
    p = _as_1d(p, dtype=float)
    p = p / np.sum(p)

    if options is None:
        opt = FrontierOptions()
    elif isinstance(options, FrontierOptions):
        opt = options
    else:
        # dict-like
        num = int(options.get("NumPortf", 20))  # type: ignore[arg-type]
        span = options.get("FrontierSpan", (0.3, 0.9))  # type: ignore[arg-type]
        span = tuple(np.ravel(np.asarray(span, dtype=float)).tolist())
        if len(span) != 2:
            raise ValueError("FrontierSpan debe tener largo 2, ej. (0.3,0.9)")
        opt = FrontierOptions(NumPortf=num, FrontierSpan=(float(span[0]), float(span[1])))

    J, N = X.shape

    # Exps = X' * p
    Exps = X.T @ p  # (N,)

    # Scnd_Mom = X'*(X.*(p*ones(1,N)))
    Scnd_Mom = X.T @ (X * p[:, None])
    Scnd_Mom = 0.5 * (Scnd_Mom + Scnd_Mom.T)
    Covs = Scnd_Mom - np.outer(Exps, Exps)

    # Constraints: sum(w)=1, 0<=w<=1
    bounds = [(0.0, 1.0) for _ in range(N)]

    def _min_variance(expected_return_target: Optional[float] = None, w0: Optional[Array] = None) -> Array:
        if w0 is None:
            w0 = np.full(N, 1.0 / N)

        def obj(w: Array) -> float:
            return float(w @ Covs @ w)

        def grad(w: Array) -> Array:
            return 2.0 * (Covs @ w)

        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
        ]
        if expected_return_target is not None:
            target = float(expected_return_target)
            cons.append(
                {"type": "eq", "fun": lambda w, t=target: w @ Exps - t, "jac": lambda w: Exps.copy()}
            )

        res = minimize(
            obj,
            x0=w0,
            jac=grad,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 10_000, "disp": False},
        )
        if not res.success:
            raise RuntimeError(f"Min-variance QP no convergió: {res.message}")
        return np.asarray(res.x, dtype=float)

    # Minimum-risk portfolio
    w_minvol = _min_variance()
    min_exp = float(w_minvol @ Exps)

    # Maximum-return portfolio via linprog
    c = -Exps  # minimize c^T w = -Exps^T w
    A_eq = np.ones((1, N))
    b_eq = np.array([1.0])
    lp = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not lp.success:
        raise RuntimeError(f"linprog para max-return no convergió: {lp.message}")
    w_maxret = np.asarray(lp.x, dtype=float)
    max_exp = float(w_maxret @ Exps)

    # Slice frontier: Grid in [FrontierSpan[0], FrontierSpan[1]]
    grid = np.linspace(opt.FrontierSpan[0], opt.FrontierSpan[1], opt.NumPortf)
    targets = min_exp + grid * (max_exp - min_exp)

    e_list = []
    s_list = []
    w_list = []

    w_prev = w_minvol.copy()
    for t in targets:
        w_t = _min_variance(expected_return_target=float(t), w0=w_prev)
        w_prev = w_t
        w_list.append(w_t)
        s_list.append(float(np.sqrt(w_t @ Covs @ w_t)))
        e_list.append(float(w_t @ Exps))

    e = np.asarray(e_list)
    s = np.asarray(s_list)
    w = np.vstack(w_list)
    return e, s, w, Exps, Covs


def _matlab_to_python_indices(idxs: Sequence[int], N: int, one_based: bool = True) -> Array:
    """
    Convierte índices estilo MATLAB (1..N) a Python (0..N-1).

    FIX: reemplaza la heurística frágil (detectar si algún índice es 0)
    por un flag explícito `one_based`.  La heurística original silenciaba
    errores: si el usuario pasaba índices genuinamente 1-based pero con
    un 0 por error, se dejaban pasar sin conversión.

    Parameters
    ----------
    idxs     : secuencia de índices
    N        : tamaño del eje (para validación de rango)
    one_based: True  -> interpreta entradas como 1..N y resta 1 (default, MATLAB-compat)
               False -> interpreta entradas como 0..N-1 (ya están en Python-base)
    """
    idxs = np.asarray(list(idxs), dtype=int).ravel()
    if idxs.size == 0:
        return idxs
    out = idxs - 1 if one_based else idxs
    if np.any(out < 0) or np.any(out >= N):
        raise IndexError(
            f"Índices fuera de rango tras conversión. "
            f"Recibí {idxs} (one_based={one_based}), N={N}. "
            f"Resultado 0-based: {out}."
        )
    return out

####### Núcleo de la tesis: cálculo de posterior con Entropy Pooling #######
def entropy_prog(
    p: ArrayLike,
    A: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    Aeq: Optional[ArrayLike] = None,
    beq: Optional[ArrayLike] = None,
    entropy_family: str = "S",
    q: float = 1.0,
    g: Optional[Callable[[Array], Array]] = None,
    dg: Optional[Callable[[Array], Array]] = None,
) -> Tuple[Array, Callable[[Array], float], Array]:
    """
    Análogo de EntropyProg.m

    Resuelve:
        min_x  fe(x)  s.t.  A x <= b,  Aeq x = beq
    donde x es el vector de probabilidades posterior (dimensión J),
    y fe es:
        - Shannon (S): x'*(log(x)-log(p))
        - Tsallis (T): (1/(q-1)) * ( t(x) - 1 )
        - Renyi   (R): (1/(q-1)) * log( t(x) )
        - General (G): (1/(q-1)) * g( t(x) )
    con:
        t(x) = sum_i p_i * (x_i/p_i)^q

    Devuelve:
        p_ : posterior (J,)
        Lx : callable L(x) con los multiplicadores óptimos
        lv : vector concatenado de multiplicadores [l; v]
             (l: K_ ineq >=0, v: K eq libres)
    """
    p = _as_1d(p, dtype=float)
    if np.any(p < 0):
        raise ValueError("p debe ser no-negativo")
    if not np.isclose(np.sum(p), 1.0):
        p = p / np.sum(p)

    J = p.size

    # Default constraints
    if A is None or np.size(A) == 0:
        A = np.zeros((0, J), dtype=float)
    A = _as_2d(A, dtype=float)

    K_ = A.shape[0]

    if b is None or np.size(b) == 0:
        b = np.zeros(K_, dtype=float)
    b = _as_1d(b, dtype=float)
    if b.size != K_:
        raise ValueError("Dimensión incompatible: b debe tener largo K_ (= filas de A)")

    if Aeq is None or np.size(Aeq) == 0:
        Aeq = np.zeros((0, J), dtype=float)
    Aeq = _as_2d(Aeq, dtype=float)
    K = Aeq.shape[0]

    if beq is None or np.size(beq) == 0:
        beq = np.zeros(K, dtype=float)
    beq = _as_1d(beq, dtype=float)
    if beq.size != K:
        raise ValueError("Dimensión incompatible: beq debe tener largo K (= filas de Aeq)")

    ef = (entropy_family or "S").upper()
    if ef not in {"S", "T", "R", "G"}:
        # en MATLAB: warning y toma 'G'
        ef = "G"

    if ef != "S":
        if q == 1:
            raise ValueError("q debe ser distinto de 1 si entropy_family != 'S'")
        if ef == "T":
            g = lambda x: x - 1.0  # type: ignore[assignment]
            dg = lambda x: np.ones_like(x)  # type: ignore[assignment]
        elif ef == "R":
            g = np.log  # type: ignore[assignment]
            dg = lambda x: 1.0 / x  # type: ignore[assignment]
        elif ef == "G":
            if g is None or dg is None:
                raise ValueError("Con entropy_family='G' hay que proveer g y dg")
    else:
        # Shannon: ignora q, g, dg
        pass

    # ----- Definición fe(x)
    def _t_of_x(x: Array) -> float:
        # t(x) = sum p_i * (x_i/p_i)^q = sum x_i^q / p_i^{q-1}
        # control de divisiones: p_i>0 en datos típicos
        x = np.maximum(x, 1e-300)
        ratio = x / p
        # ratio^q puede desbordar si q grande; acá q típico ~1.5
        return float(np.sum(p * (ratio ** q)))

    def fe(x: Array) -> float:
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 1e-300)
        if ef == "S":
            return float(np.sum(x * (np.log(x) - np.log(p))))
        else:
            t = _t_of_x(x)
            # g espera array o escalar; forzamos float
            val = float(np.asarray(g(t)).item())  # type: ignore[misc]
            return float((1.0 / (q - 1.0)) * val)

    # ----- Helpers: punto fijo (Renyi y General)
    def despeja_itera_x(
        xini: Array,
        xalter: Array,
        F_raw: Callable[[Array], Array],
        tol: float = 1e-8,
        max_outer: int = 17,
        max_inner: int = 64,
        max_random: int = 1024,
    ) -> Array:
        """
        Traducción de despeja_itera_x() de EntropyProg.m

        FIX: las lambdas originales capturaban `alpha` por referencia,
        de modo que al actualizar `alpha` en el siguiente loop todas las
        lambdas previas veían el nuevo valor.  Se corrige capturando por
        valor con argumento default (patrón estándar de Python).
        """
        def _proj_simplex_abs(v: Array) -> Array:
            v = np.abs(v)
            s = float(np.sum(v))
            if s <= 0:
                # fallback uniforme
                return np.full_like(v, 1.0 / v.size)
            return v / s

        # normalizamos como en MATLAB
        x = _proj_simplex_abs(np.asarray(xini, dtype=float))

        def F_scaled(v: Array) -> Array:
            fr = np.asarray(F_raw(v), dtype=float)
            # escala L1 como en el original
            n1_v = float(np.sum(np.abs(v)))
            n1_fr = float(np.sum(np.abs(fr)))
            if n1_fr <= 0:
                return fr
            return fr * (n1_v / n1_fr)

        def U(v: Array) -> Array:
            return _proj_simplex_abs(F_scaled(v))

        def norm_diff(v: Array) -> float:
            return float(np.linalg.norm(v - F_scaled(v)))

        # ---------- primer intento desde xini ----------
        iterador = 0
        alpha = 0.5 ** iterador
        # FIX: `a=alpha` captura el valor actual, no la referencia a `alpha`
        G = lambda v, a=alpha: a * U(v) + (1.0 - a) * v

        while iterador < max_outer and norm_diff(x) > tol:
            contador = 1
            while norm_diff(x) > tol and contador < max_inner:
                x = G(x)
                x = _proj_simplex_abs(x)
                contador += 1
            iterador += 1
            alpha = 0.5 ** iterador
            G = lambda v, a=alpha: a * U(v) + (1.0 - a) * v  # FIX: captura por valor
            if norm_diff(x) > norm_diff(_proj_simplex_abs(xini)):
                x = _proj_simplex_abs(xini)

        if norm_diff(x) <= tol:
            return x

        # ---------- segundo intento desde xalter ----------
        x = _proj_simplex_abs(np.asarray(xalter, dtype=float))
        iterador = 0
        alpha = 0.5 ** iterador
        G = lambda v, a=alpha: a * U(v) + (1.0 - a) * v  # FIX: captura por valor

        while iterador < max_outer and norm_diff(x) > tol:
            contador = 1
            while norm_diff(x) > tol and contador < max_inner:
                x = G(x)
                x = _proj_simplex_abs(x)
                contador += 1
            iterador += 1
            alpha = 0.5 ** iterador
            G = lambda v, a=alpha: a * U(v) + (1.0 - a) * v  # FIX: captura por valor
            if norm_diff(x) > norm_diff(_proj_simplex_abs(xalter)):
                x = _proj_simplex_abs(xalter)

        if norm_diff(x) <= tol:
            return x

        # ---------- random restarts ----------
        for _ in range(max_random):
            xseed = np.random.rand(J)
            xseed = xseed / np.sum(xseed)
            x = xseed
            iterador = 0
            alpha = 0.5 ** iterador
            G = lambda v, a=alpha: a * U(v) + (1.0 - a) * v  # FIX: captura por valor

            while iterador < max_outer and norm_diff(x) > tol:
                contador = 1
                while norm_diff(x) > tol and contador < max_inner:
                    x = G(x)
                    x = _proj_simplex_abs(x)
                    contador += 1
                iterador += 1
                alpha = 0.5 ** iterador
                G = lambda v, a=alpha: a * U(v) + (1.0 - a) * v  # FIX: captura por valor
                if norm_diff(x) > norm_diff(xseed):
                    x = xseed

            if norm_diff(x) <= tol:
                return x

        raise RuntimeError("no se pudo hallar punto fijo")

    def equis(l: Array, v: Array) -> Array:
        """
        Traducción de equis() dentro de EntropyProg.m

        Recibe multiplicadores:
            l: (K_,) para desigualdades (l>=0)
            v: (K,)  para igualdades (v libres)
        Devuelve x (posterior) de dimensión J.
        """
        # z = A'*l + Aeq'*v  (nota: A es K_ x J)
        z = np.zeros(J, dtype=float)
        if K_ > 0:
            z = z + (A.T @ l)
        if K > 0:
            z = z + (Aeq.T @ v)

        if ef == "S":
            x = np.exp(np.log(p) - 1.0 - z)
            x = np.maximum(x, 1e-32)
            return x

        # factor común (1/q - 1) = -(q-1)/q
        c = (1.0 / q) - 1.0
        power = 1.0 / (q - 1.0)

        if ef == "T":
            # FIX: c = (1/q)-1 < 0 cuando q > 1.  La solución KKT requiere
            # que c*z_i > 0 para cada componente, lo que equivale a z_i < 0
            # (pues c < 0).  Si algún z_i viola esto, el clamp a 1e-32 oculta
            # el problema en lugar de reportarlo.  Chequeamos y advertimos.
            arg = c * z
            n_invalid = int(np.sum(arg <= 0))
            if n_invalid > 0:
                import warnings
                warnings.warn(
                    f"equis(Tsallis): {n_invalid} componentes con c*z <= 0 "
                    f"(c={c:.4f}). El posterior puede ser inexacto en esas "
                    f"componentes. Verificá que los multiplicadores converjan.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            arg = np.maximum(arg, 1e-32)
            x = (arg ** power) * p
            x = np.maximum(x, 1e-32)
            return x

        if ef == "R":
            # Renyi: x = p * ( c * t(x) * z )^{1/(q-1)}
            # (implícito) -> punto fijo
            # seed: Tsallis con t=1 (misma forma que el caso explícito)
            xini = (np.maximum(c * z, 1e-32) ** power) * p
            xalter = p.copy()

            def F_raw(x_cur: Array) -> Array:
                t = _t_of_x(x_cur)
                arg = c * t * z
                arg = np.maximum(arg, 1e-32)
                return (arg ** power) * p

            return despeja_itera_x(xini=xini, xalter=xalter, F_raw=F_raw)

        # General (G)
        xini = (np.maximum(c * z, 1e-32) ** power) * p
        xalter = p.copy()

        def F_raw(x_cur: Array) -> Array:
            t = _t_of_x(x_cur)
            dt = float(np.asarray(dg(t)).item())  # type: ignore[misc]
            arg = c * (1.0 / dt) * z
            arg = np.maximum(arg, 1e-32)
            return (arg ** power) * p

        return despeja_itera_x(xini=xini, xalter=xalter, F_raw=F_raw)

    # Caso sin views
    if (K_ + K) == 0:
        # en MATLAB: warning y retorna p
        p_ = p.copy()

        def Lx(x: Array) -> float:
            return fe(x)

        return p_, Lx, np.zeros(0)

    # ---- Branch q<1: Newton-Raphson multi (como en MATLAB)
    # Lo implemento por completitud, pero en la práctica la tesis suele usar q>1.
    if ef != "S" and q < 1.0:
        # Armamos el sistema de ecuaciones: grad(Lagrangiano) = 0
        # Variables concatenadas: X = [x (J), l (K_), v (K)]
        def Lagr(Xbig: Array) -> float:
            x = Xbig[:J]
            l = Xbig[J : J + K_]
            v = Xbig[J + K_ :]
            return fe(x) + float(l @ (A @ x - b)) + float(v @ (Aeq @ x - beq))

        def jacob_aprox(H: Callable[[Array], Array], X0: Array, epsi: Optional[float] = None) -> Array:
            X0 = np.asarray(X0, dtype=float)
            if epsi is None:
                epsi = float(np.sqrt(np.sqrt(np.finfo(float).eps * (1.0 + np.linalg.norm(X0)))))
            I = np.eye(X0.size)
            # H puede ser escalar o vector
            H0 = np.asarray(H(X0))
            out_dim = H0.size
            Jm = np.zeros((out_dim, X0.size), dtype=float)
            for k in range(X0.size):
                f1 = np.asarray(H(X0 + epsi * I[:, k])).reshape(-1)
                f2 = np.asarray(H(X0 - epsi * I[:, k])).reshape(-1)
                Jm[:, k] = (f1 - f2) / (2.0 * epsi)
            return Jm

        def grad_Lagr(Xbig: Array) -> Array:
            # grad aproximado como en MATLAB: JACOB_APROX(Lx,X)' para Lx escalar
            return jacob_aprox(lambda z: np.array([Lagr(z)]), Xbig).reshape(-1)

        def newton_raph(h: Callable[[Array], Array], dh: Callable[[Array], Array], x: Array, n: int) -> Array:
            x = np.asarray(x, dtype=float)
            for _ in range(n):
                Jm = dh(x)
                fx = h(x)
                # FIX: capturamos LinAlgError (Jacobiano singular) y hacemos
                # fallback a lstsq en lugar de propagar la excepción sin contexto.
                try:
                    step = np.linalg.solve(Jm, fx)
                except np.linalg.LinAlgError:
                    step, _, _, _ = np.linalg.lstsq(Jm, fx, rcond=None)
                x = x - step
            return x

        def nr_multi(h: Callable[[Array], Array], x0: Array, n: int) -> Array:
            return newton_raph(h, lambda z: jacob_aprox(h, z), x0, n)

        x0 = np.concatenate([p.copy(), np.zeros(K_ + K, dtype=float)])
        PC = nr_multi(grad_Lagr, x0, 17)
        tol = 1e-8
        contador = 1
        while np.linalg.norm(grad_Lagr(PC)) > tol and contador < 4097:
            contador += 1
            # random restart como MATLAB
            PC = nr_multi(
                grad_Lagr,
                np.concatenate([p.copy(), 10.0 * np.random.rand(K_), 10.0 * np.random.randn(K)]),
                17,
            )

        if np.linalg.norm(grad_Lagr(PC)) < tol:
            p_ = PC[:J]
            p_ = np.maximum(p_, 0.0)
            p_ = p_ / np.sum(p_)
            lv = PC[J:]
            l = lv[:K_]
            v = lv[K_:]

            def Lx(x: Array) -> float:
                x = _as_1d(x, dtype=float)
                return fe(x) + float(l @ (A @ x - b)) + float(v @ (Aeq @ x - beq))

            return p_, Lx, lv

        raise RuntimeError("no se encontró solución (branch q<1)")

    # ---- Branch q>=1 (o Shannon): optimización dual en (l,v)

    def dual_objective(lv: Array) -> Tuple[float, Array]:
        lv = _as_1d(lv, dtype=float)
        l = lv[:K_]
        v = lv[K_:]
        x = equis(l, v)
        # L = fe(x) + l' (A x - b) + v'(Aeq x - beq)
        Ax_minus_b = (A @ x - b) if K_ > 0 else np.zeros(0)
        Aeqx_minus_beq = (Aeq @ x - beq) if K > 0 else np.zeros(0)
        L = fe(x) + float(l @ Ax_minus_b) + float(v @ Aeqx_minus_beq)
        mL = -L
        # grad(mL) = [b - A x ; beq - Aeq x]
        grad = []
        if K_ > 0:
            grad.append(b - A @ x)
        if K > 0:
            grad.append(beq - Aeq @ x)
        gvec = np.concatenate(grad) if grad else np.zeros(0)
        return float(mL), gvec

    x0 = np.zeros(K_ + K, dtype=float)

    # Warm-start (como en los comentarios de EntropyProg.m):
    # - Para Tsallis puede ser útil iniciar desde la solución Shannon
    # - Para Rényi/General, iniciar desde la solución Tsallis suele evitar fallas del line-search
    if ef == "T":
        try:
            _, _, lv_s = entropy_prog(p, A, b, Aeq, beq, entropy_family="S", q=1.0)
            if lv_s.size == x0.size:
                x0 = lv_s
        except Exception:
            pass
    elif ef in ("R", "G"):
        try:
            _, _, lv_t = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=q)
            if lv_t.size == x0.size:
                x0 = lv_t
        except Exception:
            pass


    # bounds: l>=0, v free
    bounds = [(0.0, None)] * K_ + [(None, None)] * K

    res = minimize(
        fun=lambda lv: dual_objective(lv),
        x0=x0,
        jac=True,
        method="L-BFGS-B" if (K_ > 0) else "BFGS",
        bounds=bounds if (K_ > 0) else None,
        options={"maxiter": 10_000, "ftol": 1e-12, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"EntropyProg no convergió: {res.message}")

    lv_opt = _as_1d(res.x, dtype=float)
    l_opt = lv_opt[:K_]
    v_opt = lv_opt[K_:]
    p_ = equis(l_opt, v_opt)
    # proyectamos numéricamente al simplex (evita drift tipo 1.0000001)
    p_ = np.maximum(p_, 0.0)
    p_ = p_ / np.sum(p_)

    def Lx(x: Array) -> float:
        x = _as_1d(x, dtype=float)
        return fe(x) + float(l_opt @ (A @ x - b)) + float(v_opt @ (Aeq @ x - beq))

    return p_, Lx, lv_opt


def view_ranking(
    X: ArrayLike,
    p: ArrayLike,
    Lower: Sequence[int],
    Upper: Sequence[int],
    entropy_family: str = "T",
    q: float = 1.5,
    one_based: bool = True,
) -> Array:
    """
    Traducción directa de ViewRanking.m

    FIX: la versión original hardcodeaba entropy_family='T' y q=1.5,
    ignorando la elección del usuario en s_main.  Ahora se pasan como
    parámetros con los mismos defaults que el original para no romper
    compatibilidad hacia atrás.

    Parameters
    ----------
    Lower, Upper : índices de activos (MATLAB 1-based por default)
    entropy_family : 'S', 'T', 'R' o 'G'  (default 'T')
    q              : parámetro de entropía generalizada (default 1.5)
    one_based      : True si los índices vienen en base-1 estilo MATLAB
    """
    X = np.asarray(X, dtype=float)
    p = _as_1d(p, dtype=float)
    p = p / np.sum(p)

    J, N = X.shape
    lower_idx = _matlab_to_python_indices(Lower, N, one_based=one_based)
    upper_idx = _matlab_to_python_indices(Upper, N, one_based=one_based)

    # constrain probabilities to sum to one
    Aeq = np.ones((1, J), dtype=float)
    beq = np.array([1.0], dtype=float)

    # constrain expectations: E[X_Lower] <= E[X_Upper]  =>  V = X[:,Lower]-X[:,Upper]
    V = X[:, lower_idx] - X[:, upper_idx]  # (J,K)
    A = V.T  # (K,J)
    b = np.zeros(A.shape[0], dtype=float)

    p_, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family=entropy_family, q=q)
    return p_


# ---------------- Plotting (matplotlib) ----------------
def plot_frontier(e: ArrayLike, s: ArrayLike, w: ArrayLike, ax=None) -> None:
    """
    Traducción de PlotFrontier.m: stacked area de composiciones vs riesgo.

    e: (NumPortf,) returns
    s: (NumPortf,) vol
    w: (NumPortf,N) weights
    """
    import matplotlib.pyplot as plt

    e = _as_1d(e)
    s = _as_1d(s)
    w = np.asarray(w, dtype=float)

    if ax is None:
        ax = plt.gca()

    data = np.cumsum(w, axis=1)  # (NumPortf,N)
    num_portf, N = w.shape

    # FIX: el esquema original `str(0.9 - (n%3)*0.2)` cicla en solo 3 grises
    # (0.9, 0.7, 0.5) y se repite para N>3 activos, haciendo ilegible el gráfico.
    # Usamos un colormap continuo que escala correctamente con N.
    cmap = plt.get_cmap("Greys")
    # reservamos el rango [0.25, 0.85] para evitar extremos (blanco puro / negro puro)
    colors = [cmap(0.85 - n * (0.6 / max(N - 1, 1))) for n in range(N)]

    # MATLAB rellena en orden inverso para apilar
    x = np.concatenate([[np.min(s)], s, [np.max(s)]])
    for n in range(N):
        y = np.concatenate([[0.0], data[:, N - n - 1], [0.0]])
        ax.fill(x, y, color=colors[n], linewidth=0.0)

    ax.set_xlim(float(np.min(s)), float(np.max(s)))
    ax.set_ylim(0.0, float(np.max(data)))


def plot_results(
    e: ArrayLike,
    s: ArrayLike,
    w: ArrayLike,
    M: ArrayLike,
    Lower: Optional[Sequence[int]] = None,
    Upper: Optional[Sequence[int]] = None,
) -> None:
    """
    Traducción de PlotResults.m
    """
    import matplotlib.pyplot as plt

    M = _as_1d(M)
    N = M.size

    fig = plt.figure()

    # subplot('position',[0.05 .1 .3 .8])
    ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
    bars = ax1.barh(np.arange(1, N + 1), M * 100.0, color="0.8", edgecolor="0.4")

    if Lower is not None and Upper is not None:
        # resalta los que están en union(Lower,Upper)
        # aceptamos 1-based (MATLAB) o 0-based
        idxs = set(list(Lower) + list(Upper))
        # convertir a 0-based para marcar
        # heurística: si hay 0 -> 0-based
        if 0 in idxs:
            changed = np.zeros_like(M)
            for i in idxs:
                if 0 <= i < N:
                    changed[i] = M[i] * 100.0
        else:
            changed = np.zeros_like(M)
            for i in idxs:
                ii = i - 1
                if 0 <= ii < N:
                    changed[ii] = M[ii] * 100.0

        ax1.barh(np.arange(1, N + 1), changed, color="r", edgecolor="0.4")

    ax1.set_ylim(0, N + 1)
    ax1.grid(True)
    ax1.set_title("expected returns")

    # subplot('position',[0.45 .1 .5 .8])
    ax2 = fig.add_axes([0.45, 0.1, 0.5, 0.8])
    plot_frontier(np.asarray(e) * 100.0, np.asarray(s) * 100.0, w, ax=ax2)
    ax2.set_title("frontier")


# ---------------- Robust Efficient Frontier ----------------
def _pcacov(cov: Array) -> Tuple[Array, Array]:
    """
    Equivalente a pcacov(Cov) de MATLAB:
    devuelve (F, G) donde columnas de F son eigenvectors, G eigenvalues,
    ordenados descendente por eigenvalue.
    """
    cov = np.asarray(cov, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vecs, vals


def robust_efficient_frontier(
    target_vols: Sequence[float],
    estimate: Dict[str, ArrayLike],
    constr: Dict[str, ArrayLike],
) -> Tuple[Array, Array, Array]:
    """
    Traducción de RobustEfficientFrontier.m

    Nota: MATLAB usa CVX (SOCP). Acá lo traducimos con SciPy 'trust-constr'
    (no es un solver cónico dedicado). Para tamaños chicos suele funcionar.

    estimate debe contener:
        - 'Cov'      (N,N)
        - 'Sigma_c'  (N,N)
        - 'Mu_c'     (N,)  (o (N,1))
        - 'ExpVal'   (N,)  (o (N,1))

    constr debe contener:
        - 'Aeq'  (p,N)
        - 'beq'  (p,)
        - 'Aleq' (q,N)
        - 'bleq' (q,)
    """
    Cov = np.asarray(estimate["Cov"], dtype=float)
    Sigma_c = np.asarray(estimate["Sigma_c"], dtype=float)
    Mu_c = _as_1d(estimate["Mu_c"], dtype=float)
    ExpVal = _as_1d(estimate["ExpVal"], dtype=float)

    N = Cov.shape[0]
    Aeq0 = _as_2d(constr.get("Aeq", np.zeros((0, N))), dtype=float)
    beq0 = _as_1d(constr.get("beq", np.zeros(Aeq0.shape[0])), dtype=float)
    Aleq0 = _as_2d(constr.get("Aleq", np.zeros((0, N))), dtype=float)
    bleq0 = _as_1d(constr.get("bleq", np.zeros(Aleq0.shape[0])), dtype=float)

    p_rows = Aeq0.shape[0]
    q_rows = Aleq0.shape[0]

    F, G = _pcacov(Cov)
    GF = np.concatenate([np.diag(np.sqrt(np.maximum(G, 0.0))) @ F.T, np.zeros((N, 1))], axis=1)

    E, L = _pcacov(Sigma_c)
    LE = np.concatenate([np.diag(np.sqrt(np.maximum(L, 0.0))) @ E.T, np.zeros((N, 1))], axis=1)

    Aeq = np.concatenate([Aeq0, np.zeros((p_rows, 1))], axis=1)
    beq = beq0.copy()
    Aleq = np.concatenate([Aleq0, np.zeros((q_rows, 1))], axis=1)
    bleq = bleq0.copy()

    m = np.concatenate([Mu_c, np.array([-1.0])])  # (N+1,)

    weights_list = []
    vol_list = []
    exp_list = []

    # bounds: dejamos sin bounds salvo el último >=0 para ayudar
    bounds = [(None, None)] * N + [(0.0, None)]

    # helper norms
    def c1(x: Array) -> float:
        # norm(LE x) - x_{N}
        return float(np.linalg.norm(LE @ x) - x[-1])

    def jac_c1(x: Array) -> Array:
        y = LE @ x
        ny = np.linalg.norm(y)
        grad = np.zeros(N + 1, dtype=float)
        if ny > 0:
            grad += LE.T @ (y / ny)
        grad[-1] -= 1.0
        return grad

    def make_c2(tv: float):
        def c2(x: Array) -> float:
            return float(np.linalg.norm(GF @ x) - tv)

        def jac_c2(x: Array) -> Array:
            y = GF @ x
            ny = np.linalg.norm(y)
            grad = np.zeros(N + 1, dtype=float)
            if ny > 0:
                grad += GF.T @ (y / ny)
            return grad

        return c2, jac_c2

    for tv in target_vols:
        tv = float(tv)

        def obj(x: Array) -> float:
            return float(-x @ m)

        def grad(x: Array) -> Array:
            return -m

        cons = []
        # linear equality Aeq x = beq
        for i in range(p_rows):
            ai = Aeq[i, :]
            bi = beq[i]
            cons.append({"type": "eq", "fun": lambda x, a=ai, b=bi: float(a @ x - b), "jac": lambda x, a=ai: a})

        # linear inequalities Aleq x <= bleq
        for i in range(q_rows):
            ai = Aleq[i, :]
            bi = bleq[i]
            cons.append({"type": "ineq", "fun": lambda x, a=ai, b=bi: float(b - a @ x), "jac": lambda x, a=ai: -a})

        # nonlinear inequalities: c1(x) <=0 and c2(x)<=0
        cons.append({"type": "ineq", "fun": lambda x: -c1(x), "jac": lambda x: -jac_c1(x)})
        c2, jc2 = make_c2(tv)
        cons.append({"type": "ineq", "fun": lambda x: -c2(x), "jac": lambda x: -jc2(x)})

        x0 = np.concatenate([np.full(N, 1.0 / N), np.array([tv])])

        res = minimize(
            obj,
            x0=x0,
            jac=grad,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 20_000, "disp": False},
        )
        if not res.success:
            raise RuntimeError(f"RobustEfficientFrontier no convergió (tv={tv}): {res.message}")

        x = np.asarray(res.x, dtype=float)
        w = x[:N]
        weights_list.append(w)
        vol_list.append(float(np.sqrt(w @ Cov @ w)))
        exp_list.append(float(w @ ExpVal))

    Weights = np.vstack(weights_list)
    Vol = np.asarray(vol_list)
    ExpVal_out = np.asarray(exp_list)
    return ExpVal_out, Vol, Weights



# ---------------------------------------------------------------------
# MATLAB-style aliases (para que s_main.py se parezca a S_MAIN.m)
# ---------------------------------------------------------------------

def ImpliedExpRets(S: ArrayLike, w: ArrayLike) -> Array:
    return implied_exp_rets(S, w)

def EfficientFrontier(X: ArrayLike, p: ArrayLike, Options: Union[FrontierOptions, Dict[str, object], None] = None):
    return efficient_frontier(X, p, Options)

def PlotFrontier(e: ArrayLike, s: ArrayLike, w: ArrayLike, ax=None) -> None:
    return plot_frontier(e, s, w, ax=ax)

def PlotResults(e: ArrayLike, s: ArrayLike, w: ArrayLike, M: ArrayLike, Lower=None, Upper=None) -> None:
    return plot_results(e, s, w, M, Lower=Lower, Upper=Upper)

def RobustEfficientFrontier(TargetVols: Sequence[float], Estimate: Dict[str, ArrayLike], Constr: Dict[str, ArrayLike]):
    return robust_efficient_frontier(TargetVols, Estimate, Constr)

def EntropyProg(
    p: ArrayLike,
    A: Optional[ArrayLike] = None,
    b: Optional[ArrayLike] = None,
    Aeq: Optional[ArrayLike] = None,
    beq: Optional[ArrayLike] = None,
    entropy_family: str = "S",
    q: float = 1.0,
    g: Optional[Callable[[Array], Array]] = None,
    dg: Optional[Callable[[Array], Array]] = None,
):
    return entropy_prog(p, A, b, Aeq, beq, entropy_family=entropy_family, q=q, g=g, dg=dg)

def ViewRanking(
    X: ArrayLike,
    p: ArrayLike,
    Lower: Sequence[int],
    Upper: Sequence[int],
    entropy_family: str = "T",
    q: float = 1.5,
    one_based: bool = True,
) -> Array:
    return view_ranking(X, p, Lower, Upper, entropy_family=entropy_family, q=q, one_based=one_based)