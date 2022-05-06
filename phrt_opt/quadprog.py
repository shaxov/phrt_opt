import numpy as np
from phrt_opt import typedef
from scipy.linalg import solve_triangular


def conjugate_gradient_solver(
        mat, b,
        x0=None,
        tol=typedef.DEFAULT_CG_TOL,
        max_iter=typedef.DEFAULT_CG_MAX_ITER,
        dlt=typedef.DEFAULT_REG_DLT):
    m, n = np.shape(mat)
    if m != n:
        raise ValueError("Matrix is not square.")
    if dlt is not None:
        mat = mat + dlt * np.eye(n)

    xk = x0
    if x0 is None:
        xk = np.array([1 / n] * n).reshape(-1, 1)
    if max_iter is None:
        max_iter = n ** 2 + 1
    rk = np.dot(mat, xk) - b
    pk = -rk
    for it in range(1, max_iter + 1):
        if np.linalg.norm(rk) < tol:
            break
        mat_pk = np.dot(mat, pk)
        alpha = np.dot(rk.T.conj(), rk) / np.dot(pk.T.conj(), mat_pk)
        xk = xk + alpha * pk
        rk_prev = rk
        rk = rk + alpha * mat_pk
        beta = np.dot(rk.T.conj(), rk) / np.dot(rk_prev.T.conj(), rk_prev)
        pk = -rk + beta * pk
    return xk


def cholesky_solver(mat, b, dlt=typedef.DEFAULT_REG_DLT):
    m, n = np.shape(mat)
    if m != n:
        raise ValueError("Matrix is not square.")
    if dlt is not None:
        mat = mat + dlt * np.eye(n)

    L = np.linalg.cholesky(mat)
    Lh = np.conj(L.T)
    y = solve_triangular(L, b, lower=True)
    x = solve_triangular(Lh, y)
    return x
