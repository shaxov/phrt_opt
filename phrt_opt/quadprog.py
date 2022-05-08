import numpy as np
from phrt_opt import typedef
from scipy.linalg import solve_triangular


class ConjugateGradient:

    def __init__(
            self,
            x0=typedef.DEFAULT_CG_X0,
            tol=typedef.DEFAULT_CG_TOL,
            max_iter=typedef.DEFAULT_CG_MAX_ITER,
            dlt=typedef.DEFAULT_REG_DLT):
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.dlt = dlt
        self.it = 0

    @staticmethod
    def name():
        return "conjugate_gradient"

    def __call__(self, mat, b):
        m, n = np.shape(mat)
        if m != n:
            raise ValueError("Matrix is not square.")
        if self.dlt is not None:
            mat = mat + self.dlt * np.eye(n)

        xk = self.x0
        if self.x0 is None:
            xk = np.array([1 / n] * n).reshape(-1, 1)
        if self.max_iter is None:
            self.max_iter = n ** 2 + 1
        rk = np.dot(mat, xk) - b
        pk = -rk
        for _ in range(self.max_iter):
            if np.linalg.norm(rk) < self.tol:
                break
            mat_pk = np.dot(mat, pk)
            alpha = np.dot(rk.T.conj(), rk) / np.dot(pk.T.conj(), mat_pk)
            xk = xk + alpha * pk
            rk_prev = rk
            rk = rk + alpha * mat_pk
            beta = np.dot(rk.T.conj(), rk) / np.dot(rk_prev.T.conj(), rk_prev)
            pk = -rk + beta * pk
            self.it += 1
        return xk


class Cholesky:

    def __init__(self, dlt=typedef.DEFAULT_REG_DLT):
        self.dlt = dlt
        self.it = 1

    @staticmethod
    def name():
        return "cholesky"

    def __call__(self, mat, b):
        m, n = np.shape(mat)
        if m != n:
            raise ValueError("Matrix is not square.")
        if self.dlt is not None:
            mat = mat + self.dlt * np.eye(n)

        L = np.linalg.cholesky(mat)
        Lh = np.conj(L.T)
        y = solve_triangular(L, b, lower=True)
        x = solve_triangular(Lh, y)
        return x


def get(name):
    return {
        ConjugateGradient.name(): ConjugateGradient,
        Cholesky.name(): Cholesky,
    }[name]