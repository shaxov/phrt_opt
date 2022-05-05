import numpy as np


def backtracking(fun, x, gk, c1=1e-4, rate=0.5, max_iter=100, alpha0=1.):
    """ Armijo backtracking line search. [1, p.33, (3.4)]

    Reference:
        [1] Jorge Nocedal, Stephen J. Wright, Numerical Optimization, 2nd edition.
    """
    fun_x = fun(x)
    alpha = alpha0
    gk_norm2 = np.linalg.norm(gk) ** 2
    for _ in range(max_iter):
        h = fun(x - alpha * gk) + alpha * c1 * gk_norm2
        if fun_x > h: break
        alpha *= rate
    return alpha


class secant:

    def __init__(self):
        self.prev_x, self.prev_gk = None, None

    def __call__(self, fun, x, gk, c1=1e-4, rate=0.5, max_iter=100, alpha0=1.):
        if self.prev_x is None or self.prev_gk is None:
            self.prev_x, self.prev_gk = x, gk
            return 1.
        sk, yk = x - self.prev_x, gk - self.prev_gk
        self.prev_x, self.prev_gk = x, gk
        ck = np.real(np.vdot(sk, yk))
        if ck > 0: return np.vdot(sk, sk) / ck
        return 1.


class secant_symmetric(secant):

    def __call__(self, fun, x, gk, c1=1e-4, rate=0.5, max_iter=5, alpha0=1.):
        if self.prev_x is None or self.prev_gk is None:
            self.prev_x, self.prev_gk = x, gk
            return backtracking(fun, x, gk, c1, rate, max_iter, alpha0)
        sk, yk = x - self.prev_x, gk - self.prev_gk
        self.prev_x, self.prev_gk = x, gk
        ck = np.real(np.vdot(sk, yk))
        if ck > 0: return ck / (np.vdot(yk, yk) + np.finfo(float).eps)
        return backtracking(fun, x, gk, c1, rate, max_iter, alpha0)
        # return np.abs(ck / (np.vdot(yk, yk) + np.finfo(float).eps))
