import numpy as np
from phrt_opt import typedef
from numpy.linalg import norm


class backtracking:
    """ Armijo backtracking line search [1, p.33, (3.4)].

    Reference:
        [1] Jorge Nocedal, Stephen J. Wright, Numerical Optimization, 2nd edition.
    """

    def __init__(
            self,
            c1=typedef.DEFAULT_BACKTRACKING_C1,
            rate=typedef.DEFAULT_BACKTRACKING_RATE,
            max_iter=typedef.DEFAULT_BACKTRACKING_MAX_ITER,
            alpha0=typedef.DEFAULT_BACKTRACKING_ALPHA0):
        self.c1 = c1
        self.rate = rate
        self.max_iter = max_iter
        self.alpha0 = alpha0
        self.it = 0

    def __call__(self, fun, x, grad):
        f_x = fun(x)
        alpha = self.alpha0
        g_norm2 = norm(grad)**2
        for _ in range(self.max_iter):
            h = fun(x - alpha * grad) + alpha * self.c1 * g_norm2
            if f_x > h: break
            alpha *= self.rate
            self.it += 1
        return alpha


class secant:
    """ Secant equation based line search [1].

    Reference:
        [1] JONATHAN BARZILAI, JONATHAN M. BORWEIN, Two-Point Step Size Gradient Methods,
            IMA Journal of Numerical Analysis, Volume 8, Issue 1, January 1988, Pages 141–148,
    """

    def __init__(self, backtracking_):
        self.backtracking_ = backtracking_
        self.prev_x, self.prev_p = None, None

    def __call__(self, fun, x, p):
        if self.prev_x is None or self.prev_p is None:
            self.prev_x, self.prev_p = x, p
            return self.backtracking_(fun, x, p)
        sk, yk = x - self.prev_x, p - self.prev_p
        self.prev_x, self.prev_p = x, p
        ck = np.real(np.vdot(sk, yk))
        if ck > 0: return np.vdot(sk, sk) / ck
        return self.backtracking_(fun, x, p)


class secant_symmetric(secant):
    """ Secant equation based line search (symmetric variant) [1].

    Reference:
        [1] JONATHAN BARZILAI, JONATHAN M. BORWEIN, Two-Point Step Size Gradient Methods,
            IMA Journal of Numerical Analysis, Volume 8, Issue 1, January 1988, Pages 141–148,
    """

    def __call__(self, fun, x, p):
        if self.prev_x is None or self.prev_p is None:
            self.prev_x, self.prev_p = x, p
            return self.backtracking_(fun, x, p)
        sk, yk = x - self.prev_x, p - self.prev_p
        self.prev_x, self.prev_p = x, p
        ck = np.real(np.vdot(sk, yk))
        if ck > 0: return ck / (np.vdot(yk, yk) + np.finfo(float).eps)
        return self.backtracking_(fun, x, p)

