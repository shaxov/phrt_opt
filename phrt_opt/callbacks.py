import numpy as np
from phrt_opt import linesearch
from phrt_opt.ops import counters


class IsConvergedCallback:

    def __init__(self, metric: callable, tol: float):
        self.metric = metric
        self.tol = tol
        self.x = None

    def __call__(self, x) -> bool:
        if self.x is None:
            self.x = x
            return False
        success_ok = self.metric(x, self.x) < self.tol
        self.x = x
        return success_ok


class IsSolvedCallback:

    def __init__(self, x_opt: np.array, metric: callable, tol: float):
        self.x_opt = x_opt
        self.metric = metric
        self.tol = tol

    def __call__(self, x) -> bool:
        return self.metric(x, self.x_opt) < self.tol


class MetricCallback:

    def __init__(self, x_opt: np.array, metric: callable):
        self.x_opt = x_opt
        self.metric = metric

    def __call__(self, x) -> float:
        return self.metric(x, self.x_opt)


class OpsBacktrackingCallback:

    def __init__(self, fun: callable, gradient: callable = None, **params):
        self.fun = fun
        self.tm_shape = fun.tm_shape
        self.gradient = gradient
        self.params = params

    def __call__(self, x, **kwargs):
        bls = linesearch.backtracking(**self.params)
        p = kwargs.get('p', self.gradient(x))
        _ = bls(self.fun, x, p)
        return counters.backtracking_init(*self.tm_shape) + \
               bls.it * counters.backtracking_step(*self.tm_shape)


class OpsSecantCallback:

    def __init__(self, fun: callable, gradient: callable = None, **bls_params):
        self.fun = fun
        self.tm_shape = fun.tm_shape
        self.gradient = gradient
        self.bls_params = bls_params
        self.prev_x, self.prev_p = None, None

    def __call__(self, x, **kwargs):
        p = kwargs.get('p', self.gradient(x))
        bls = OpsBacktrackingCallback(self.fun, self.gradient, **self.bls_params)
        if self.prev_x is None or self.prev_p is None:
            self.prev_x, self.prev_p = x, p
            return bls(x, **kwargs)
        sk, yk = x - self.prev_x, p - self.prev_p
        self.prev_x, self.prev_p = x, p
        ck = np.real(np.vdot(sk, yk))
        if ck > 0: return counters.secant(*self.tm_shape)
        return bls(x, **kwargs)


OpsSecantSymmetricCallback = OpsSecantCallback


class RhoStrategyCallback:

    def __init__(self, tm, tm_pinv, b, strategy):
        m = np.shape(tm)[0]
        self.lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
        self.eps = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
        self.strategy = strategy

        self.tm = tm
        self.tm_pinv = tm_pinv
        self.b = b
        self.it = 0

    def __call__(self, x):
        tm_x = self.tm.dot(x)
        z = self.b * np.exp(1j * np.angle(tm_x - self.eps + self.lmd))
        x = self.tm_pinv.dot(z + self.eps - self.lmd)
        y = self.tm.dot(x)

        rho = self.strategy(self.it, y, z)

        # self.eps = (rho / (1 + rho)) * (y - z + self.lmd)
        # self.lmd = self.lmd + y - z - self.eps
        self.it += 1
        return rho
