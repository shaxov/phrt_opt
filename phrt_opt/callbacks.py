import numpy as np


class ConvergenceCallback:

    def __init__(self, metric: callable, tol: float):
        self.metric = metric
        self.tol = tol
        self.x = None

    def __call__(self, x):
        if self.x is None:
            self.x = x
            return False
        success_ok = self.metric(x, self.x) < self.tol
        self.x = x
        return success_ok


class MetricCallback:

    def __init__(self, x_opt: np.array, metric: callable):
        self.x_opt = x_opt
        self.metric = metric

    def __call__(self, x):
        return self.metric(x, self.x_opt)


class IsSolvedCallback:

    def __init__(self, x_opt: np.array, metric: callable, tol: float):
        self.x_opt = x_opt
        self.metric = metric
        self.tol = tol

    def __call__(self, x):
        return self.metric(x, self.x_opt) < self.tol


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