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