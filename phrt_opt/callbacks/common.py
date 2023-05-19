import numpy as np


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

