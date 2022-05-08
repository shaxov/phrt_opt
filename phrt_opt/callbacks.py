import abc
import numpy as np
import phrt_opt.utils
from phrt_opt import typedef
from phrt_opt.ops import counters

from phrt_opt.linesearch import Backtracking

from phrt_opt.quadprog import Cholesky
from phrt_opt.quadprog import ConjugateGradient



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



class OpsCallback(metaclass=abc.ABCMeta):

    def __init__(self, tm, b, preliminary_step: callable = None):
        self.tm, self.b = tm, b
        self.preliminary_step = preliminary_step
        self.shape = np.shape(tm)

        self.result = None

    @staticmethod
    @abc.abstractmethod
    def name():
        pass

    @abc.abstractmethod
    def __call__(self, x):
        pass


class OpsLinesearchCallback(OpsCallback):

    def __init__(self, tm, b,
                 linesearch_params: dict,
                 preliminary_step: callable = None):
        super().__init__(tm, b, preliminary_step)
        self.fun = phrt_opt.utils.define_objective(tm, b)
        self.gradient = phrt_opt.utils.define_gradient(tm, b)
        self.linesearch_params = linesearch_params


class OpsQuadprogCallback(OpsCallback):

    def __init__(self, tm, b,
                 quadprog_params: dict,
                 preliminary_step: callable = None):
        super().__init__(tm, b, preliminary_step)
        self.quadprog_params = quadprog_params


class OpsBacktrackingCallback(OpsLinesearchCallback):

    def __init__(self, tm, b,
                 linesearch_params=typedef.DEFAULT_BACKTRACKING_PARAMS,
                 preliminary_step: callable = None):
        super().__init__(tm, b, linesearch_params, preliminary_step)

    @staticmethod
    def name():
        return "backtracking"

    def __call__(self, x, p=None):
        if p is None:
            p = self.gradient(x)
        args = self.fun, x, p
        if self.preliminary_step is not None:
            args = self.preliminary_step(x)

        backtracking = Backtracking(**self.linesearch_params)
        self.result = backtracking(*args)
        return counters.backtracking_init(*self.shape) + \
               backtracking.it * counters.backtracking_step(*self.shape)


class OpsSecantCallback(OpsLinesearchCallback):

    def __init__(self,
                 ops_backtracking_callback: OpsBacktrackingCallback,
                 linesearch_params=typedef.DEFAULT_LINESEARCH_PARAMS,
                 preliminary_step: callable = None):
        super().__init__(
            ops_backtracking_callback.tm,
            ops_backtracking_callback.b,
            linesearch_params,
            preliminary_step,
        )
        self.prev_x, self.prev_p = None, None
        self.ops_backtracking_callback = ops_backtracking_callback

    @staticmethod
    def name():
        return "secant"

    def __call__(self, x, p=None):
        args = (x,)
        if self.preliminary_step is not None:
            args = self.preliminary_step(x)

        if p is None:
            p = self.gradient(x)

        if self.prev_x is None or self.prev_p is None:
            self.prev_x, self.prev_p = x, p
            ops_count = self.ops_backtracking_callback(x)
            self.result = self.ops_backtracking_callback.result
            return ops_count

        sk, yk = x - self.prev_x, p - self.prev_p
        self.prev_x, self.prev_p = x, p
        ck = np.real(np.vdot(sk, yk))
        if ck > 0:
            self.result =ck / (np.vdot(yk, yk) + np.finfo(float).eps) \
                if self.linesearch_params.get('sym') \
                else np.vdot(sk, sk) / (ck + + np.finfo(float).eps)
            return counters.secant_init(*self.shape) + \
                   counters.secant_step(*self.shape)

        ops_count = self.ops_backtracking_callback(x)
        self.result = self.ops_backtracking_callback.result
        return counters.secant_init(*self.shape) + ops_count


class OpsConjugateGradientCallback(OpsQuadprogCallback):

    def __init__(self, tm, b,
                 quadprog_params=typedef.DEFAULT_CG_PARAMS,
                 preliminary_step: callable = None):
        super().__init__(tm, b, quadprog_params, preliminary_step)

    @staticmethod
    def name():
        return "conjugate_gradient"

    def __call__(self, x):
        args = (x,)
        if self.preliminary_step is not None:
            args = self.preliminary_step(x)

        conjugate_gradient = ConjugateGradient(**self.quadprog_params)
        self.result = conjugate_gradient(*args)

        return counters.conjugate_gradient_init(*self.shape) + \
               conjugate_gradient.it * counters.conjugate_gradient_init(*self.shape)


class OpsCholeskyCallback(OpsQuadprogCallback):

    def __init__(self, tm, b,
                 quadprog_params=typedef.DEFAULT_CHOLESKY_PARAMS,
                 preliminary_step: callable = None):
        super().__init__(tm, b, quadprog_params, preliminary_step)

    @staticmethod
    def name():
        return "cholesky"

    def __call__(self, x):
        args = (x,)
        if self.preliminary_step is not None:
            args = self.preliminary_step(x)

        cholesky = Cholesky(**self.quadprog_params)
        self.result = cholesky(*args)

        return counters.cholesky_init(*self.shape) + \
               cholesky.it * counters.cholesky_step(*self.shape)



class OpsGradientDescentCallback:

    def __init__(self, ops_linesearch_callback: OpsLinesearchCallback):
        self.ops_linesearch_callback = ops_linesearch_callback

    @staticmethod
    def name():
        return "gradient_descent"

    def __call__(self, x):
        return counters.gradient_descent(*self.ops_linesearch_callback.shape) + \
               self.ops_linesearch_callback(x)


class OpsGaussNewtonCallback:

    def __init__(self,
                 ops_linesearch_callback: OpsLinesearchCallback,
                 ops_quadprog_callback: OpsQuadprogCallback):
        self.ops_linesearch_callback = ops_linesearch_callback
        self.ops_quadprog_callback = ops_quadprog_callback

    @staticmethod
    def name():
        return "gauss_newton"

    def __call__(self, x):
        ops_gauss_newton = counters.gauss_newton(*self.ops_linesearch_callback.shape)
        ops_quadprog_count = self.ops_quadprog_callback(x)
        ops_linesearch_count = self.ops_linesearch_callback(
            x, self.ops_quadprog_callback.result[:self.ops_linesearch_callback.shape[1]])
        return ops_gauss_newton + ops_quadprog_count + ops_linesearch_count


class OpsAlternatingProjectionsCallback:

    def __init__(self, shape: tuple):
        self.shape = shape

    @staticmethod
    def name():
        return "alternating_projections"

    def __call__(self, x):
        return counters.alternating_projections(*self.shape)


class OpsADMMCallback:

    def __init__(self, shape: tuple):
        self.shape = shape

    @staticmethod
    def name():
        return "admm"

    def __call__(self, x):
        return counters.admm(*self.shape)


def get_ops(name):
    return {
        OpsBacktrackingCallback.name(): OpsBacktrackingCallback,
        OpsSecantCallback.name(): OpsSecantCallback,
        OpsConjugateGradientCallback.name(): OpsConjugateGradientCallback,
        OpsCholeskyCallback.name(): OpsCholeskyCallback,
        OpsGradientDescentCallback.name(): OpsGradientDescentCallback,
        OpsGaussNewtonCallback.name(): OpsGaussNewtonCallback,
    }[name]