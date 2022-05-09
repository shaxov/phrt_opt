import abc
import numpy as np
import phrt_opt.utils
from phrt_opt import typedef
from phrt_opt.ops import counters
from phrt_opt.linesearch import Backtracking
from phrt_opt.quadprog import Cholesky
from phrt_opt.quadprog import ConjugateGradient


class _Callback(metaclass=abc.ABCMeta):

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


class _LinesearchCallback(_Callback, metaclass=abc.ABCMeta):

    def __init__(self, tm, b,
                 linesearch_params: dict,
                 preliminary_step: callable = None):
        super().__init__(tm, b, preliminary_step)
        self.fun = phrt_opt.utils.define_objective(tm, b)
        self.gradient = phrt_opt.utils.define_gradient(tm, b)
        self.linesearch_params = linesearch_params

    @abc.abstractmethod
    def __call__(self, x, p=None):
        pass


class _QuadprogCallback(_Callback, metaclass=abc.ABCMeta):

    def __init__(self, tm, b,
                 quadprog_params: dict,
                 preliminary_step: callable = None):
        super().__init__(tm, b, preliminary_step)
        self.quadprog_params = quadprog_params


class BacktrackingCallback(_LinesearchCallback):

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


class SecantCallback(_LinesearchCallback):

    def __init__(self,
                 ops_backtracking_callback: BacktrackingCallback,
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
        if self.preliminary_step is not None:
            x = self.preliminary_step(x)

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
            self.result = ck / (np.vdot(yk, yk) + np.finfo(float).eps) \
                if self.linesearch_params.get('sym') \
                else np.vdot(sk, sk) / (ck + np.finfo(float).eps)
            return counters.secant_init(*self.shape) + \
                   counters.secant_step(*self.shape)

        ops_count = self.ops_backtracking_callback(x)
        self.result = self.ops_backtracking_callback.result
        return counters.secant_init(*self.shape) + ops_count


class ConjugateGradientCallback(_QuadprogCallback):

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


class CholeskyCallback(_QuadprogCallback):

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


class GradientDescentCallback:

    def __init__(self, ops_linesearch_callback: _LinesearchCallback):
        self.ops_linesearch_callback = ops_linesearch_callback

    @staticmethod
    def name():
        return "gradient_descent"

    def __call__(self, x):
        return counters.gradient_descent(*self.ops_linesearch_callback.shape) + \
               self.ops_linesearch_callback(x)


class GaussNewtonCallback:

    def __init__(self,
                 ops_linesearch_callback: _LinesearchCallback,
                 ops_quadprog_callback: _QuadprogCallback):
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


class AlternatingProjectionsCallback:

    def __init__(self, shape: tuple):
        self.shape = shape

    @staticmethod
    def name():
        return "alternating_projections"

    def __call__(self, x):
        return counters.alternating_projections(*self.shape)


class ADMMCallback:

    def __init__(self, shape: tuple):
        self.shape = shape

    @staticmethod
    def name():
        return "admm"

    def __call__(self, x):
        return counters.admm(*self.shape)


class PowerMethodCallback:

    def __init__(self,
                 tol=typedef.DEFAULT_POWER_METHOD_TOL,
                 preliminary_step: callable = None):
        self.power_method = phrt_opt.utils.PowerMethod(tol)
        self.preliminary_step = preliminary_step
        self.result = None

    @staticmethod
    def name():
        return "power_method"

    def __call__(self, *args):
        mat = args[0]
        if self.preliminary_step is not None:
            mat = self.preliminary_step(*args)
        n = np.shape(mat)[0]
        self.result = self.power_method(mat)
        return counters.power_method_init(n) + \
               self.power_method.it * counters.power_method_step(n)


def get(name):
    return {
        BacktrackingCallback.name(): BacktrackingCallback,
        SecantCallback.name(): SecantCallback,
        ConjugateGradientCallback.name(): ConjugateGradientCallback,
        CholeskyCallback.name(): CholeskyCallback,
        GradientDescentCallback.name(): GradientDescentCallback,
        GaussNewtonCallback.name(): GaussNewtonCallback,
        AlternatingProjectionsCallback.name(): AlternatingProjectionsCallback,
        ADMMCallback.name(): ADMMCallback,
        PowerMethodCallback.name(): PowerMethodCallback,
    }[name]
