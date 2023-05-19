import abc
import numpy as np
import phrt_opt.utils
from phrt_opt import typedef
from phrt_opt.ops import counters
from phrt_opt.linesearch import Backtracking
from phrt_opt.quadprog import Cholesky
from phrt_opt.quadprog import ConjugateGradient
from phrt_opt.eig import PowerMethod


class _Counter(metaclass=abc.ABCMeta):

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


class _LinesearchCounter(_Counter, metaclass=abc.ABCMeta):

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


class _QuadprogCounter(_Counter, metaclass=abc.ABCMeta):

    def __init__(self, tm, b,
                 quadprog_params: dict,
                 preliminary_step: callable = None):
        super().__init__(tm, b, preliminary_step)
        self.quadprog_params = quadprog_params


class BacktrackingCounter(_LinesearchCounter):

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


class SecantCounter(_LinesearchCounter):

    def __init__(self,
                 backtracking_callback: BacktrackingCounter,
                 linesearch_params=typedef.DEFAULT_LINESEARCH_PARAMS,
                 preliminary_step: callable = None):
        super().__init__(
            backtracking_callback.tm,
            backtracking_callback.b,
            linesearch_params,
            preliminary_step,
        )
        self.prev_x, self.prev_p = None, None
        self.ops_backtracking_callback = backtracking_callback

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


class ConjugateGradientCounter(_QuadprogCounter):

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


class CholeskyCounter(_QuadprogCounter):

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


class GradientDescentCounter:

    def __init__(self, linesearch_callback: _LinesearchCounter):
        self.linesearch_callback = linesearch_callback

    @staticmethod
    def name():
        return "gradient_descent"

    def __call__(self, x):
        return counters.gradient_descent(*self.linesearch_callback.shape) + \
               self.linesearch_callback(x)


class GaussNewtonCounter:

    def __init__(self,
                 linesearch_callback: _LinesearchCounter,
                 quadprog_callback: _QuadprogCounter):
        self.linesearch_callback = linesearch_callback
        self.quadprog_callback = quadprog_callback

    @staticmethod
    def name():
        return "gauss_newton"

    def __call__(self, x):
        ops_gauss_newton = counters.gauss_newton(*self.linesearch_callback.shape)
        ops_quadprog_count = self.quadprog_callback(x)
        ops_linesearch_count = self.linesearch_callback(
            x, self.quadprog_callback.result[:self.linesearch_callback.shape[1]])
        return ops_gauss_newton + ops_quadprog_count + ops_linesearch_count


class AlternatingProjectionsCounter:

    def __init__(self, shape: tuple):
        self.shape = shape

    @staticmethod
    def name():
        return "alternating_projections"

    def __call__(self, x):
        return counters.alternating_projections(*self.shape)


class ADMMCounter:

    def __init__(self, shape: tuple):
        self.shape = shape

    @staticmethod
    def name():
        return "admm"

    def __call__(self, x):
        return counters.admm(*self.shape)


class _EigCounter(metaclass=abc.ABCMeta):

    def __init__(self, preliminary_step: callable = None):
        self.preliminary_step = preliminary_step
        self.result = None


class PowerMethodCounter(_EigCounter):

    def __init__(self,
                 power_method_params: dict,
                 preliminary_step: callable = None):
        super().__init__(preliminary_step)
        self.power_method = PowerMethod(power_method_params.get('tol'))

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


class RandomInitializationCounter:

    @staticmethod
    def name():
        return "random"

    def __call__(self, tm, b):
        return 0.


class WirtingerInitializationCounter:

    def __init__(self, eig_callback: _EigCounter):
        self.eig_callback = eig_callback

    @staticmethod
    def name():
        return "wirtinger"

    def __call__(self, tm, b):
        return counters.wirtinger(*np.shape(tm)) + self.eig_callback(tm, b)


class GaoXuInitializationCounter:

    def __init__(self, eig_callback: _EigCounter):
        self.eig_callback = eig_callback

    @staticmethod
    def name():
        return "gao_xu"

    def __call__(self, tm, b):
        return counters.gao_xu(*np.shape(tm)) + self.eig_callback(tm, b)


def get(name):
    return {
        BacktrackingCounter.name(): BacktrackingCounter,
        SecantCounter.name(): SecantCounter,
        ConjugateGradientCounter.name(): ConjugateGradientCounter,
        CholeskyCounter.name(): CholeskyCounter,
        GradientDescentCounter.name(): GradientDescentCounter,
        GaussNewtonCounter.name(): GaussNewtonCounter,
        AlternatingProjectionsCounter.name(): AlternatingProjectionsCounter,
        ADMMCounter.name(): ADMMCounter,
        PowerMethodCounter.name(): PowerMethodCounter,
        WirtingerInitializationCounter.name(): WirtingerInitializationCounter,
        RandomInitializationCounter.name(): RandomInitializationCounter,
        GaoXuInitializationCounter.name(): GaoXuInitializationCounter,
    }[name]
