import typing
import numpy as np
import phrt_opt.utils
from phrt_opt import typedef
from phrt_opt.loop import loop


def gradient_descent(tm, b, *,
                     x0: np.array = None,
                     tol: float = typedef.DEFAULT_TOL,
                     max_iter: int = typedef.DEFAULT_MAX_ITER,
                     metric: callable = typedef.DEFAULT_METRIC,
                     callbacks: typing.List[callable] = None,
                     random_state: np.random.RandomState = None,
                     linesearch: callable = typedef.DEFAULT_LINESEARCH,
                     persist_iterations: bool = False):
    dim = np.shape(tm)[1]
    fun = phrt_opt.utils.define_objective(tm, b)
    gradient = phrt_opt.utils.define_gradient(tm, b)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)

    def update(x):
        g = gradient(x)
        return x - linesearch(fun, x, g) * g

    return loop(update, x0, tol, max_iter, metric, callbacks, persist_iterations)


def gauss_newton(tm, b, *,
                 x0: np.array = None,
                 tol: float = typedef.DEFAULT_TOL,
                 max_iter: int = typedef.DEFAULT_MAX_ITER,
                 metric: callable = typedef.DEFAULT_METRIC,
                 callbacks: typing.List[callable] = None,
                 random_state: np.random.RandomState = None,
                 quadprog: callable = typedef.DEFAULT_QUADPROG,
                 linesearch: callable = typedef.DEFAULT_LINESEARCH,
                 persist_iterations: bool = False):
    dim = np.shape(tm)[1]
    fun = phrt_opt.utils.define_objective(tm, b)
    gauss_newton_system = phrt_opt.utils.define_gauss_newton_system(tm, b)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)

    def update(x):
        hk, gk = gauss_newton_system(x)
        pk = quadprog(hk, -gk)[:dim]
        ak = linesearch(fun, x, -pk)
        return x + ak * pk

    return loop(update, x0, tol, max_iter, metric, callbacks, persist_iterations)


def alternating_projections(tm, b, *,
                            x0: np.array = None,
                            tol: float = typedef.DEFAULT_TOL,
                            max_iter: int = typedef.DEFAULT_MAX_ITER,
                            metric: callable = typedef.DEFAULT_METRIC,
                            callbacks: typing.List[callable] = None,
                            random_state: np.random.RandomState = None,
                            tm_pinv: np.ndarray = None,
                            persist_iterations: bool = False):
    if x0 is None:
        dim = np.shape(tm)[1]
        x0 = phrt_opt.utils.random_x0(dim, random_state)
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    def update(x):
        return tm_pinv.dot(b * np.exp(1j * np.angle(tm.dot(x))))

    return loop(update, x0, tol, max_iter, metric, callbacks, persist_iterations)


def admm(tm, b, *,
         x0: np.array = None,
         tol: float = 1e-4,
         max_iter: int = typedef.DEFAULT_MAX_ITER,
         metric: callable = typedef.DEFAULT_METRIC,
         strategy: callable = typedef.DEFAULT_STRATEGY,
         callbacks: typing.List[callable] = None,
         random_state: np.random.RandomState = None,
         tm_pinv: np.ndarray = None,
         persist_iterations: bool = False):
    m, dim = np.shape(tm)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    rho = 0
    it = 0

    def update(x):
        nonlocal lmd, rho, it
        tm_x = tm.dot(x)
        z = b * np.exp(1j * np.angle(tm_x + (1 - rho) * lmd))
        x = tm_pinv.dot(z)
        y = tm.dot(x)
        rho = strategy(it, y, z)
        lmd = (1 / (1 + rho)) * (lmd + y - z)
        it += 1
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, persist_iterations)


def get(name):
    return {
        "admm": admm,
        "gauss_newton": gauss_newton,
        "gradient_descent": gradient_descent,
        "alternating_projections": alternating_projections,
    }[name]
