import typing
import numpy as np
import phrt_opt.utils
from phrt_opt import typedef
from phrt_opt.loop import loop
from phrt_opt import linesearch as ls


@ls.make_instance_if_class
def gradient_descent(tm, b, *,
                     x0: np.array = None,
                     tol: float = typedef.DEFAULT_TOL,
                     max_iter: int = typedef.DEFAULT_MAX_ITER,
                     metric: callable = typedef.DEFAULT_METRIC,
                     callbacks: typing.List[callable] = None,
                     random_state: np.random.RandomState = None,
                     linesearch: callable = typedef.DEFAULT_LINESEARCH):
    dim = np.shape(tm)[1]
    fun = phrt_opt.utils.define_objective(tm, b)
    gradient = phrt_opt.utils.define_gradient(tm, b)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)

    def update(x):
        g = gradient(x)
        return x - linesearch(fun, x, g) * g

    return loop(update, x0, tol, max_iter, metric, callbacks)


@ls.make_instance_if_class
def gauss_newton(tm, b, *,
                 x0: np.array = None,
                 tol: float = typedef.DEFAULT_TOL,
                 max_iter: int = typedef.DEFAULT_MAX_ITER,
                 metric: callable = typedef.DEFAULT_METRIC,
                 callbacks: typing.List[callable] = None,
                 random_state: np.random.RandomState = None,
                 quadprog_solver: callable = typedef.DEFAULT_QUADPROG_SOLVER,
                 linesearch: callable = typedef.DEFAULT_LINESEARCH):
    dim = np.shape(tm)[1]
    fun = phrt_opt.utils.define_objective(tm, b)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)

    def update(x):
        tm_x = np.dot(tm, x)
        r = np.power(np.abs(tm_x), 2) - np.power(b, 2)

        jac = tm * np.conj(tm_x)
        jac = np.concatenate([jac, np.conj(jac)], axis=1)
        jac_h = np.conj(jac.T)

        gk = np.dot(jac_h, r)
        hk = np.dot(jac_h, jac)

        pk = quadprog_solver(hk, -gk)[:dim]
        ak = linesearch(fun, x, -pk)

        return x + ak * pk

    return loop(update, x0, tol, max_iter, metric, callbacks)


def alternating_projections(tm, b, *,
                            x0: np.array = None,
                            tol: float = typedef.DEFAULT_TOL,
                            max_iter: int = typedef.DEFAULT_MAX_ITER,
                            metric: callable = typedef.DEFAULT_METRIC,
                            callbacks: typing.List[callable] = None,
                            random_state: np.random.RandomState = None,
                            tm_pinv: np.ndarray = None):
    if x0 is None:
        dim = np.shape(tm)[1]
        x0 = phrt_opt.utils.random_x0(dim, random_state)
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    def update(x):
        return tm_pinv.dot(b * np.exp(1j * np.angle(tm.dot(x))))

    return loop(update, x0, tol, max_iter, metric, callbacks)


def admm(tm, b, *,
         x0: np.array = None,
         tol: float = typedef.DEFAULT_TOL,
         max_iter: int = typedef.DEFAULT_MAX_ITER,
         metric: callable = typedef.DEFAULT_METRIC,
         rho_strategy: callable = typedef.DEFAULT_RHO_STRATEGY,
         callbacks: typing.List[callable] = None,
         random_state: np.random.RandomState = None,
         tm_pinv: np.ndarray = None):
    m, dim = dim = np.shape(tm)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    eps = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    it = 0

    def update(x):
        nonlocal lmd, eps, it
        tm_x = tm.dot(x)
        z = b * np.exp(1j * np.angle(tm_x - eps + lmd))
        x = tm_pinv.dot(z + eps - lmd)
        y = tm.dot(x)
        rho = rho_strategy(it, y, z)
        eps = (rho / (1 + rho)) * (y - z + lmd)
        lmd = lmd + y - z - eps
        it += 1
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks)


def get(name):
    return {
        "admm": admm,
        "gauss_newton": gauss_newton,
        "gradient_descent": gradient_descent,
        "alternating_projections": alternating_projections,
    }[name]
