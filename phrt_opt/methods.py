import typing
import numpy as np
from phrt_opt import typedef
from phrt_opt.loop import loop


def alternating_projections(tm, b, *,
                            x0: np.array = None,
                            tol: float = typedef.DEFAULT_TOL,
                            max_iter: int = typedef.DEFAULT_MAX_ITER,
                            metric: callable = typedef.DEFAULT_METRIC,
                            callbacks: typing.List[callable] = None,
                            decorators: typing.List[callable] = None,
                            seed: int = None,
                            tm_pinv: np.ndarray = None):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    def update(x):
        return tm_pinv.dot(b * np.exp(1j * np.angle(tm.dot(x))))

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed)


def phare_admm(tm, b, *,
               x0: np.array = None,
               tol: float = typedef.DEFAULT_TOL,
               max_iter: int = typedef.DEFAULT_MAX_ITER,
               metric: callable = typedef.DEFAULT_METRIC,
               callbacks: typing.List[callable] = None,
               decorators: typing.List[callable] = None,
               seed: int = None,
               tm_pinv: np.ndarray = None,
               rho: float = .5):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x):
        nonlocal lmd
        tm_x = tm.dot(x)
        g = tm_x + lmd / rho
        tht = np.angle(g)
        u = (rho * np.abs(g) + b) / (rho + 1)
        u_exp_tht = u * np.exp(1j * tht)
        y = u_exp_tht - lmd / rho
        x = tm_pinv.dot(y)
        reg = tm.dot(x) - u_exp_tht
        lmd = lmd + rho * reg
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed)


def dual_ascent(tm, b, *,
                x0: np.array = None,
                tol: float = typedef.DEFAULT_TOL,
                max_iter: int = typedef.DEFAULT_MAX_ITER,
                metric: callable = typedef.DEFAULT_METRIC,
                callbacks: typing.List[callable] = None,
                decorators: typing.List[callable] = None,
                seed: int = None,
                tm_pinv: np.ndarray = None):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x):
        nonlocal lmd
        tm_x = tm.dot(x)
        b_exp_tht = b * np.exp(1j * np.angle(tm_x + lmd))
        x = tm_pinv.dot(b_exp_tht)
        lmd = lmd + tm.dot(x) - b_exp_tht
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed)
