import typing
import numpy as np
import phrt_opt.utils
from phrt_opt import metrics
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
                            tm_pinv: np.ndarray = None,
                            **kwargs):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    def update(x, **kwargs):
        return tm_pinv.dot(b * np.exp(1j * np.angle(tm.dot(x))))

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed, **kwargs)


def phare_admm(tm, b, *,
               x0: np.array = None,
               tol: float = typedef.DEFAULT_TOL,
               max_iter: int = typedef.DEFAULT_MAX_ITER,
               metric: callable = typedef.DEFAULT_METRIC,
               callbacks: typing.List[callable] = None,
               decorators: typing.List[callable] = None,
               seed: int = None,
               tm_pinv: np.ndarray = None,
               rho: float = .5,
               **kwargs):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x, **kwargs):
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

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed, **kwargs)


def dual_ascent(tm, b, *,
                x0: np.array = None,
                tol: float = typedef.DEFAULT_TOL,
                max_iter: int = typedef.DEFAULT_MAX_ITER,
                metric: callable = typedef.DEFAULT_METRIC,
                callbacks: typing.List[callable] = None,
                decorators: typing.List[callable] = None,
                seed: int = None,
                tm_pinv: np.ndarray = None,
                **kwargs):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x, **kwargs):
        nonlocal lmd
        tm_x = tm.dot(x)
        b_exp_tht = b * np.exp(1j * np.angle(tm_x + lmd))
        x = tm_pinv.dot(b_exp_tht)
        lmd = lmd + tm.dot(x) - b_exp_tht
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed, **kwargs)


def relaxed_dual_ascent(tm, b, *,
                        x0: np.array = None,
                        tol: float = typedef.DEFAULT_TOL,
                        max_iter: int = typedef.DEFAULT_MAX_ITER,
                        metric: callable = typedef.DEFAULT_METRIC,
                        callbacks: typing.List[callable] = None,
                        decorators: typing.List[callable] = None,
                        seed: int = None,
                        tm_pinv: np.ndarray = None,
                        rho: float = .5,
                        **kwargs):
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    eps = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x, **kwargs):
        nonlocal lmd, eps
        tm_x = tm.dot(x)
        z = b * np.exp(1j * np.angle(tm_x - eps + lmd))
        x = tm_pinv.dot(z + eps - lmd)
        y = tm.dot(x)
        eps = (rho / (1 + rho)) * (y - z + lmd)
        lmd = lmd + y - z - eps
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed, **kwargs)


def accelerated_relaxed_dual_ascent(tm, b, *,
                                    x0: np.array = None,
                                    tol: float = typedef.DEFAULT_TOL,
                                    max_iter: int = typedef.DEFAULT_MAX_ITER,
                                    metric: callable = typedef.DEFAULT_METRIC,
                                    callbacks: typing.List[callable] = None,
                                    decorators: typing.List[callable] = None,
                                    seed: int = None,
                                    tm_pinv: np.ndarray = None,
                                    rho: float = .5,
                                    restart_freq: int = 3,
                                    restart_rate: float = 0.15,
                                    lmd_tol: float = 1e-2,
                                    **kwargs):
    it = 1
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    eps = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x, **kwargs):
        nonlocal lmd, eps, it
        tm_x = tm.dot(x)
        lmd_dist = metrics.quality(tm_x - eps + lmd, tm_x)
        z = b * np.exp(1j * np.angle(tm_x - eps + lmd))
        x = tm_pinv.dot(z + eps - lmd)
        y = tm.dot(x)
        eps = (rho / (1 + rho)) * (y - z + lmd)
        lmd += (y - z - eps)
        if lmd_dist < lmd_tol:
            if it % restart_freq == 0:
                lmd *= restart_rate
        it += 1
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed, **kwargs)


def garda(tm, b, *,
          x0: np.array = None,
          tol: float = typedef.DEFAULT_TOL,
          max_iter: int = typedef.DEFAULT_MAX_ITER,
          metric: callable = typedef.DEFAULT_METRIC,
          callbacks: typing.List[callable] = None,
          decorators: typing.List[callable] = None,
          seed: int = None,
          tm_pinv: np.ndarray = None,
          rho: float = .5,
          restart_rate: float = 0.,
          lmd_tol: float = 1e-2,
          **kwargs):
    it = 1
    if x0 is None:
        x0 = np.shape(tm)[1]
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)
    gradient = phrt_opt.utils.define_gradient(tm, b)
    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    eps = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))

    def update(x, **kwargs):
        nonlocal lmd, eps, it
        x_prev = x
        tm_x = tm.dot(x)
        lmd_dist = metrics.quality(tm_x - eps + lmd, tm_x)
        z = b * np.exp(1j * np.angle(tm_x - eps + lmd))
        x = tm_pinv.dot(z + eps - lmd)
        y = tm.dot(x)
        eps = (rho / (1 + rho)) * (y - z + lmd)
        lmd += (y - z - eps)
        if lmd_dist < lmd_tol and np.real(np.vdot(gradient(x), x - x_prev)) > 0:
            lmd *= restart_rate
        it += 1
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, seed, **kwargs)
