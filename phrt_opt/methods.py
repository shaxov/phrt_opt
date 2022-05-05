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
                            random_state: np.random.RandomState = None,
                            tm_pinv: np.ndarray = None,
                            **kwargs):
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    def update(x, **kwargs):
        return tm_pinv.dot(b * np.exp(1j * np.angle(tm.dot(x))))

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def gradient_descent(tm, b, *,
                     x0: np.array = None,
                     tol: float = typedef.DEFAULT_TOL,
                     max_iter: int = typedef.DEFAULT_MAX_ITER,
                     metric: callable = typedef.DEFAULT_METRIC,
                     callbacks: typing.List[callable] = None,
                     decorators: typing.List[callable] = None,
                     random_state: np.random.RandomState = None,
                     linesearch_method: callable = typedef.DEFAULT_LINESEARCH_METHOD,
                     **kwargs):
    try:
        linesearch_method = linesearch_method()
    except:
        pass
    dim = np.shape(tm)[1]
    fun = phrt_opt.utils.define_objective(tm, b)
    gradient = phrt_opt.utils.define_gradient(tm, b)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)

    def update(x, **kwargs):
        g = gradient(x)
        return x - linesearch_method(fun, x, g) * g

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def gauss_newton(tm, b, *,
                 x0: np.array = None,
                 tol: float = typedef.DEFAULT_TOL,
                 max_iter: int = typedef.DEFAULT_MAX_ITER,
                 metric: callable = typedef.DEFAULT_METRIC,
                 callbacks: typing.List[callable] = None,
                 decorators: typing.List[callable] = None,
                 random_state: np.random.RandomState = None,
                 quadprog_solver: callable = typedef.DEFAULT_QUADPROG_SOLVER,
                 linesearch_method: callable = typedef.DEFAULT_LINESEARCH_METHOD,
                 **kwargs):
    try:
        linesearch_method = linesearch_method()
    except:
        pass

    dim = np.shape(tm)[1]
    fun = phrt_opt.utils.define_objective(tm, b)
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(dim, random_state)

    def update(x, **kwargs):
        tm_x = np.dot(tm, x)
        r = np.power(np.abs(tm_x), 2) - np.power(b, 2)

        jac = tm * np.conj(tm_x)
        jac = np.concatenate([jac, np.conj(jac)], axis=1)
        jac_h = np.conj(jac.T)

        gk = np.dot(jac_h, r)
        hk = np.dot(jac_h, jac)
        pk = quadprog_solver(hk, -gk)
        pk = pk
        pk = pk[:dim]

        ak = linesearch_method(fun, x, -pk)
        x = x + ak * pk
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def phare_admm(tm, b, *,
               x0: np.array = None,
               tol: float = typedef.DEFAULT_TOL,
               max_iter: int = typedef.DEFAULT_MAX_ITER,
               metric: callable = typedef.DEFAULT_METRIC,
               callbacks: typing.List[callable] = None,
               decorators: typing.List[callable] = None,
               random_state: np.random.RandomState = None,
               tm_pinv: np.ndarray = None,
               rho: float = .5,
               **kwargs):
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)
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

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def dual_ascent(tm, b, *,
                x0: np.array = None,
                tol: float = typedef.DEFAULT_TOL,
                max_iter: int = typedef.DEFAULT_MAX_ITER,
                metric: callable = typedef.DEFAULT_METRIC,
                callbacks: typing.List[callable] = None,
                decorators: typing.List[callable] = None,
                random_state: np.random.RandomState = None,
                tm_pinv: np.ndarray = None,
                **kwargs):
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)
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

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def relaxed_dual_ascent(tm, b, *,
                        x0: np.array = None,
                        tol: float = typedef.DEFAULT_TOL,
                        max_iter: int = typedef.DEFAULT_MAX_ITER,
                        metric: callable = typedef.DEFAULT_METRIC,
                        strategy: callable = typedef.DEFAULT_STRATEGY,
                        callbacks: typing.List[callable] = None,
                        decorators: typing.List[callable] = None,
                        random_state: np.random.RandomState = None,
                        tm_pinv: np.ndarray = None,
                        **kwargs):
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)
    if tm_pinv is None:
        tm_pinv = np.linalg.pinv(tm)

    m = np.shape(tm)[0]
    lmd = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    eps = np.zeros(shape=(m, 1)) + 1j * np.zeros(shape=(m, 1))
    it = 0

    def update(x, **kwargs):
        nonlocal lmd, eps, it
        tm_x = tm.dot(x)
        z = b * np.exp(1j * np.angle(tm_x - eps + lmd))
        x = tm_pinv.dot(z + eps - lmd)
        y = tm.dot(x)
        rho = strategy(it, y, z)
        eps = (rho / (1 + rho)) * (y - z + lmd)
        lmd = lmd + y - z - eps
        it += 1
        return x

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def accelerated_relaxed_dual_ascent(tm, b, *,
                                    x0: np.array = None,
                                    tol: float = typedef.DEFAULT_TOL,
                                    max_iter: int = typedef.DEFAULT_MAX_ITER,
                                    metric: callable = typedef.DEFAULT_METRIC,
                                    callbacks: typing.List[callable] = None,
                                    decorators: typing.List[callable] = None,
                                    random_state: np.random.RandomState = None,
                                    tm_pinv: np.ndarray = None,
                                    rho: float = .5,
                                    restart_freq: int = 3,
                                    restart_rate: float = 0.15,
                                    lmd_tol: float = 1e-2,
                                    **kwargs):
    it = 1
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)
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

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def garda(tm, b, *,
          x0: np.array = None,
          tol: float = typedef.DEFAULT_TOL,
          max_iter: int = typedef.DEFAULT_MAX_ITER,
          metric: callable = typedef.DEFAULT_METRIC,
          callbacks: typing.List[callable] = None,
          decorators: typing.List[callable] = None,
          random_state: np.random.RandomState = None,
          tm_pinv: np.ndarray = None,
          rho: float = .5,
          restart_rate: float = 0.,
          lmd_tol: float = 1e-2,
          **kwargs):
    it = 1
    if x0 is None:
        x0 = phrt_opt.utils.random_x0(np.shape(tm)[1], random_state)
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

    return loop(update, x0, tol, max_iter, metric, callbacks, decorators, **kwargs)


def get(name):
    return {
        "alternating_projections": alternating_projections,
        "dual_ascent": dual_ascent,
        "relaxed_dual_ascent": relaxed_dual_ascent,
        "accelerated_relaxed_dual_ascent": accelerated_relaxed_dual_ascent,
        "phare_admm": phare_admm,
        "garda": garda,
        "gauss_newton": gauss_newton,
        "gradient_descent": gradient_descent,
    }[name]
