import typing
import typedef
import numpy as np
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
