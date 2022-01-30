import typing
import typedef
import numpy as np


def _loop(
        update: callable,
        x0: typing.Union[np.array, int] = None,
        tol: float = typedef.DEFAULT_TOL,
        max_iter: int = typedef.DEFAULT_MAX_ITER,
        metric: callable = typedef.DEFAULT_METRIC,
        callbacks: typing.List[callable] = None,
        decorators: typing.List[callable] = None,
        seed: int = None,
):
    pass
