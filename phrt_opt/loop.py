import typing
import numpy as np

from phrt_opt import typedef


def _check_callable(obj):
    if not callable(obj):
        raise ValueError(f"'{obj}' argument must be callable.")


def _check_x0(x0):
    if isinstance(x0, int):
        if x0 <= 0:
            raise ValueError("If 'x0' is integer, then it must be positive.")
    elif isinstance(x0, np.ndarray):
        if np.shape(x0) != 2:
            raise ValueError("If 'x0' is numpy array, then it must be with shape (n, 1), where n > 1.")
        if np.shape(x0)[1] != 1:
            raise ValueError("If 'x0' is numpy array, then the last dimension must be equal to 1.")
    else:
        raise ValueError(f"Invalid type of 'x0'. Expected types are 'int', 'numpy.ndarray' but found '{type(x0)}'.")


def _check_tol(tol):
    if not isinstance(tol, float):
        raise ValueError(f"'tol' argument must have a 'float' type, but found '{type(tol)}'.")
    if tol < 0:
        raise ValueError("'tol' argument must be non negative.")


def _check_max_iter(max_iter):
    if not isinstance(max_iter, int):
        raise ValueError(f"'max_iter' argument must have a 'int' type, but found '{type(max_iter)}'.")
    if max_iter <= 0:
        raise ValueError("'max_iter' argument must be positive.")


def loop(
        update: callable,
        x0: typing.Union[np.array, int],
        tol: float = typedef.DEFAULT_TOL,
        max_iter: int = typedef.DEFAULT_MAX_ITER,
        metric: callable = typedef.DEFAULT_METRIC,
        callbacks: typing.List[callable] = None,
        decorators: typing.List[callable] = None,
        seed: int = None,
        **kwargs,
):
    _check_callable(update)
    _check_x0(x0)
    _check_tol(tol)
    _check_max_iter(max_iter)
    _check_callable(metric)
    if callbacks:
        for callback in callbacks:
            _check_callable(callback)
    if decorators:
        for decorator in decorators:
            _check_callable(decorator)

    random = np.random.RandomState(seed)
    if isinstance(x0, int):
        x0 = random.randn(x0, 1) + 1j * random.randn(x0, 1)
    x = x0

    info = {}
    if decorators:
        info[typedef.DECORATOR_INFO_KEY] = []
        for decorator in decorators:
            update = decorator(update)
    if callbacks:
        info[typedef.CALLBACK_INFO_KEY] = [
            [callback(x) for callback in callbacks]
        ]

    for _ in range(max_iter):
        x_n = update(x, **kwargs)
        if decorators:
            x_n, dinfo = x_n
            info[typedef.DECORATOR_INFO_KEY].append(dinfo)
        if callbacks:
            info[typedef.CALLBACK_INFO_KEY].append(
                [callback(x_n) for callback in callbacks])
        if metric(x_n, x) < tol and not (callbacks or decorators):
            break
        x = x_n
    if info:
        if callbacks:
            info[typedef.CALLBACK_INFO_KEY] = np.array(info[typedef.CALLBACK_INFO_KEY])
        if decorators:
            info[typedef.DECORATOR_INFO_KEY] = np.array(info[typedef.DECORATOR_INFO_KEY])
        return x, info
    return x
