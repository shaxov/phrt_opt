import numpy as np
from phrt_opt import utils


def random(tm, b, **kwargs):
    random_state = kwargs.get('random_state', None)
    if random_state is None:
        random_state = np.random.RandomState()
    x0 = utils.random_x0(np.shape(tm)[1], random_state)
    return x0


def wirtinger(tm, b, tol=1e-3, **kwargs):
    m, n = np.shape(tm)
    b2 = np.square(b[..., np.newaxis])
    mat = tm[..., np.newaxis].conj() * tm[:, np.newaxis]
    mat = np.sum(b2 * mat, axis=0) / m

    _, v = utils.power_method(mat, tol, **kwargs)
    lmd = np.linalg.norm(tm, axis=1)
    lmd = np.square(lmd)
    lmd = np.sqrt(n * np.sum(b) / np.sum(lmd))
    x0 = lmd * v
    return x0


def get(name):
    return {
        "random": random,
        "wirtinger": wirtinger,
    }[name]
