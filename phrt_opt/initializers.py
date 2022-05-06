import numpy as np
from phrt_opt import utils
from phrt_opt import typedef


def random(tm, b, random_state=None):
    """ Random starting point generation. """
    if random_state is None:
        random_state = np.random.RandomState()
    dim = np.shape(tm)[1]
    x0 = utils.random_x0(dim, random_state)
    return x0


def wirtinger(tm, b, tol=typedef.DEFAULT_POWER_METHOD_TOL):
    """ Starting point computation via Wirtinger flow [1].

    Reference:
        [1] Candes, Emmanuel & Soltanolkotabi, Mahdi. (2014). Phase Retrieval via Wirtinger Flow: Theory and Algorithms.
         IEEE Transactions on Information Theory. 61. 10.1109/TIT.2015.2399924.
    """
    m, n = np.shape(tm)
    b2 = np.square(b[..., np.newaxis])
    mat = tm[..., np.newaxis].conj() * tm[:, np.newaxis]
    mat = np.sum(b2 * mat, axis=0) / m

    _, v = utils.power_method(mat, tol)
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
