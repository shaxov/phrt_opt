import numpy as np
from phrt_opt import utils


def wirtinger(tm, b, tol=1e-3, **kwargs):
    m, n = np.shape(tm)
    b2 = np.square(b[..., np.newaxis])
    mat = tm[..., np.newaxis].conj() * tm[:, np.newaxis]
    mat = np.sum(b2 * mat, axis=0) / m

    _, v = utils.power_method(mat, tol, **kwargs)
    lmd = np.linalg.norm(tm, axis=1)
    lmd = np.square(lmd)
    lmd = np.sqrt(n * np.sum(b) / np.sum(lmd))
    return lmd * v
