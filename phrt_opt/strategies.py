import numpy as np
from phrt_opt.metrics import projection_norm
from scipy.optimize import curve_fit


def constant(rho=1.):
    """ Constant strategy for regularization parameter rho. """

    return lambda it, y, z: rho


def linear(thresh_iter=10, rho=.5, thresh_dist=.5):
    """ Linear strategy for regularization parameter rho. """

    return lambda it, y, z: rho / thresh_iter * it \
        if it < thresh_iter and projection_norm(y, z) > thresh_dist else rho


def exponential(thresh_iter=10, rho=.5, thresh_dist=.5):
    """ Exponential strategy for regularization parameter rho. """

    xx = np.array([0, 1.5 * thresh_iter / 2, thresh_iter])
    yy = np.array([0, - (rho / thresh_iter) * xx[1] + rho, rho])

    def func(x, a, b):
        return a * np.exp(-b * x)

    params, _ = curve_fit(func, xx, yy)
    return lambda it, y, z: func(it, *params) \
        if it < thresh_iter and projection_norm(y, z) > thresh_dist else rho


def auto(thresh_dist=0., thresh_rho=1.):

    def _strategy(it, y, z):
        alpha = np.real(np.conj(y) * z) / np.abs(z) ** 2 - 1
        rho = 1. - np.min([1., np.max(-alpha)])
        if projection_norm(y, z) < thresh_dist:
            rho = thresh_rho
        return max(0, rho - 2*rho / np.sqrt(y.size))

    return _strategy


def get(name):
    return {
        "constant": constant,
        "linear": linear,
        "exponential": exponential,
        "auto": auto,
    }[name]
