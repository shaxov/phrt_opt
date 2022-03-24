import numpy as np
from numpy.linalg import norm
from phrt_opt.metrics import projection
from scipy.optimize import curve_fit


def constant_strategy(rho=.5):
    """ Constant strategy for regularization parameter rho. """

    return lambda it, y, z: rho


def linear_strategy(thresh_iter=10, rho=.5, thresh_dist=.5):
    """ Linear strategy for regularization parameter rho. """

    return lambda it, y, z: rho / thresh_iter * it \
        if it < thresh_iter and projection(y, z) / norm(z) > thresh_dist else rho


def exponential_strategy(thresh_iter=10, rho=.5, thresh_dist=.5):
    """ Exponential strategy for regularization parameter rho. """

    xx = np.array([0, 1.5 * thresh_iter / 2, thresh_iter])
    yy = np.array([0, - (rho / thresh_iter) * xx[1] + rho, rho])

    def func(x, a, b):
        return a * np.exp(-b * x)

    params, _ = curve_fit(func, xx, yy)
    return lambda it, y, z: func(it, *params) \
        if it < thresh_iter and projection(y, z) / norm(z) > thresh_dist else rho


def get(name):
    return {
        "const": constant_strategy,
        "linear": linear_strategy,
        "expon": exponential_strategy,
    }[name]
