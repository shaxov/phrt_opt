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
        if it < thresh_iter and projection(y, z) / max(norm(z), norm(y)) > thresh_dist else rho


def distance_strategy(min_norm_dist=0.):
    """ Distance based strategy for regularization parameter rho. """

    return lambda it, y, z: 1 - np.clip(projection(y, z) / max(norm(z), norm(y)) - min_norm_dist, 0., 1.)


def average_distance_strategy(min_norm_dist=0., avg_factor=0.5):
    """ Distance based strategy for regularization parameter rho. """

    prev_dist = None

    def _strategy(it, y, z):
        nonlocal prev_dist
        if prev_dist is None:
            prev_dist = 1 - np.clip(projection(y, z) / max(norm(z), norm(y)) - min_norm_dist, 0., 1.)
            return 0.
        dist = 1 - np.clip(projection(y, z) / max(norm(z), norm(y)) - min_norm_dist, 0., 1.)
        dist = (1 - avg_factor) * dist + avg_factor * prev_dist
        prev_dist = dist
        return dist

    return _strategy


def rho_opt_strategy(thresh_dist=0.):

    def _strategy(it, y, z):
        alpha = np.real(np.conj(y) * z) / np.abs(z) ** 2 - 1
        rho = 1 - np.min([1., np.max([0., np.max(-alpha)])])
        if projection(y, z) / max(norm(z), norm(y)) < thresh_dist:
            rho = 1.
        return rho

    return _strategy


def get(name):
    return {
        "const": constant_strategy,
        "linear": linear_strategy,
        "expon": exponential_strategy,
        "dist": distance_strategy,
        "avg_dist": average_distance_strategy,
        "rho_opt": rho_opt_strategy,
    }[name]
