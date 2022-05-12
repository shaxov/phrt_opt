import numpy as np


def define_objective(tm, b):
    m, n = np.shape(tm)

    def fun(x):
        tm_x = np.dot(tm, x)
        y = np.power(np.abs(tm_x), 2)
        y -= np.power(b, 2)
        y = np.power(y, 2)
        y = np.sum(y, axis=0) / (2 * m)
        return np.real(y)
    fun.tm_shape = (m, n)
    return fun


def define_gradient(tm, b):
    m, n = np.shape(tm)
    tm_h = np.transpose(np.conj(tm))

    def gradient(x):
        tm_x = np.dot(tm, x)
        r = np.abs(tm_x) ** 2 - b ** 2
        r_tm_x = np.multiply(r, tm_x)
        grad = np.dot(tm_h, r_tm_x) / m
        return grad

    return gradient


def define_gauss_newton_system(tm, b):

    def system(x):
        tm_x = np.dot(tm, x)
        r = np.power(np.abs(tm_x), 2) - np.power(b, 2)

        jac = tm * np.conj(tm_x)
        jac = np.concatenate([jac, np.conj(jac)], axis=1)
        jac_h = np.conj(jac.T)

        gk = np.dot(jac_h, r)
        hk = np.dot(jac_h, jac)
        return hk, gk

    return system


def random_x0(dim, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    x0 = random_state.randn(dim, 1) + 1j * random_state.randn(dim, 1)
    return x0
