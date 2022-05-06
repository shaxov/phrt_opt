import numpy as np
from phrt_opt import typedef


def power_method(mat, tol=typedef.DEFAULT_POWER_METHOD_TOL):
    n = np.shape(mat)[1]

    def eig(mat_, v_):
        mat_v_ = mat_.dot(v_)
        v_conj_ = np.conj(v_)
        v_conj_mat_v_ = v_conj_.dot(mat_v_)
        return np.squeeze(v_conj_mat_v_)[()]

    v = np.ones(n) / np.sqrt(n)
    lmd = eig(mat, v)
    while True:
        mat_v = mat.dot(v)
        v = mat_v / np.linalg.norm(mat_v)
        lmd_n = eig(mat, v)
        success = np.abs(lmd - lmd_n) < tol
        lmd = lmd_n
        if success:
            break
    return lmd, v[:, np.newaxis]


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


def random_x0(dim, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    x0 = random_state.randn(dim, 1) + 1j * random_state.randn(dim, 1)
    return x0
