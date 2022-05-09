import numpy as np
from phrt_opt import typedef


class PowerMethod:

    def __init__(self, tol=typedef.DEFAULT_POWER_METHOD_TOL):
        self.tol = tol
        self.it = 0

    @staticmethod
    def name():
        return "power_method"

    @staticmethod
    def _eig(mat, v):
        mat_v_ = mat.dot(v)
        v_conj_ = np.conj(v)
        v_conj_mat_v_ = v_conj_.dot(mat_v_)
        return np.squeeze(v_conj_mat_v_)[()]

    def __call__(self, mat):
        n = np.shape(mat)[1]
        v = np.ones(n) / np.sqrt(n)
        lmd = PowerMethod._eig(mat, v)
        while True:
            mat_v = mat.dot(v)
            v = mat_v / np.linalg.norm(mat_v)
            lmd_n = PowerMethod._eig(mat, v)
            success = np.abs(lmd - lmd_n) < self.tol
            lmd = lmd_n
            self.it += 1
            if success:
                break
        return lmd, v[:, np.newaxis]


def get(name):
    return {
        PowerMethod.name(): PowerMethod,
    }[name]