import numpy as np
from phrt_opt import typedef
from phrt_opt.ops import counters


def power_method(mat, tol=1e-3, **kwargs):
    n = np.shape(mat)[1]
    sub_ops = kwargs.get(typedef.DECORATOR_OPS_COUNT_NAME, [0])
    sub_ops[0] += counters.eig(n)

    def eig(mat_, v_):
        mat_v_ = mat_.dot(v_)
        v_conj_ = np.conj(v_)
        v_conj_mat_v_ = v_conj_.dot(mat_v_)
        return np.squeeze(v_conj_mat_v_)[()]

    v = np.ones(n) / np.sqrt(n)
    lmd = eig(mat, v)
    while True:
        sub_ops[0] += counters.power_method_step(n)

        mat_v = mat.dot(v)
        v = mat_v / np.linalg.norm(mat_v)
        lmd_n = eig(mat, v)
        success = np.abs(lmd - lmd_n) < tol
        lmd = lmd_n
        if success:
            break
    return lmd, v[:, np.newaxis]