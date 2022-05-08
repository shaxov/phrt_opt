from phrt_opt.ops import costs


r_ops_costs = costs.RealFLOPS(costs.TIME_CONFIG)
c_ops_costs = costs.ComplexFLOPS(costs.TIME_CONFIG)


def c_mat_vec_dot(m, n):
    return m * c_ops_costs.dot(n)


def r_mat_vec_dot(m, n):
    return m * r_ops_costs.dot(n)


def c_norm_vec(n):
    return n * c_ops_costs.prod + (n - 1) * c_ops_costs.add + r_ops_costs.sqrt


def fun_eval(m, n):
    flops = m * c_ops_costs.dot(n)
    flops += m * (c_ops_costs.abs + r_ops_costs.prod)
    flops += m * (r_ops_costs.prod + r_ops_costs.sub)
    flops += m * r_ops_costs.prod
    flops += (m - 1) * r_ops_costs.add + r_ops_costs.div * r_ops_costs.prod
    return flops


def alternating_projections(m, n):
    ops_cnt = c_mat_vec_dot(m, n)          # Ax = [a1Tx, ..., amTx]
    ops_cnt += m * c_ops_costs.angle           # phi = angle(Ax)
    ops_cnt += m * c_ops_costs.exp             # e_phi = exp(1j * phi)
    ops_cnt += 2 * m * r_ops_costs.prod        # z = b * e_phi
    ops_cnt += c_mat_vec_dot(n, m)         # x = A_pinv * z
    return ops_cnt


def admm(m, n):
    ops_cnt = c_mat_vec_dot(m, n)                                             # Ax = A*x = [a1Tx, ..., amTx]
    ops_cnt += m * (c_ops_costs.add + c_ops_costs.sub)                            # g = Ax - eps + lmd
    ops_cnt += m * c_ops_costs.angle                                              # tht = angle(g)
    ops_cnt += m * (c_ops_costs.exp + 2 * r_ops_costs.prod)                       # b_exp_tht = b * exp(1j * tht)
    ops_cnt += c_mat_vec_dot(n, m) + m * (c_ops_costs.add + c_ops_costs.sub)  # x = A_pinv*(b_exp_tht + eps - lmd)
    ops_cnt += m * (2 * c_ops_costs.sub + c_ops_costs.add)                        # reg = Ax - b_exp_tht - eps
    ops_cnt += m * c_ops_costs.add                                                # res = res + reg
    return ops_cnt


def gauss_newton(m, n):
    flops = m * c_ops_costs.dot(n)
    flops += m * c_ops_costs.abs
    flops += m * r_ops_costs.prod
    flops += m * (r_ops_costs.prod + r_ops_costs.sub)
    flops += m * (2 * r_ops_costs.prod)
    flops += m * n * c_ops_costs.prod
    flops += m * n * 6 * r_ops_costs.prod
    flops += 2 * n * c_ops_costs.dot(m)
    flops += 2 * n * c_ops_costs.dot(m) * 2 * n
    flops += n * 2 * r_ops_costs.prod
    return flops


def gradient_descent(m, n):
    # gk = gradient(x)
    flops = m * c_ops_costs.dot(n)
    flops += m * (c_ops_costs.abs + 2 * r_ops_costs.prod * r_ops_costs.sub)
    flops += m * (2 * r_ops_costs.prod)
    flops += n * (c_ops_costs.dot(m) + 2 * r_ops_costs.div)
    # x += -alpha * gk
    flops += n * 2 * r_ops_costs.prod
    return flops


def backtracking_init(m, n):
    flops = c_ops_costs.dot(n) + r_ops_costs.sqrt + n * r_ops_costs.div
    flops += r_ops_costs.prod
    flops += fun_eval(m, n)
    return flops


def backtracking_step(m, n):
    flops = n * 2 * r_ops_costs.prod
    flops += fun_eval(m, n) + 2 * r_ops_costs.prod
    flops += r_ops_costs.prod
    return flops


def secant_init(m, n):
    flops = c_ops_costs.dot(n) + 2 * c_ops_costs.sub * n
    return flops


def secant_step(m, n):
    flops = c_ops_costs.dot(n)
    return flops


def conjugate_gradient_init(m, n):
    flops = n * (c_ops_costs.dot(n) + c_ops_costs.sub)
    flops += n * 2 * r_ops_costs.prod
    return flops


def conjugate_gradient_step(m, n):
    flops = c_ops_costs.dot(n) + n * r_ops_costs.prod
    flops += n * c_ops_costs.dot(n)
    flops += 2 * n * (c_ops_costs.dot(n) + r_ops_costs.prod) + r_ops_costs.div
    flops += 2 * n * 2 * r_ops_costs.prod + n * c_ops_costs.add
    flops += 2 * n * (c_ops_costs.dot(n) + r_ops_costs.prod) + r_ops_costs.div
    flops += n * 2 * r_ops_costs.prod + n * c_ops_costs.add + 2 * n * r_ops_costs.prod
    return flops


def cholesky_init(m, n):
    return 0.


def cholesky_step(m, n):
    flops = (4 / 3) * (2 * n) ** 3 * r_ops_costs.add
    flops += (2 * n) * (2 * n - 1) * (c_ops_costs.sub + c_ops_costs.prod)
    return flops


def power_method_init(n):
    return 0.


def power_method_step(n):
    ops_cnt = c_mat_vec_dot(n, n)
    ops_cnt += c_norm_vec(n) + 2 * n * r_ops_costs.div
    ops_cnt += c_mat_vec_dot(n, n) + c_ops_costs.dot(n)
    ops_cnt += r_ops_costs.sub
    return ops_cnt


def wirtinger(m, n):
    ops_cnt = m * r_ops_costs.prod
    ops_cnt += m * n * n * (c_ops_costs.prod + r_ops_costs.prod)
    ops_cnt += (m - 1) * n * n * c_ops_costs.add
    ops_cnt += 2 * n * n * r_ops_costs.div
    ops_cnt += m * (c_norm_vec(n) + r_ops_costs.prod)
    ops_cnt += 2 * (m - 1) * r_ops_costs.add + r_ops_costs.div + r_ops_costs.prod + r_ops_costs.sqrt
    return ops_cnt


def get(name):
    return {
        "admm": admm,
        "gauss_newton": gauss_newton,
        "gradient_descent": gradient_descent,
        "alternating_projections": alternating_projections,

        "secant": (secant_init, secant_step),
        "cholesky": (cholesky_init, cholesky_step),
        "backtracking": (backtracking_init, backtracking_step),
        "conjugate_gradient": (conjugate_gradient_init, conjugate_gradient_step),

        "power_method": (power_method_init, power_method_step),

        "wirtinger": wirtinger,
    }[name]
