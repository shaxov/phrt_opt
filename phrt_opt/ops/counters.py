from phrt_opt.ops import costs


r_ops_costs = costs.RealFLOPS(costs.TIME_CONFIG)
c_ops_costs = costs.ComplexFLOPS(costs.TIME_CONFIG)


def c_mat_vec_dot(m, n):
    return m * c_ops_costs.dot(n)


def r_mat_vec_dot(m, n):
    return m * r_ops_costs.dot(n)


def c_norm_vec(n):
    return n * c_ops_costs.prod + (n - 1) * c_ops_costs.add + r_ops_costs.sqrt


def alternating_projections(m, n):
    ops_cnt = c_mat_vec_dot(m, n)          # Ax = [a1Tx, ..., amTx]
    ops_cnt += m * c_ops_costs.angle           # phi = angle(Ax)
    ops_cnt += m * c_ops_costs.exp             # e_phi = exp(1j * phi)
    ops_cnt += 2 * m * r_ops_costs.prod        # z = b * e_phi
    ops_cnt += c_mat_vec_dot(n, m)         # x = A_pinv * z
    return ops_cnt


def phare_admm(m, n):
    ops_cnt = c_mat_vec_dot(m, n)                                                               # Ax = A*x = [a1Tx, ..., amTx]
    ops_cnt += 2 * m * r_ops_costs.div + m * c_ops_costs.add                                        # g = Ax - lmd / rho
    ops_cnt += m * c_ops_costs.angle                                                                # tht = angle(g)
    ops_cnt += m * (c_ops_costs.abs + 2 * r_ops_costs.prod + r_ops_costs.add + r_ops_costs.div)     # u = (rho * abs(g) + b) / (rho + 1)
    ops_cnt += m * (c_ops_costs.exp + 2 * r_ops_costs.prod)                                         # u_exp_tht = u * exp(1j * tht)
    ops_cnt += m * c_ops_costs.sub                                                                  # y = u_exp_tht - lmd / rho
    ops_cnt += c_mat_vec_dot(n, m)                                                              # x = A_pinv*y
    ops_cnt += m * c_ops_costs.sub                                                                  # reg = Ax - u_exp_tht
    ops_cnt += m * 2 * (r_ops_costs.prod + r_ops_costs.add)                                         # lmd = lmd + rho * reg
    return ops_cnt


def dual_ascent(m, n):
    ops_cnt = c_mat_vec_dot(m, n)                        # Ax = A*x = [a1Tx, ..., amTx]
    ops_cnt += m * c_ops_costs.add                           # g = Ax + res
    ops_cnt += m * c_ops_costs.angle                         # tht = angle(g)
    ops_cnt += m * (c_ops_costs.exp + 2 * r_ops_costs.prod)  # b_exp_tht = b * exp(1j * tht)
    ops_cnt += c_mat_vec_dot(n, m)                       # x = A_pinv*b_exp_tht
    ops_cnt += m * c_ops_costs.sub                           # reg = Ax - u_exp_tht
    ops_cnt += m * c_ops_costs.add                           # res = res + reg
    return ops_cnt


def relaxed_dual_ascent(m, n):
    ops_cnt = c_mat_vec_dot(m, n)                                             # Ax = A*x = [a1Tx, ..., amTx]
    ops_cnt += m * (c_ops_costs.add + c_ops_costs.sub)                            # g = Ax - eps + lmd
    ops_cnt += m * c_ops_costs.angle                                              # tht = angle(g)
    ops_cnt += m * (c_ops_costs.exp + 2 * r_ops_costs.prod)                       # b_exp_tht = b * exp(1j * tht)
    ops_cnt += c_mat_vec_dot(n, m) + m * (c_ops_costs.add + c_ops_costs.sub)  # x = A_pinv*(b_exp_tht + eps - lmd)
    ops_cnt += m * (2 * c_ops_costs.sub + c_ops_costs.add)                        # reg = Ax - b_exp_tht - eps
    ops_cnt += m * c_ops_costs.add                                                # res = res + reg
    return ops_cnt


accelerated_relaxed_dual_ascent = relaxed_dual_ascent


def eig(n):
    return c_mat_vec_dot(n, n) + c_ops_costs.dot(n)


def power_method_step(n):
    ops_cnt = c_mat_vec_dot(n, n)
    ops_cnt += c_norm_vec(n) + 2 * n * r_ops_costs.div
    ops_cnt += eig(n)
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
