from phrt_opt.ops import costs


r_ops_costs = costs.RealFLOPS(costs.TIME_CONFIG)
c_ops_costs = costs.ComplexFLOPS(costs.TIME_CONFIG)


def alternating_projections(m, n):
    ops_cnt = m * c_ops_costs.vdot(n)          # Ax = [a1Tx, ..., amTx]
    ops_cnt += m * c_ops_costs.angle           # phi = angle(Ax)
    ops_cnt += m * c_ops_costs.exp             # e_phi = exp(1j * phi)
    ops_cnt += 2 * m * r_ops_costs.prod        # z = b * e_phi
    ops_cnt += n * c_ops_costs.vdot(m)         # x = A_pinv * z
    return ops_cnt


def phare_admm(m, n):
    ops_cnt = m * c_ops_costs.vdot(n)                                                               # Ax = A*x = [a1Tx, ..., amTx]
    ops_cnt += 2 * m * r_ops_costs.div + m * c_ops_costs.add                                        # g = Ax - lmd / rho
    ops_cnt += m * c_ops_costs.angle                                                                # tht = angle(g)
    ops_cnt += m * (c_ops_costs.abs + 2 * r_ops_costs.prod + r_ops_costs.add + r_ops_costs.div)     # u = (rho * abs(g) + b) / (rho + 1)
    ops_cnt += m * (c_ops_costs.exp + 2 * r_ops_costs.prod)                                         # u_exp_tht = u * exp(1j * tht)
    ops_cnt += m * c_ops_costs.sub                                                                  # y = u_exp_tht - lmd / rho
    ops_cnt += n * c_ops_costs.vdot(m)                                                              # x = A_pinv*y
    ops_cnt += m * c_ops_costs.sub                                                                  # reg = Ax - u_exp_tht
    ops_cnt += m * 2 * (r_ops_costs.prod + r_ops_costs.add)                                         # lmd = lmd + rho * reg
    return ops_cnt


def dual_ascent(m, n):
    ops_cnt = m * c_ops_costs.vdot(n)                        # Ax = A*x = [a1Tx, ..., amTx]
    ops_cnt += m * c_ops_costs.add                           # g = Ax + res
    ops_cnt += m * c_ops_costs.angle                         # tht = angle(g)
    ops_cnt += m * (c_ops_costs.exp + 2 * r_ops_costs.prod)  # b_exp_tht = b * exp(1j * tht)
    ops_cnt += n * c_ops_costs.vdot(m)                       # x = A_pinv*b_exp_tht
    ops_cnt += m * c_ops_costs.sub                           # reg = Ax - u_exp_tht
    ops_cnt += m * c_ops_costs.add                           # res = res + reg
    return ops_cnt
