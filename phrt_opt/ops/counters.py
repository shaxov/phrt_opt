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
