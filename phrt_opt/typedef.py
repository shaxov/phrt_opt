DEFAULT_TOL = 1e-8
DEFAULT_REG_DLT = 1e-8
DEFAULT_MAX_ITER = 1000
DEFAULT_POWER_METHOD_TOL = 1e-3

from phrt_opt import metrics
DEFAULT_METRIC = metrics.quality_norm

from phrt_opt import strategies
DEFAULT_RHO_STRATEGY = strategies.constant_strategy()

DEFAULT_BACKTRACKING_RATE = .5
DEFAULT_BACKTRACKING_MAX_ITER = 100
DEFAULT_BACKTRACKING_ALPHA0 = 1.
DEFAULT_BACKTRACKING_C1 = 1e-4
from phrt_opt import linesearch
DEFAULT_LINESEARCH = linesearch.backtracking

DEFAULT_CG_TOL = 1e-8
DEFAULT_CG_MAX_ITER = None

from phrt_opt import quadprog
DEFAULT_QUADPROG_SOLVER = quadprog.conjugate_gradient_solver
