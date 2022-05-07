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
DEFAULT_BACKTRACKING_PARAMS = dict(
    c1=DEFAULT_BACKTRACKING_C1,
    rate=DEFAULT_BACKTRACKING_RATE,
    max_iter=DEFAULT_BACKTRACKING_MAX_ITER,
    alpha0=DEFAULT_BACKTRACKING_ALPHA0,
)
DEFAULT_LINESEARCH_PARAMS = DEFAULT_BACKTRACKING_PARAMS
DEFAULT_SECANT_LINESEARCH_PARAMS = dict(sym=False)
from phrt_opt import linesearch
DEFAULT_LINESEARCH = linesearch.Backtracking

DEFAULT_CG_TOL = 1e-8
DEFAULT_CG_MAX_ITER = None
DEFAULT_CG_X0 = None
DEFAULT_CG_PARAMS = dict(
    x0=DEFAULT_CG_X0,
    tol=DEFAULT_CG_TOL,
    max_iter=DEFAULT_CG_MAX_ITER,
    dlt=DEFAULT_REG_DLT,
)
DEFAULT_QUADPROG_PARAMS = DEFAULT_CG_PARAMS

from phrt_opt import quadprog
DEFAULT_QUADPROG = quadprog.ConjugateGradient
