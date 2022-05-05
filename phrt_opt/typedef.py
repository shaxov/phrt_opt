from phrt_opt import metrics
from phrt_opt import quadprog
from phrt_opt import linesearch
from phrt_opt import strategies

DEFAULT_TOL = 1e-8
DEFAULT_MAX_ITER = 1000
DEFAULT_METRIC = metrics.quality_norm
DEFAULT_STRATEGY = strategies.constant_strategy()
DECORATOR_INFO_KEY = "dinfo"
CALLBACK_INFO_KEY = "cinfo"
DECORATOR_OPS_COUNT_NAME = "_ops_count"

DEFAULT_QUADPROG_SOLVER = quadprog.conjugate_gradient_solver
DEFAULT_LINESEARCH_METHOD = linesearch.backtracking
