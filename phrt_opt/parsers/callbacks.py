import numpy as np
import phrt_opt.utils
from phrt_opt.callbacks import counters


def _linesearch(tm, b, params: dict, preliminary_step: callable = None):
    if params["name"] == counters.BacktrackingCallback.name():
        return counters.BacktrackingCallback(
            tm, b, params["params"], preliminary_step=preliminary_step)
    if params["name"] == counters.SecantCallback.name():
        linesearch_params = params["params"]
        return counters.SecantCallback(
                counters.BacktrackingCallback(
                    tm, b, linesearch_params["linesearch"]["params"]
                ),
                linesearch_params,
                preliminary_step=preliminary_step,
            )
    raise ValueError(f"Linesearch with name '{params['name']}' is not valid.")


def _quadprog(tm, b, params: dict, preliminary_step: callable = None):
    if params["name"] == counters.ConjugateGradientCallback.name():
        return counters.ConjugateGradientCallback(
            tm, b, params["params"], preliminary_step=preliminary_step)
    if params["name"] == counters.CholeskyCallback.name():
        return counters.CholeskyCallback(
            tm, b, params["params"], preliminary_step=preliminary_step)
    raise ValueError(f"Quadprog with name '{params['name']}' is not valid.")


def gradient_descent(tm, b, params: dict):
    return counters.GradientDescentCallback(_linesearch(tm, b, params["linesearch"]))


def gauss_newton(tm, b, params: dict):
    return counters.GaussNewtonCallback(
        _linesearch(tm, b, params["linesearch"]),
        _quadprog(tm, b, params["quadprog"],
                  preliminary_step=phrt_opt.utils.define_gauss_newton_system(tm, b))
    )

def alternating_projections(tm, b, params: dict):
    return counters.AlternatingProjectionsCallback(np.shape(tm))


def admm(tm, b, params: dict):
    return counters.ADMMCallback(np.shape(tm))


def random(params: dict):
    return counters.RandomInitializationCallback()


def wirtinger(params: dict):
    from phrt_opt.initializers import Wirtinger

    return counters.WirtingerInitializationCallback(
        counters.get(params["eig"]["name"])(
            params["eig"]["params"],
            preliminary_step=Wirtinger.compute_initialization_matrix,
        )
    )


def gao_xu(params: dict):
    from phrt_opt.initializers import GaoXu

    return counters.WirtingerInitializationCallback(
        counters.get(params["eig"]["name"])(
            params["eig"]["params"],
            preliminary_step=GaoXu.compute_initialization_matrix,
        )
    )



def get(name):
    return {
        "admm": admm,
        "alternating_projections": alternating_projections,
        "gradient_descent": gradient_descent,
        "gauss_newton": gauss_newton,

        "random": random,
        "wirtinger": wirtinger,
        "gao_xu": gao_xu,
    }[name]