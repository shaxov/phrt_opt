from copy import deepcopy
from phrt_opt.strategy import get as get_strategy
from phrt_opt.quadprog import get as get_quadprog
from phrt_opt.linesearch import get as get_linesearch
from phrt_opt.methods import get as get_method


def linesearch(params: dict):
    linesearch_class = get_linesearch(params["name"])
    if "linesearch" not in params:
        return linesearch_class(**params["params"])
    params_c = deepcopy(params["params"])
    nested = params_c.pop("linesearch")
    return linesearch_class(linesearch=linesearch(nested), **params_c)


def quadprog(params: dict):
    return get_quadprog(params["name"])(**params["params"])


def strategy(params: dict):
    return get_strategy(params["name"])(**params["params"])


def method(params: dict):
    method_function = get_method(params["name"])

    parsed_method_params = {}
    method_params = deepcopy(params["params"])
    if "linesearch" in method_params:
        parsed_method_params["linesearch"] = linesearch(method_params.pop("linesearch"))
    if "quadprog" in method_params:
        parsed_method_params["quadprog"] = quadprog(method_params.pop("quadprog"))
    if "strategy" in method_params:
        parsed_method_params["strategy"] = strategy(method_params.pop("strategy"))
    parsed_method_params.update(method_params)

    return lambda *args, **kwargs: method_function(*args, **kwargs, **parsed_method_params)