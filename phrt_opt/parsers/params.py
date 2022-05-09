from copy import deepcopy
from phrt_opt.strategies import get as get_strategy
from phrt_opt.quadprog import get as get_quadprog
from phrt_opt.linesearch import get as get_linesearch
from phrt_opt.methods import get as get_method
from phrt_opt.initializers import get as get_initializer


def linesearch(params: dict):
    linesearch_class = get_linesearch(params["name"])
    params_params = params.get("params", {})
    if "linesearch" not in params_params:
        return linesearch_class(**params_params)
    params_c = deepcopy(params_params)
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
        parsed_method_params["linesearch"] = linesearch(method_params["linesearch"])
        method_params.pop("linesearch")
    if "quadprog" in method_params:
        parsed_method_params["quadprog"] = quadprog(method_params["quadprog"])
        method_params.pop("quadprog")
    if "strategy" in method_params:
        parsed_method_params["strategy"] = strategy(method_params["strategy"])
        method_params.pop("strategy")
    parsed_method_params.update(method_params)

    return lambda *args, **kwargs: method_function(*args, **kwargs, **parsed_method_params)


def initializer(params: dict, random_state=None):
    initializer_class = get_initializer(params["name"])
    initializer_params = params["params"] if "params" in params else {}
    return initializer_class(**initializer_params, random_state=random_state)
