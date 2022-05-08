from copy import deepcopy
from phrt_opt.quadprog import get as get_quadprog
from phrt_opt.linesearch import get as get_linesearch


def linesearch(params: dict):
    linesearch_class = get_linesearch(params["name"])
    if "linesearch" not in params:
        return linesearch_class(**params["params"])
    params_c = deepcopy(params["params"])
    nested = params_c.pop("linesearch")
    return linesearch_class(linesearch=linesearch(nested), **params_c)


def quadprog(params: dict):
    return get_quadprog(params["name"])(**params["params"])
