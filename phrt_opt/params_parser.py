from copy import deepcopy
from phrt_opt.linesearch import get as get_linesearch


def linesearch(params: dict):
    callable_obj = get_linesearch(params["name"])
    if "linesearch" not in params:
        return callable_obj(**params["params"])
    params_c = deepcopy(params["params"])
    nested = params_c.pop("linesearch")
    return callable_obj(linesearch=linesearch(nested), **params_c)