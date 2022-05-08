from phrt_opt.callbacks import counters


def gradient_descent(tm, b, params: dict):
    linesearch = params["linesearch"]
    linesearch_params = linesearch["params"]
    gradient_descent_callback = counters.GradientDescentCallback

    if linesearch["name"] == counters.BacktrackingCallback.name():
        return gradient_descent_callback(
            counters.BacktrackingCallback(tm, b, linesearch_params),
        )
    if linesearch["name"] == counters.SecantCallback.name():
        return gradient_descent_callback(
            counters.SecantCallback(
                counters.BacktrackingCallback(
                    tm, b, linesearch_params["linesearch"]["params"]),
                linesearch_params,
            )
        )
    raise ValueError(f"Linesearch with name '{linesearch['name']}' is not valid.")
