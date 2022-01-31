from phrt_opt import typedef


def ops_count(count):
    def _func_wrap(func):
        def _func(*args, **kwargs):
            try:
                kwargs.update({typedef.DECORATOR_OPS_COUNT_NAME: [0]})
                x = func(*args, **kwargs)
            except TypeError:
                kwargs.pop(typedef.DECORATOR_OPS_COUNT_NAME)
                x = func(*args, **kwargs)
            dinfo = []
            if isinstance(x, tuple):
                x, dinfo = x
            sub_ops_cnt = kwargs.get(typedef.DECORATOR_OPS_COUNT_NAME, [0]).pop()
            dinfo.append(count + sub_ops_cnt)
            return x, dinfo
        return _func
    return _func_wrap
