from phrt_opt import typedef


def ops_count(count):
    def _ops_count(func):
        def _func(*args, **kwargs):
            if typedef.DECORATOR_OPS_COUNT_NAME in kwargs:
                x = func(*args, **kwargs)
                kwargs.get(typedef.DECORATOR_OPS_COUNT_NAME)[0] += count
                return x

            kwargs.update({typedef.DECORATOR_OPS_COUNT_NAME: [0]})
            x = func(*args, **kwargs)
            dinfo = []
            if isinstance(x, tuple):
                x, dinfo = x
            ops_cnt = count + kwargs.get(typedef.DECORATOR_OPS_COUNT_NAME, [0]).pop()
            dinfo.append(ops_cnt)
            return x, dinfo
        return _func
    return _ops_count
