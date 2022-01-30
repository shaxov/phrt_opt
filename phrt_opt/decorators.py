def ops_count(count):
    def _update_wrap(update):
        def _update(*args, **kwargs):
            x = update(*args, **kwargs)
            dinfo = []
            if isinstance(x, tuple):
                x, dinfo = x
            sub_ops_cnt = kwargs.get('sub_ops', [0]).pop()
            dinfo.append(count + sub_ops_cnt)
            return x, dinfo
        return _update
    return _update_wrap
