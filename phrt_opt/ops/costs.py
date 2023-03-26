import numpy as np

TIME_CONFIG = dict(
    add=np.float64(0.1089426) * 1e-8,
    sub=np.float64(0.0834372) * 1e-8,
    prod=np.float64(0.1665280) * 1e-8,
    div=np.float64(0.3297135) * 1e-8,
    abs=np.float64(0.2500649) * 1e-8,
    sqrt=np.float64(0.1788236) * 1e-8,
    sin=np.float64(1.4113816) * 1e-8,
    cos=np.float64(1.1528648) * 1e-8,
    tan=np.float64(2.5544987) * 1e-8,
    atan=np.float64(2.2179606) * 1e-8,
    exp=np.float64(1.3695341) * 1e-8,
)

UNIT_CONFIG = dict(
    add=np.float64(1.),
    sub=np.float64(1.),
    prod=np.float64(1.),
    div=np.float64(1.),
    abs=np.float64(1.),
    sqrt=np.float64(1.),
    sin=np.float64(1.),
    cos=np.float64(1.),
    tan=np.float64(1.),
    atan=np.float64(1.),
    exp=np.float64(1.),
)


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(_AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class RealFLOPS(_AttrDict):

    def __init__(self, config):
        super().__init__(**config)

    def dot(self, dim):
        return self.prod * dim + self.add * (dim - 1)


class ComplexFLOPS:

    def __init__(self, config):
        self._real_flops = RealFLOPS(config)

    @property
    def add(self):
        return 2 * self._real_flops.add

    @property
    def sub(self):
        return 2 * self._real_flops.sub

    @property
    def prod(self):
        return 4 * self._real_flops.prod + 2 * self._real_flops.add

    @property
    def abs(self):
        return 2 * self._real_flops.prod + self._real_flops.add + self._real_flops.sqrt

    def dot(self, dim):
        return 4 * self._real_flops.prod * dim + 2 * self._real_flops.add * (2 * dim - 1)

    @property
    def angle(self):
        return self._real_flops.atan + self._real_flops.div

    @property
    def exp(self):
        return self._real_flops.cos + self._real_flops.sin

    @property
    def div(self):
        return 8 * self.prod + 3 * self.add + self.sub
