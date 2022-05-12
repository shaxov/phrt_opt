import abc
import numpy as np
import phrt_opt.utils


class _Initializer:

    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.RandomState()

    @staticmethod
    @abc.abstractmethod
    def name():
        pass


class Random(_Initializer):

    @staticmethod
    def name():
        return "random"

    def __call__(self, tm, b):
        """ Random starting point generation. """
        dim = np.shape(tm)[1]
        x0 = phrt_opt.utils.random_x0(dim, self.random_state)
        return x0


class Wirtinger(_Initializer):
    """ Starting point computation via Wirtinger flow [1].

        Reference:
            [1] Candes, Emmanuel & Soltanolkotabi, Mahdi. (2014). Phase Retrieval via Wirtinger Flow: Theory and Algorithms.
            IEEE Transactions on Information Theory. 61. 10.1109/TIT.2015.2399924.
    """

    def __init__(self, eig: callable, random_state=None):
        super().__init__(random_state=random_state)
        self.eig = eig

    @staticmethod
    def name():
        return "wirtinger"

    @staticmethod
    def compute_initialization_matrix(tm, b):
        m, n = np.shape(tm)
        b2 = np.square(b[..., np.newaxis])
        mat = tm[..., np.newaxis].conj() * tm[:, np.newaxis]
        return np.sum(b2 * mat, axis=0) / m

    def __call__(self, tm, b):
        _, n = np.shape(tm)
        matrix = Wirtinger.compute_initialization_matrix(tm, b)
        _, v = self.eig(matrix)
        lmd = np.linalg.norm(tm, axis=1)
        lmd = np.square(lmd)
        lmd = np.sqrt(n * np.sum(b) / np.sum(lmd))
        x0 = lmd * v
        return x0


def get(name):
    return {
        Random.name(): Random,
        Wirtinger.name(): Wirtinger,
    }[name]
