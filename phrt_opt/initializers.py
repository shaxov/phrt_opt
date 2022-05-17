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


class _Spectral(_Initializer):

    def __init__(self, eig: callable, random_state=None):
        super().__init__(random_state=random_state)
        self.eig = eig

    @staticmethod
    @abc.abstractmethod
    def compute_initialization_matrix(tm, b):
        pass


class Wirtinger(_Spectral):
    """ Starting point computation via Wirtinger flow [1].

        Reference:
            [1] Candes, Emmanuel & Soltanolkotabi, Mahdi. (2014). Phase Retrieval via Wirtinger Flow: Theory and Algorithms.
            IEEE Transactions on Information Theory. 61. 10.1109/TIT.2015.2399924.
    """

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


class GaoXu(_Spectral):
    """ Starting point computation via Gao and Xu initialization [1].

        Reference:
            [1] B. Gao and Z. Xu, "Phaseless Recovery Using the Gauss–Newton Method," in
            IEEE Transactions on Signal Processing, vol. 65, no. 22, pp. 5885-5896, 15 Nov.15, 2017,
             doi: 10.1109/TSP.2017.2742981.
    """

    @staticmethod
    def name():
        return "gao_xu"

    @staticmethod
    def compute_initialization_matrix(tm, b):
        m, n = np.shape(tm)
        b2 = np.square(b[..., np.newaxis])
        lmd = np.sum(b2) / m
        beta = .5 - np.exp(-b2 / lmd)
        mat = tm[..., np.newaxis].conj() * tm[:, np.newaxis]
        return np.sum(beta * mat, axis=0) / m

    def __call__(self, tm, b):
        m, n = np.shape(tm)
        matrix = GaoXu.compute_initialization_matrix(tm, b)
        _, v = self.eig(matrix)
        lmd = np.sum(b**2) / m
        x0 = lmd * v
        return x0


def get(name):
    return {
        Random.name(): Random,
        Wirtinger.name(): Wirtinger,
        GaoXu.name(): GaoXu,
    }[name]
