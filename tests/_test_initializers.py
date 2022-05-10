import phrt_opt
import unittest
import numpy as np


class TestInitializers(unittest.TestCase):

    def setUp(self):
        m, n = 20, 5
        self.seed = 2
        random = np.random.RandomState(self.seed)

        self.num_beams = n
        self.num_measurements = m
        self.x = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        self.tm = random.randn(m, n) + 1j * random.randn(m, n)
        self.b = np.abs(self.tm.dot(self.x))

    def test_wirtinger_initializer(self):
        from phrt_opt import utils
        b = self.b
        tm = self.tm
        tol = 1e-3
        x0 = phrt_opt.initializers.wirtinger(self.tm, self.b, tol=tol)

        m, n = np.shape(tm)
        y_mat = 0
        for bi, ai in zip(b.ravel(), tm[:, :, None]):
            y_mat += bi ** 2 * ai.conj().dot(ai.T)
        y_mat /= m
        _, v = utils.power_method(y_mat, tol)
        tm_row_norm = np.linalg.norm(tm, axis=1) ** 2
        lmd = np.sqrt(n * np.sum(b) / np.sum(tm_row_norm))
        x0_ref = lmd * v
        self.assertTrue(np.allclose(x0, x0_ref))