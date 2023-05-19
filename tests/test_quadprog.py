import phrt_opt
import unittest
import numpy as np


class TestQuadprog(unittest.TestCase):

    def setUp(self):
        m, n = 50, 50
        self.seed = 2
        random = np.random.RandomState(self.seed)

        self.x = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        self.x0 = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        mat = random.randn(m, n) + 1j * random.randn(m, n)
        self.mat = mat.conj().T @ mat / n
        self.b = self.mat.dot(self.x)

    def test_conjugate_gradient(self):
        cg = phrt_opt.quadprog.ConjugateGradient(x0=self.x0)
        x_hat = cg(self.mat, self.b)
        self.assertTrue(np.allclose(x_hat, self.x))

    def test_cholesky(self):
        cg = phrt_opt.quadprog.Cholesky()
        x_hat = cg(self.mat, self.b)
        self.assertTrue(np.allclose(x_hat, self.x))
