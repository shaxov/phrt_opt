import phrt_opt
import unittest
import numpy as np


class TestEig(unittest.TestCase):

    def setUp(self):
        m, n = 64, 64
        self.seed = 2
        self.random_state = np.random.RandomState(self.seed)
        self.mat = self.random_state.randn(n, n) + 1j * self.random_state.randn(n, n)

    def test_power_method(self):
        eig = phrt_opt.eig.PowerMethod(tol=1e-12)
        lmd, v = eig(self.mat)
        self.assertAlmostEquals(np.linalg.norm(self.mat @ v - lmd * v), 0.)

        eval, evec = np.linalg.eig(self.mat)
        self.assertAlmostEquals(np.abs(eval).max(), np.abs(lmd))


