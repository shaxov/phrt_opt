import phrt_opt
import unittest
import numpy as np


class TestInitializers(unittest.TestCase):

    def setUp(self):
        m, n = 64, 16
        self.seed = 2
        self.random_state = np.random.RandomState(self.seed)

        self.num_beams = n
        self.num_measurements = m
        self.x = np.exp(1j * self.random_state.uniform(-np.pi, np.pi, size=(n, 1)))
        self.x0 = np.exp(1j * self.random_state.uniform(-np.pi, np.pi, size=(n, 1)))
        self.tm = self.random_state.randn(m, n) + 1j * self.random_state.randn(m, n)
        self.b = np.abs(self.tm.dot(self.x))

    def test_random_initializer(self):
        initializer = phrt_opt.initializers.Random(self.random_state)
        x0 = initializer(self.tm, self.b)

        q = phrt_opt.metrics.quality_norm(self.x, x0)
        self.assertAlmostEquals(q, 0.9746962824)

    def test_wirtinger_initializer(self):
        initializer = phrt_opt.initializers.Wirtinger(
            eig=phrt_opt.eig.PowerMethod(1e-2), 
            random_state=self.random_state,
        )
        x0 = initializer(self.tm, self.b)

        q = phrt_opt.metrics.quality_norm(self.x, x0)
        self.assertAlmostEquals(q, 0.8670098401)

        initializer = phrt_opt.initializers.Wirtinger(
            eig=phrt_opt.eig.PowerMethod(1e-8), 
            random_state=self.random_state,
        )
        x0 = initializer(self.tm, self.b)

        q = phrt_opt.metrics.quality_norm(self.x, x0)
        self.assertAlmostEquals(q, 0.86502822)

    def test_gao_xu_initializer(self):
        initializer = phrt_opt.initializers.GaoXu(
            eig=phrt_opt.eig.PowerMethod(1e-2), 
            random_state=self.random_state,
        )
        x0 = initializer(self.tm, self.b)

        q = phrt_opt.metrics.quality_norm(self.x, x0)
        self.assertAlmostEquals(q, 0.692435820)

        initializer = phrt_opt.initializers.GaoXu(
            eig=phrt_opt.eig.PowerMethod(1e-8), 
            random_state=self.random_state,
        )
        x0 = initializer(self.tm, self.b)

        q = phrt_opt.metrics.quality_norm(self.x, x0)
        self.assertAlmostEquals(q, 0.662985768)