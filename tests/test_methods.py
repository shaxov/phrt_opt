import phrt_opt
import unittest
import numpy as np


class TestMethods(unittest.TestCase):

    def setUp(self):
        m, n = 20, 5
        self.seed = 2
        random = np.random.RandomState(self.seed)

        self.num_beams = n
        self.num_measurements = m
        self.x = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        self.x0 = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        self.tm = random.randn(m, n) + 1j * random.randn(m, n)
        self.b = np.abs(self.tm.dot(self.x))

    def test_alternating_projections(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.0002464228)

    def test_gradient_descent(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.000445933)

    def test_gauss_newton(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 3.881517329773487e-09)

    def test_admm(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
            strategy=phrt_opt.strategies.auto(),
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 4.161275528302699e-05)
