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
            seed=self.seed,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.00024642286613196784)

    def test_phare_admm(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.phare_admm(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 2.682048674373938e-05)

    def test_dual_ascent(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.dual_ascent(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.0004532004668830192)

    def test_relaxed_dual_ascent(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.relaxed_dual_ascent(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 2.2636871992776086e-05)

    def test_accelerated_relaxed_dual_ascent(self):
        x, x0 = self.x, self.x0
        x_bar = phrt_opt.methods.accelerated_relaxed_dual_ascent(
            self.tm, self.b,
            x0=x0,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.00143715897585861)
