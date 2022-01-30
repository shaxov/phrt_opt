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
        self.tm = random.randn(m, n) + 1j * random.randn(m, n)
        self.b = np.abs(self.tm.dot(self.x))

    def test_alternating_projections_ops_count_decorator(self):
        from phrt_opt.decorators import ops_count
        from phrt_opt.ops.counters import alternating_projections

        x = self.x
        n = self.num_beams
        m = self.num_measurements
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
            decorators=[ops_count(alternating_projections(m, n))],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.)

    def test_phare_admm_ops_count_decorator(self):
        from phrt_opt.decorators import ops_count
        from phrt_opt.ops.counters import phare_admm

        x = self.x
        n = self.num_beams
        m = self.num_measurements
        x_bar, info = phrt_opt.methods.phare_admm(
            self.tm, self.b,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
            decorators=[ops_count(phare_admm(m, n))],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.)

    def test_dual_ascent_ops_count_decorator(self):
        from phrt_opt.decorators import ops_count
        from phrt_opt.ops.counters import dual_ascent

        x = self.x
        n = self.num_beams
        m = self.num_measurements
        x_bar, info = phrt_opt.methods.dual_ascent(
            self.tm, self.b,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
            decorators=[ops_count(dual_ascent(m, n))],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.)

    def test_relaxed_dual_ascent_ops_count_decorator(self):
        from phrt_opt.decorators import ops_count
        from phrt_opt.ops.counters import relaxed_dual_ascent

        x = self.x
        n = self.num_beams
        m = self.num_measurements
        x_bar, info = phrt_opt.methods.relaxed_dual_ascent(
            self.tm, self.b,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
            decorators=[ops_count(relaxed_dual_ascent(m, n))],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.)
