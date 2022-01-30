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
