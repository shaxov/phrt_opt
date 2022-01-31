import phrt_opt
import unittest
import numpy as np


class TestCallbacks(unittest.TestCase):

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
        from phrt_opt.callbacks import MetricCallback
        from phrt_opt.callbacks import ConvergenceCallback
        from phrt_opt.metrics import quality_norm

        x = self.x
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            tol=1e-6,
            max_iter=100,
            seed=self.seed,
            callbacks=[
                ConvergenceCallback(quality_norm, 1e-6),
                MetricCallback(x, quality_norm),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 0.)
