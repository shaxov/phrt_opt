import phrt_opt
import unittest
import numpy as np
from phrt_opt import typedef


class TestCallbacks(unittest.TestCase):

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

    def test_convergence_and_metric_callbacks(self):
        from phrt_opt.callbacks import MetricCallback
        from phrt_opt.callbacks import IsConvergedCallback
        from phrt_opt.metrics import quality_norm

        x = self.x
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                IsConvergedCallback(quality_norm, 1e-6),
                MetricCallback(x, quality_norm),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 1.66825933511916e-07)

    def test_ops_backtracking_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks import OpsGradientDescentCallback
        from phrt_opt.callbacks import OpsBacktrackingCallback

        x = self.x
        x_bar, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                OpsGradientDescentCallback(
                    OpsBacktrackingCallback(self.tm, self.b),
                ),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 7.197200514452717e-06)

    def test_ops_secant_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks import OpsGradientDescentCallback
        from phrt_opt.callbacks import OpsBacktrackingCallback
        from phrt_opt.callbacks import OpsSecantCallback

        x = self.x
        x_bar, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                OpsGradientDescentCallback(
                    OpsSecantCallback(
                        OpsBacktrackingCallback(self.tm, self.b),
                    ),
                ),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 7.197200514452717e-06)

    def test_gauss_newton_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks import OpsConjugateGradientCallback
        from phrt_opt.callbacks import OpsBacktrackingCallback
        from phrt_opt.callbacks import OpsGaussNewtonCallback

        x = self.x
        x_bar, info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                OpsGaussNewtonCallback(
                    OpsBacktrackingCallback(self.tm, self.b),
                    OpsConjugateGradientCallback(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 3.3306690738754696e-16)

    def test_alternating_projections_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks import OpsAlternatingProjectionsCallback

        x = self.x
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                OpsAlternatingProjectionsCallback(np.shape(self.tm)),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 1.66825933511916e-07)

    def test_admm_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks import OpsADMMCallback

        x = self.x
        x_bar, info = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                OpsADMMCallback(np.shape(self.tm)),
            ],
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 3.3306690738754696e-16)