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
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 1.66825933511916e-07)

    def test_ops_backtracking_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import GradientDescentCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback

        x = self.x
        x_bar, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GradientDescentCallback(
                    BacktrackingCallback(self.tm, self.b),
                ),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 7.197200514452717e-06)

    def test_ops_secant_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import GradientDescentCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback
        from phrt_opt.callbacks.counters import SecantCallback

        x = self.x
        x_bar, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GradientDescentCallback(
                    SecantCallback(
                        BacktrackingCallback(self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 7.197200514452717e-06)

    def test_gauss_newton_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import ConjugateGradientCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback
        from phrt_opt.callbacks.counters import GaussNewtonCallback

        x = self.x
        x_bar, info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GaussNewtonCallback(
                    BacktrackingCallback(self.tm, self.b),
                    ConjugateGradientCallback(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 3.3306690738754696e-16)

    def test_alternating_projections_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import AlternatingProjectionsCallback

        x = self.x
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                AlternatingProjectionsCallback(np.shape(self.tm)),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 1.66825933511916e-07)

    def test_admm_callback(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import ADMMCallback

        x = self.x
        x_bar, info = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                ADMMCallback(np.shape(self.tm)),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 1.668259338449829e-07)

    def test_gradient_descent_backtracking_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gradient_descent
        from phrt_opt.callbacks.counters import GradientDescentCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback

        x = self.x
        _, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                gradient_descent(self.tm, self.b, {
                    "linesearch": {
                        "name": "backtracking",
                        "params": {
                            "max_iter": 15,
                        }
                    }
                })
            ],
            persist_iterations=True,
        )
        _, ref_info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GradientDescentCallback(
                    BacktrackingCallback(self.tm, self.b),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gradient_descent_secant_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gradient_descent
        from phrt_opt.callbacks.counters import GradientDescentCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback
        from phrt_opt.callbacks.counters import SecantCallback

        x = self.x
        _, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                gradient_descent(self.tm, self.b, {
                    "linesearch": {
                        "name": "secant",
                        "params": {
                            "linesearch": {
                                "name": "backtracking",
                                "params": {
                                    "max_iter": 100,
                                },
                            },
                            "sym": False,
                        }
                    }
                })
            ],
            persist_iterations=True,
        )
        _, ref_info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GradientDescentCallback(
                    SecantCallback(
                        BacktrackingCallback(self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gauss_newton_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gauss_newton
        from phrt_opt.callbacks.counters import GaussNewtonCallback
        from phrt_opt.callbacks.counters import ConjugateGradientCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback

        x = self.x
        _, info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                gauss_newton(self.tm, self.b, {
                    "linesearch": {
                        "name": "backtracking",
                        "params": {
                            "max_iter": 100,
                        }
                    },
                    "quadprog": {
                        "name": "conjugate_gradient",
                        "params": {
                            "dlt": 1e-8,
                        }
                    }
                })
            ],
            persist_iterations=True,
        )
        _, ref_info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GaussNewtonCallback(
                    BacktrackingCallback(self.tm, self.b),
                    ConjugateGradientCallback(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gauss_newton_secant_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gauss_newton
        from phrt_opt.callbacks.counters import GaussNewtonCallback
        from phrt_opt.callbacks.counters import ConjugateGradientCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback
        from phrt_opt.callbacks.counters import SecantCallback

        x = self.x
        _, info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                gauss_newton(self.tm, self.b, {
                    "linesearch": {
                        "name": "secant",
                        "params": {
                            "linesearch": {
                                "name": "backtracking",
                                "params": {
                                    "max_iter": 5,
                                },
                            },
                            "sym": True,
                        }
                    },
                    "quadprog": {
                        "name": "conjugate_gradient",
                        "params": {
                            "dlt": 1e-8,
                        }
                    }
                })
            ],
            persist_iterations=True,
        )
        _, ref_info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GaussNewtonCallback(
                    SecantCallback(BacktrackingCallback(self.tm, self.b, {"max_iter": 5}), {"sym": True}),
                    ConjugateGradientCallback(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gauss_newton_secant_cholesky_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gauss_newton
        from phrt_opt.callbacks.counters import GaussNewtonCallback
        from phrt_opt.callbacks.counters import CholeskyCallback
        from phrt_opt.callbacks.counters import BacktrackingCallback
        from phrt_opt.callbacks.counters import SecantCallback

        x = self.x
        _, info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                gauss_newton(self.tm, self.b, {
                    "linesearch": {
                        "name": "secant",
                        "params": {
                            "linesearch": {
                                "name": "backtracking",
                                "params": {
                                    "max_iter": 5,
                                },
                            },
                            "sym": True,
                        }
                    },
                    "quadprog": {
                        "name": "cholesky",
                        "params": {
                            "dlt": 1e-8,
                        }
                    }
                })
            ],
            persist_iterations=True,
        )
        _, ref_info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GaussNewtonCallback(
                    SecantCallback(BacktrackingCallback(self.tm, self.b, {"max_iter": 5}), {"sym": True}),
                    CholeskyCallback(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_alternating_projections_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import alternating_projections
        from phrt_opt.callbacks.counters import AlternatingProjectionsCallback

        x = self.x
        x_bar, ref_info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                AlternatingProjectionsCallback(np.shape(self.tm)),
            ],
            persist_iterations=True,
        )
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                alternating_projections(self.tm, self.b, {}),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_admm_count_callback(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import admm
        from phrt_opt.callbacks.counters import ADMMCallback

        x = self.x
        x_bar, ref_info = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                ADMMCallback(np.shape(self.tm)),
            ],
            persist_iterations=True,
        )
        x_bar, info = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                admm(self.tm, self.b, {}),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))
