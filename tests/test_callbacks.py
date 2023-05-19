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

    def test_ops_backtracking_counter(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import GradientDescentCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter

        x = self.x
        x_bar, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GradientDescentCounter(
                    BacktrackingCounter(self.tm, self.b),
                ),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 7.197200514452717e-06)

    def test_ops_secant_counter(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import GradientDescentCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter
        from phrt_opt.callbacks.counters import SecantCounter

        x = self.x
        x_bar, info = phrt_opt.methods.gradient_descent(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GradientDescentCounter(
                    SecantCounter(
                        BacktrackingCounter(self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 7.197200514452717e-06)

    def test_gauss_newton_counter(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import ConjugateGradientCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter
        from phrt_opt.callbacks.counters import GaussNewtonCounter

        x = self.x
        x_bar, info = phrt_opt.methods.gauss_newton(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                GaussNewtonCounter(
                    BacktrackingCounter(self.tm, self.b),
                    ConjugateGradientCounter(
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

    def test_alternating_projections_counter(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import AlternatingProjectionsCounter

        x = self.x
        x_bar, info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                AlternatingProjectionsCounter(np.shape(self.tm)),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 1.66825933511916e-07)

    def test_admm_counter(self):
        import phrt_opt.utils
        from phrt_opt.callbacks.counters import ADMMCounter

        x = self.x
        x_bar, info = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                ADMMCounter(np.shape(self.tm)),
            ],
            persist_iterations=True,
        )
        dist = phrt_opt.metrics.quality_norm(x, x_bar)
        self.assertAlmostEqual(dist, 2.298161660974074e-14)

    def test_gradient_descent_backtracking_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gradient_descent
        from phrt_opt.callbacks.counters import GradientDescentCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter

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
                GradientDescentCounter(
                    BacktrackingCounter(self.tm, self.b),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gradient_descent_secant_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gradient_descent
        from phrt_opt.callbacks.counters import GradientDescentCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter
        from phrt_opt.callbacks.counters import SecantCounter

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
                GradientDescentCounter(
                    SecantCounter(
                        BacktrackingCounter(self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gauss_newton_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gauss_newton
        from phrt_opt.callbacks.counters import GaussNewtonCounter
        from phrt_opt.callbacks.counters import ConjugateGradientCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter

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
                GaussNewtonCounter(
                    BacktrackingCounter(self.tm, self.b),
                    ConjugateGradientCounter(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gauss_newton_secant_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gauss_newton
        from phrt_opt.callbacks.counters import GaussNewtonCounter
        from phrt_opt.callbacks.counters import ConjugateGradientCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter
        from phrt_opt.callbacks.counters import SecantCounter

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
                GaussNewtonCounter(
                    SecantCounter(BacktrackingCounter(self.tm, self.b, {"max_iter": 5}), {"sym": True}),
                    ConjugateGradientCounter(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_gauss_newton_secant_cholesky_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import gauss_newton
        from phrt_opt.callbacks.counters import GaussNewtonCounter
        from phrt_opt.callbacks.counters import CholeskyCounter
        from phrt_opt.callbacks.counters import BacktrackingCounter
        from phrt_opt.callbacks.counters import SecantCounter

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
                GaussNewtonCounter(
                    SecantCounter(BacktrackingCounter(self.tm, self.b, {"max_iter": 5}), {"sym": True}),
                    CholeskyCounter(
                        self.tm, self.b,
                        preliminary_step=phrt_opt.utils.define_gauss_newton_system(
                            self.tm, self.b),
                    ),
                ),
            ],
            persist_iterations=True,
        )
        self.assertTrue(np.allclose(info, ref_info))

    def test_alternating_projections_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import alternating_projections
        from phrt_opt.callbacks.counters import AlternatingProjectionsCounter

        x = self.x
        x_bar, ref_info = phrt_opt.methods.alternating_projections(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                AlternatingProjectionsCounter(np.shape(self.tm)),
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

    def test_admm_counter(self):
        import phrt_opt.utils
        from phrt_opt.parsers.callbacks import admm
        from phrt_opt.callbacks.counters import ADMMCounter

        x = self.x
        x_bar, ref_info = phrt_opt.methods.admm(
            self.tm, self.b,
            x0=self.x0,
            tol=1e-6,
            max_iter=100,
            callbacks=[
                ADMMCounter(np.shape(self.tm)),
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
