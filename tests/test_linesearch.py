import phrt_opt
import unittest
import numpy as np


class TestLinesearch(unittest.TestCase):

    def setUp(self):
        m, n = 20, 5
        self.seed = 2
        random = np.random.RandomState(self.seed)

        self.x = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        self.x0 = np.exp(1j * random.uniform(-np.pi, np.pi, size=(n, 1)))
        tm = random.randn(m, n) + 1j * random.randn(m, n)
        b = np.abs(tm.dot(self.x))

        self.fun = phrt_opt.utils.define_objective(tm, b)
        self.gradient = phrt_opt.utils.define_gradient(tm, b)

    def test_backtracking(self):
        linesearch = phrt_opt.linesearch.Backtracking()
        x0 = self.x0
        fun = self.fun
        g = self.gradient(x0)
        alpha = linesearch(fun, x0, g)
        self.assertTrue(fun(x0 - alpha * g) < fun(x0))

    def test_secant(self):
        linesearch = phrt_opt.linesearch.Secant()
        x0 = self.x0
        fun = self.fun
        g = self.gradient(x0)
        alpha = linesearch(fun, x0, g)
        self.assertTrue(fun(x0 - alpha * g) < fun(x0))

