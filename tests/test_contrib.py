import phrt_opt
import unittest
import numpy as np


class TestContrib(unittest.TestCase):

    def setUp(self):
        self.seed = 2
        self.random_state = np.random.RandomState(self.seed)
        self.phases = self.random_state.randn(100, 8) + 1j * self.random_state.randn(100, 8)
        tm = self.random_state.randn(32, 8) + 1j * self.random_state.randn(32, 8)
        self.intens = np.abs(self.phases @ tm.T)

    def test_retrieve_transmission_matrix(self):
        tm = phrt_opt.retrieve_transmission_matrix(
            self.phases, self.intens, initializer="gao_xu", max_iter=500, tol=1e-14, power_method_tol=1e-8)
        r = np.linalg.norm(np.abs(self.phases @ tm) - self.intens) / np.linalg.norm(self.intens)
        self.assertAlmostEquals(r, 1e-7)

    def test_compute_camera_bias(self):
        bias = phrt_opt.compute_camera_bias(
            self.phases, self.intens, initializer="gao_xu", max_iter=500, tol=1e-14, power_method_tol=1e-8)
        self.assertAlmostEquals(bias, 1e-8)
