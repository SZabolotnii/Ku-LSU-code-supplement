"""Focused checks for population/empirical comparator separation."""

import unittest

import numpy as np
from scipy import integrate, stats

from run_comparators import g_population, laplace, student_t


class PopulationProjectionTests(unittest.TestCase):
    def test_student_anchor(self):
        g, _ = g_population(student_t(10), [1.0, 3.0], nu=10)
        self.assertAlmostEqual(g, 65 / 66, places=12)

    def test_inadmissible_student_polynomial_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in L2"):
            g_population(student_t(3), [1.0, 3.0], nu=3)
        with self.assertRaises(ValueError):
            g_population(student_t(3), [1.5], nu=3)

    def test_laplace_linear_fraction(self):
        g, _ = g_population(laplace(), [1.0])
        self.assertAlmostEqual(g, 0.5, places=12)

    def test_fractional_moments_against_quadrature(self):
        a = np.array([0.4, 0.8, 1.2])
        fam = student_t(3)
        G = np.array([[integrate.quad(
            lambda x: 2 * x ** (u + v) * stats.t.pdf(x, 3),
            0, np.inf, epsabs=1e-9)[0] for v in a] for u in a])
        b = np.array([integrate.quad(
            lambda x: 2 * x ** u * fam["score"](x) * stats.t.pdf(x, 3),
            0, np.inf, epsabs=1e-9)[0] for u in a])
        k = np.linalg.solve(G, b)
        g, actual_k = g_population(fam, a, nu=3)
        np.testing.assert_allclose(actual_k, k, rtol=1e-7)
        self.assertAlmostEqual(g, b @ k / fam["I"], places=8)


if __name__ == "__main__":
    unittest.main()
