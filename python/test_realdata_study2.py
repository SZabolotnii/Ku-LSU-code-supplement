import unittest
import numpy as np
from scipy import stats
import run_realdata_study2 as s2


class Study2ProtocolTests(unittest.TestCase):
    def test_pair_differences_have_known_location_and_shape(self):
        rng = np.random.default_rng(0)
        e = rng.standard_t(3, size=(20, 512)) + 1234.5      # an arbitrary unknown offset
        d = s2.pair_differences(e)
        self.assertEqual(d.shape, (20, 256))
        self.assertLess(abs(np.median(d)), 0.2)             # the offset is gone by construction

    def test_block_scale_is_shift_invariant_and_ignores_test_values(self):
        rng = np.random.default_rng(1)
        D = rng.standard_t(2, size=(30, 256))
        Z = s2.standardise_blocks(D)
        Zs = s2.standardise_blocks(D + 5.0)
        np.testing.assert_allclose(Z + 5.0 / stats.median_abs_deviation(D, axis=1, scale="normal", keepdims=True), Zs)
        train = s2.split_epochs(30)
        self.assertEqual(train.sum(), 18)
        D2 = D.copy(); D2[~train] *= 1000.0
        np.testing.assert_array_equal(s2.standardise_blocks(D2)[train], Z[train])

    def test_kappa_of_sign_dictionary_under_cauchy_is_8_over_pi_squared(self):
        self.assertAlmostEqual(s2.kappa_under_t([0.0], 1.0, 1.0), 8 / np.pi**2, places=4)

    def test_kappa_is_a_fraction_and_grows_with_the_dictionary(self):
        k1 = s2.kappa_under_t([0.5], 3.0, 1.0)
        k2 = s2.kappa_under_t([0.3, 0.5, 0.8], 3.0, 1.0)
        self.assertTrue(0 < k1 <= k2 <= 1.0 + 1e-9)

    def test_t_mle_recovers_an_injected_shift(self):
        rng = np.random.default_rng(2)
        b = stats.t.rvs(2.5, size=4000, random_state=rng) + 0.7
        self.assertAlmostEqual(s2.t_mle_location(b, 2.5, 1.0), 0.7, delta=0.06)


if __name__ == "__main__":
    unittest.main()
