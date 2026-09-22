import unittest
import numpy as np
import run_realdata as rd


class ProtocolTests(unittest.TestCase):
    def series(self, offset=0):
        dates = np.arange('2000-01-01', '2000-04-10', dtype='datetime64[D]')
        return rd.prepare_series('test', dates, np.arange(100.) + offset)

    def test_test_values_do_not_change_training_center_or_threshold(self):
        a = self.series()
        changed_returns = np.arange(100.)
        changed_returns[60:] += 10000
        b = rd.prepare_series('changed', a['dates'], changed_returns)
        original = rd.scale_samples([a])
        changed = rd.scale_samples([b])
        np.testing.assert_array_equal(original['thresholds'], changed['thresholds'])
        np.testing.assert_array_equal(original['xtr'], changed['xtr'])
        self.assertEqual(a['center'], np.median(np.arange(60.)))
        self.assertEqual(a['center'], b['center'])
        np.testing.assert_array_equal(a['rv'][a['train']], b['rv'][b['train']])

    def test_filtering_never_resplits_or_crosses_series(self):
        a, b = self.series(), self.series(1000)
        samples = rd.scale_samples([a, b])
        self.assertTrue(np.all(samples['dates_tr'] < a['dates'][60]))
        self.assertTrue(np.all(samples['dates_te'] >= a['dates'][60]))
        np.testing.assert_array_equal(a['r'], b['r'])
        self.assertTrue(np.isnan(a['rv'][:rd.VOL_WINDOW]).all())
        self.assertAlmostEqual(a['rv'][rd.VOL_WINDOW], np.std(a['r'][:rd.VOL_WINDOW]))

    def test_calendar_resampling_keeps_all_same_date_rows_together(self):
        dates = np.repeat(np.arange('2000-01-01', '2000-03-01', dtype='datetime64[D]'), 4)
        idx = rd.calendar_boot_indices(dates, np.random.default_rng(3))
        counts = np.bincount(idx, minlength=len(dates)).reshape(-1, 4)
        self.assertTrue(np.all(counts == counts[:, :1]))


if __name__ == '__main__':
    unittest.main()
