import unittest
import sys
import numpy as np
sys.path.append('../')


class Testget_linearfit_features(unittest.TestCase):
    def test_simple_values(self):
        from clean import get_linearfit_features
        fit_params = get_linearfit_features([1, 2])
        self.assertEqual(np.allclose(fit_params, np.array([1., 1.])), True)

    def test_single_value(self):
        from clean import get_linearfit_features
        fit_params = get_linearfit_features([5, ])
        self.assertEqual(np.allclose(fit_params, np.array([5., 5.])), True)

    def test_single_nan(self):
        from clean import get_linearfit_features
        fit_params = get_linearfit_features([np.nan, ])
        np.testing.assert_equal(fit_params, np.array([np.nan, np.nan]))
