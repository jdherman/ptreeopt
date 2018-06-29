from __future__ import division

from nose.tools import raises
from numpy.testing import assert_allclose, assert_equal

import numpy as np

from ..tree import *


def test_compute_mu_star_confidence():
    '''
    Tests that compute mu_star_confidence is computed correctly
    '''

    ee = np.array([2.52, 2.01, 2.30, 0.66, 0.93, 1.3], dtype=np.float)
    num_trajectories = 6
    num_resamples = 1000
    conf_level = 0.95

    actual = compute_mu_star_confidence(
        ee, num_trajectories, num_resamples, conf_level)
    expected = 0.5
    assert_allclose(actual, expected, atol=1e-01)
