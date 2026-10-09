"""Unit tests for the dataset module."""

import numpy as np

from pygtm.dataset import trajectory


def test_monotonic():
    """Test monotonic checking on 1D arrays."""
    assert trajectory.monotonic(np.array([1, 2, 3, 4]))
    assert trajectory.monotonic(np.array([4, 3, 2, 1]))
    assert not trajectory.monotonic(np.array([1, 3, 2, 4]))
    assert not trajectory.monotonic(np.array([1]))
