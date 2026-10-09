import numpy as np
import pytest

from pygtm.tools import filter_vector, ismember


def test_ismember():
    a = [1, 2]
    b = [3, 4]
    assert ismember(a, b).tolist() == [-1, -1]

    a = [1, 2, 3]
    b = [1, 2, 4, 5]
    assert ismember(a, b).tolist() == [0, 1, -1]
    assert ismember(b, a).tolist() == [0, 1, -1, -1]


def test_filter_vector():
    # the second arguments is either a list of indices or a
    # boolean the size of the first arguments
    a = np.array([1, 2, 3, 4, 5])
    b = [0, 1, 2]
    assert filter_vector(a, b).tolist() == [1, 2, 3]

    b = [True, True, True, False, False]
    assert filter_vector(a, b).tolist() == [1, 2, 3]

    # also work if a is a list
    a = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    b = [0, 1, 2]
    ret = filter_vector(a, b)
    assert ret[0].tolist() == [1, 2, 3]
    assert ret[1].tolist() == [1, 2, 3]

    # this will create an IndexError exception because
    # it's over the range of the variable a
    a = np.array([1, 2, 3, 4, 5])
    b = [5]
    with pytest.raises(IndexError):
        filter_vector(a, b)
    a = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    with pytest.raises(IndexError):
        filter_vector(a, b)
