import unittest
from unittest import TestCase
import numpy as np
from pygtm.tools import ismember, filter_vector

if __name__ == '__main__':
    unittest.main()


class tools_tests(TestCase):
    def test_ismember(self):
        a = [1, 2]
        b = [3, 4]
        self.assertSequenceEqual(ismember(a, b).tolist(), [-1, -1])

        a = [1, 2, 3]
        b = [1, 2, 4, 5]
        self.assertSequenceEqual(ismember(a, b).tolist(), [0, 1, -1])
        self.assertSequenceEqual(ismember(b, a).tolist(), [0, 1, -1, -1])

    def test_filter_vector(self):
        # the second arguments is either a list of indices or a
        # boolean the size of the first arguments
        a = np.array([1, 2, 3, 4, 5])
        b = [0, 1, 2]
        self.assertSequenceEqual(filter_vector(a, b).tolist(), [1, 2, 3])

        b = [True, True, True, False, False]
        self.assertSequenceEqual(filter_vector(a, b).tolist(), [1, 2, 3])

        # also work if a is a list
        a = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
        b = [0, 1, 2]
        ret = filter_vector(a, b)
        self.assertSequenceEqual(ret[0].tolist(), [1, 2, 3])
        self.assertSequenceEqual(ret[1].tolist(), [1, 2, 3])

        # this will create an IndexError exception because
        # it's over the range of the variable a
        a = np.array([1, 2, 3, 4, 5])
        b = [5]
        self.assertRaises(IndexError, filter_vector, a, b)
        a = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
        self.assertRaises(IndexError, filter_vector, a, b)
