import unittest
import pandas as pd
from pymatch.functions import find_nearest_n


class TestFunctions(unittest.TestCase):

    def test_find_nearest_n(self):

        series = pd.Series([1, 3, 1, 7, 2, 10, 2, 5, 6]).sort_values()

        self.assertEqual(find_nearest_n(series, 3, 1), [3])
        self.assertEqual(set(find_nearest_n(series, 3, 3)), set([2, 2, 3]))
        self.assertEqual(set(find_nearest_n(series, 20, 2)), set([10, 7]))
        self.assertEqual(set(find_nearest_n(series, 11, 2, threshold=2)), set([10]))
        self.assertEqual(find_nearest_n(series, 12, 2, threshold=1), [])

        nn = find_nearest_n(series, 4, 1)
        self.assertTrue(len(nn) == 1 and (nn[0] == 3 or nn[0] == 5))

        self.assertEqual(
            set(find_nearest_n(series, 4, 2)), set([3, 5]))
