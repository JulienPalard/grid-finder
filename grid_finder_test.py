#!/usr/bin/env python3

import math
import unittest
import itertools
from grid_finder import angle_between


class GridFinderTests(unittest.TestCase):
    def test_angle_between(self):
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, 20, 0)), 0.0)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, -10, 0)), 0.0)
        self.assertEqual(angle_between((0, 1, 0, 0), (1, 0, 1, 1)), 0.0)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, 0, 10)),
                         1.5707963267948966)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, 10, 10)),
                         0.7853981633974483)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, 5, 10)),
                         1.1071487177940904)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, 10, 5)),
                         0.4636476090008061)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, 10, -10)),
                         0.7853981633974483)
        self.assertEqual(angle_between((0, 0, 10, 0), (0, 0, -10, 10)),
                         0.7853981633974483)
        for line_b in itertools.combinations_with_replacement((10, -10), 4):
            angle = angle_between((0, 0, 0, 10), line_b)
            self.assertGreaterEqual(angle, 0)
            self.assertLessEqual(angle, math.pi / 2)


if __name__ == '__main__':
    unittest.main()
