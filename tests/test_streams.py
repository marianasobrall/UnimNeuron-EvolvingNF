"""Checks for the complexity-level streams (point 4)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.streams import COMPLEXITY_LEVELS, bayes_accuracy, gaussian_clouds


class StreamTests(unittest.TestCase):
    def test_same_seed_same_stream(self):
        x1, y1 = gaussian_clouds(1000, 2.0, seed=7)
        x2, y2 = gaussian_clouds(1000, 2.0, seed=7)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_shape_and_labels(self):
        x, y = gaussian_clouds(500, 2.0, seed=0)
        self.assertEqual(x.shape, (500, 2))
        self.assertTrue(set(np.unique(y)) <= {0, 1})

    def test_balanced_classes(self):
        _, y = gaussian_clouds(20000, 2.0, seed=0)
        self.assertLess(abs(y.mean() - 0.5), 0.02)

    def test_distance_between_centres(self):
        for d in COMPLEXITY_LEVELS.values():
            x, y = gaussian_clouds(50000, d, seed=1)
            gap = np.linalg.norm(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
            self.assertLess(abs(gap - d), 0.05)

    def test_levels_get_harder(self):
        acc = [bayes_accuracy(d) for d in COMPLEXITY_LEVELS.values()]
        self.assertEqual(acc, sorted(acc, reverse=True))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            gaussian_clouds(0, 2.0, seed=0)
        with self.assertRaises(ValueError):
            gaussian_clouds(10, -1.0, seed=0)


if __name__ == "__main__":
    unittest.main()