"""Synthetic streams for the experiments.

Complexity levels (point 4): two Gaussian classes in 2-D. Only the distance
between the class centres changes. Drift scenarios (point 5) will go here too.
"""

from math import erf, sqrt

import numpy as np

# distance between the two class centres
COMPLEXITY_LEVELS = {"simple": 6.0, "medium": 2.0, "overlap": 0.5}


def bayes_accuracy(separation):
    """Best accuracy any classifier can reach for this distance."""
    return 0.5 * (1 + erf(separation / 2 / sqrt(2)))


def gaussian_clouds(size, separation, seed):
    """Stationary stream with two balanced classes, in random order."""
    if size <= 0:
        raise ValueError("size must be positive")
    if separation < 0:
        raise ValueError("separation must be non-negative")

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=size)
    sign = np.where(y == 1, 1.0, -1.0)[:, None]
    centres = sign * (separation / 2) * np.array([1, 1]) / sqrt(2)
    x = centres + rng.standard_normal((size, 2))
    return x, y