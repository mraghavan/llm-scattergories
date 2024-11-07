"""
Copyright 2020 Google LLC.
SPDX-License-Identifier: Apache-2.0
"""
# Code from https://stats.stackexchange.com/questions/459601/are-these-any-existing-implementation-of-l1-isotonic-regression-in-python

import heapq
import numpy as np

def isotonic_regression_l1_total_order(y, w):
    """Finds a non-decreasing fit for the specified `y` under L1 norm.

    The O(n log n) algorithm is described in:
    "Isotonic Regression by Dynamic Programming", Gunter Rote, SOSA@SODA 2019.

    Args:
        y: The values to be fitted, 1d-numpy array.
        w: The loss weights vector, 1d-numpy array.

    Returns:
        An isotonic fit for the specified `y` which minimizies the weighted
        L1 norm of the fit's residual.
    """
    h = []    # max heap of values
    p = np.zeros_like(y)    # breaking position
    for i in range(y.size):
        a_i = y[i]
        w_i = w[i]
        heapq.heappush(h, (-a_i, 2 * w_i))
        s = -w_i
        b_position, b_value = h[0]
        while s + b_value <= 0:
            s += b_value
            heapq.heappop(h)
            b_position, b_value = h[0]
        b_value += s
        h[0] = (b_position, b_value)
        p[i] = -b_position
    z = np.flip(np.minimum.accumulate(np.flip(p)))    # right_to_left_cumulative_min
    return z
