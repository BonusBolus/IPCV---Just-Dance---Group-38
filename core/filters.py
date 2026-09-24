"""Temporal filters shared by Task 1 (faces), Task 2 (keypoints) and Task 3 (tracks).

All filters work on numpy arrays of any shape, so one instance can smooth a whole (17, 2)
skeleton or a face bbox at once.
"""
from __future__ import annotations

import numpy as np


class ExponentialSmoother:
    """Baseline filter: y = a*x + (1-a)*y_prev. Compare the One Euro filter against this in the report.

    A fixed `alpha` is a hard trade-off: low alpha = smooth but laggy, high alpha = responsive
    but jittery.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._y: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        self._y = x.copy() if self._y is None else self.alpha * x + (1 - self.alpha) * self._y
        return self._y

    def reset(self) -> None:
        self._y = None


class OneEuroFilter:
    """Speed-adaptive low-pass filter (Casiez et al., CHI 2012). Owner: Task 1 / Task 2.

    TODO(T1/T2): implement. The idea:
        dx      = (x - x_prev) / dt, smoothed with a fixed cutoff d_cutoff
        cutoff  = min_cutoff + beta * |dx|
        alpha   = 1 / (1 + tau / dt), with tau = 1 / (2*pi*cutoff)
        y       = alpha * x + (1 - alpha) * y_prev
    Slow movement gives a low cutoff (heavy smoothing, no jitter). Fast movement gives a high
    cutoff (little lag). Tune min_cutoff first (jitter at rest), then beta (lag during motion).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError("OneEuroFilter: Task 1/2")

    def reset(self) -> None:
        pass
