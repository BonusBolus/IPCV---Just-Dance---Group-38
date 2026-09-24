"""Temporal filters shared by Task 1 (faces), Task 2 (keypoints) and Task 3 (tracks).

The smoothing filters work on numpy arrays of any shape, so one instance can smooth a whole
(17, 2) skeleton or a face box at once.
"""
from __future__ import annotations

import math

import numpy as np


class ExponentialSmoother:
    """Baseline filter: y = a*x + (1-a)*y_prev. Compare the One Euro filter against this in the report.

    A fixed `alpha` is a hard trade-off: low alpha = smooth but laggy, high alpha = responsive
    but jittery.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._y: np.ndarray | None = None

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        self._y = x.copy() if self._y is None else self.alpha * x + (1 - self.alpha) * self._y
        return self._y

    def reset(self) -> None:
        self._y = None


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Smoothing factor of a first-order low-pass filter with the given cutoff frequency (Hz)."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Speed-adaptive low-pass filter (Casiez, Roussel & Vogel, CHI 2012).

        dx      = (x - y_prev) / dt, itself low-passed with cutoff d_cutoff
        cutoff  = min_cutoff + beta * |dx|
        y       = alpha(cutoff) * x + (1 - alpha) * y_prev

    Slow movement gives a low cutoff (heavy smoothing, no jitter). Fast movement gives a high
    cutoff (little lag). Tune `min_cutoff` first (jitter at rest), then `beta` (lag in motion).
    The cutoff is computed per element, so a fast wrist does not un-smooth a still head.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._y: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._y is None or self._t is None or x.shape != self._y.shape:
            self._y, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return self._y
        dt = t - self._t
        if dt <= 0:
            return self._y
        self._t = t
        dx = (x - self._y) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _alpha(cutoff, dt)
        self._y = self._y + a * (x - self._y)
        return self._y

    @property
    def velocity(self) -> np.ndarray | None:
        """Smoothed derivative (units per second) of the last update."""
        return self._dx

    def reset(self) -> None:
        self._y = self._dx = self._t = None


class ConstantVelocityKalman:
    """Kalman filter with state [x, y, vx, vy] and position measurements (Task 3 tracks).

    `predict(dt)` moves the state forward; `update(z)` corrects it with a measured position.
    The velocity is damped while no measurements arrive, so a lost track slows down instead of
    flying off screen.
    """

    def __init__(self, xy, process_noise: float = 2.0, measurement_noise: float = 0.01):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], np.float64)
        self.P = np.diag([0.01, 0.01, 1.0, 1.0])
        self.q = process_noise
        self.R = np.eye(2) * measurement_noise ** 2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float64)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    def predict(self, dt: float, damping: float = 1.0) -> np.ndarray:
        if dt <= 0:
            return self.position
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, damping, 0], [0, 0, 0, damping]], np.float64)
        # white-noise acceleration model
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.position

    def update(self, z) -> np.ndarray:
        z = np.asarray(z, np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.position

    def position_std(self) -> float:
        """Uncertainty of the position estimate (grows while the track is lost)."""
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1])))
