"""Synthetic dancers, plus the 2D body model used to build the default choreography.

Press M in the game to toggle mock players. They produce PoseObs/FaceObs in exactly the format
Tasks 1/2 deliver, including jitter noise, so Tasks 3/4/5 can be developed and tested without
a second person. The "cross" scenario makes the two dancers swap places: the identity-switch
test case for Task 3.
"""
from __future__ import annotations

import numpy as np

from core.types import KP, NUM_KEYPOINTS, FaceObs, PoseObs

# Neutral standing pose in "body units": origin at hip centre, y down, 1.0 = torso length.
# Same convention as the pose model: for a person facing the camera, left_* is on the image right.
_BASE = np.array([
    [0.00, -1.45],                   # nose
    [0.07, -1.52], [-0.07, -1.52],   # eyes (left, right)
    [0.15, -1.48], [-0.15, -1.48],   # ears
    [0.45, -1.00], [-0.45, -1.00],   # shoulders
    [0.55, -0.50], [-0.55, -0.50],   # elbows (overwritten by the arm angles)
    [0.60, -0.05], [-0.60, -0.05],   # wrists (overwritten by the arm angles)
    [0.25, 0.00], [-0.25, 0.00],     # hips
    [0.27, 0.80], [-0.27, 0.80],     # knees
    [0.28, 1.60], [-0.28, 1.60],     # ankles
], dtype=np.float32)
_UPPER_ARM, _FOREARM = 0.50, 0.47
_UPPER_BODY = slice(0, KP["left_knee"])   # nose .. hips


def body_pose(left=(20.0, 20.0), right=(20.0, 20.0), squat: float = 0.0, bob: float = 0.0) -> np.ndarray:
    """(17, 2) keypoints in body units.

    `left`/`right` = (upper-arm angle, forearm angle) in degrees, measured from hanging straight
    down, positive = outward: 0 = hanging, 90 = horizontal (T-pose), 180 = straight up; a forearm
    angle > 180 bends the forearm inwards. `squat` in [0, 1] lowers the hips, `bob` shifts the
    upper body down a little (groove on the beat).
    """
    kp = _BASE.copy()
    kp[_UPPER_BODY, 1] += 0.45 * squat + bob
    for side, sign in (("left", 1.0), ("right", -1.0)):
        kp[KP[f"{side}_knee"], 0] += sign * 0.12 * squat
        kp[KP[f"{side}_knee"], 1] += 0.30 * squat + bob / 2
    for side, (upper, fore), sign in (("left", left, 1.0), ("right", right, -1.0)):
        u, f = np.radians(upper), np.radians(fore)
        shoulder = kp[KP[f"{side}_shoulder"]]
        elbow = shoulder + _UPPER_ARM * np.array([sign * np.sin(u), np.cos(u)], np.float32)
        kp[KP[f"{side}_elbow"]] = elbow
        kp[KP[f"{side}_wrist"]] = elbow + _FOREARM * np.array([sign * np.sin(f), np.cos(f)], np.float32)
    return kp


def to_image(body_kp: np.ndarray, center: tuple[float, float], scale: float, aspect: float) -> np.ndarray:
    """Body units -> normalized image coords. scale = torso length / image height, aspect = W/H."""
    out = np.empty_like(body_kp)
    out[:, 0] = center[0] + body_kp[:, 0] * scale / aspect
    out[:, 1] = center[1] + body_kp[:, 1] * scale
    return out


def dance_pose(t: float, phase: float = 0.0) -> np.ndarray:
    """A simple looping free-style dance in body units."""
    a = 90 + 80 * np.sin(2 * np.pi * 0.5 * t + phase)
    b = 90 + 80 * np.sin(2 * np.pi * 0.5 * t + phase + np.pi / 2)
    bob = 0.08 * max(0.0, np.sin(2 * np.pi * 1.0 * t + phase))
    return body_pose((a, a), (b, b), bob=bob)


class MockPoseSource:
    """Synthetic players. Detection order is shuffled (like a real detector), but
    `true_ids` of the last `generate` call gives the ground-truth identity of each detection."""

    SCENARIOS = ("dance", "cross")

    def __init__(self, num_players: int = 2, scenario: str = "dance", noise: float = 0.003,
                 seed: int = 0, scale: float = 0.2, dropout: float = 0.0, shuffle: bool = True):
        assert scenario in self.SCENARIOS
        self.num_players = num_players
        self.scenario = scenario
        self.noise = noise
        self.scale = scale
        self.dropout = dropout        # probability that a player is not detected in a frame
        self.shuffle = shuffle
        self.true_ids: list[int] = []
        self._rng = np.random.default_rng(seed)

    def generate(self, t: float, image_shape) -> tuple[list[PoseObs], list[FaceObs]]:
        h, w = image_shape[:2]
        aspect = w / h
        items = []
        for i in range(self.num_players):
            if self.dropout and self._rng.random() < self.dropout:
                continue
            kp = to_image(dance_pose(t, phase=i * 2.2), (self._center_x(i, t), 0.62), self.scale, aspect)
            kp += self._rng.normal(0.0, self.noise, kp.shape)
            conf = np.full(NUM_KEYPOINTS, 0.9, np.float32)
            items.append((i + 1, PoseObs(kp.astype(np.float32), conf, timestamp=t, aspect=aspect),
                          self._face(kp, aspect, t)))
        if self.shuffle:
            self._rng.shuffle(items)
        self.true_ids = [pid for pid, _, _ in items]
        return [p for _, p, _ in items], [f for _, _, f in items]

    def _center_x(self, i: int, t: float) -> float:
        n = self.num_players
        base = (i + 1) / (n + 1)
        if self.scenario == "cross" and n >= 2 and i < 2:
            other = (2 - i) / (n + 1)          # the other of the first two players
            s = (1 - np.cos(2 * np.pi * t / 10.0)) / 2   # 0 -> 1 -> 0 every 10 s
            return float(base + (other - base) * s)
        return base

    def _face(self, kp: np.ndarray, aspect: float, t: float) -> FaceObs:
        nx, ny = kp[KP["nose"]]
        fh = 0.45 * self.scale
        fw = 0.8 * fh / aspect
        return FaceObs(bbox=(float(nx - fw / 2), float(ny - fh * 0.55), fw, fh), confidence=0.95, timestamp=t)
