"""Synthetic dancers for development without a second person or finished CV modules.

Press M in the game to toggle. It produces PoseObs/FaceObs in exactly the format Tasks 1/2 must
deliver, including jitter noise, so Tasks 3/4/5 can develop and test independently.
The "cross" scenario makes the two dancers swap places: the identity-switch test case for Task 3.
"""
from __future__ import annotations

import numpy as np

from core.types import KP, FaceObs, PoseObs

# Neutral standing pose in "body units": origin at hip centre, y down, 1.0 = torso length.
# Coordinates are for the mirrored (selfie) view, so the person's left side is at negative x.
_BASE = np.array([
    [0.00, -1.45],                   # nose
    [-0.07, -1.52], [0.07, -1.52],   # eyes
    [-0.15, -1.48], [0.15, -1.48],   # ears
    [-0.45, -1.00], [0.45, -1.00],   # shoulders
    [-0.55, -0.50], [0.55, -0.50],   # elbows (overwritten by the arm angles)
    [-0.60, -0.05], [0.60, -0.05],   # wrists (overwritten by the arm angles)
    [-0.25, 0.00], [0.25, 0.00],     # hips
    [-0.27, 0.80], [0.27, 0.80],     # knees
    [-0.28, 1.60], [0.28, 1.60],     # ankles
], dtype=np.float32)
_UPPER_ARM, _FOREARM = 0.50, 0.47


def body_pose(arm_left_deg: float, arm_right_deg: float, bob: float = 0.0) -> np.ndarray:
    """(17, 2) keypoints in body units. Arm angles: 0 = hanging, 90 = T-pose, 180 = straight up."""
    kp = _BASE.copy()
    kp[:KP["left_knee"], 1] += bob              # upper body goes down (squat bob)
    kp[KP["left_knee"]:KP["left_ankle"], 1] += bob / 2
    for side, angle, sign in (("left", arm_left_deg, -1.0), ("right", arm_right_deg, 1.0)):
        th = np.radians(angle)
        direction = np.array([sign * np.sin(th), np.cos(th)], np.float32)
        shoulder = kp[KP[f"{side}_shoulder"]]
        kp[KP[f"{side}_elbow"]] = shoulder + direction * _UPPER_ARM
        kp[KP[f"{side}_wrist"]] = shoulder + direction * (_UPPER_ARM + _FOREARM)
    return kp


def to_image(body_kp: np.ndarray, center: tuple[float, float], scale: float, aspect: float) -> np.ndarray:
    """Body units -> normalized image coords. scale = torso length / image height, aspect = W/H."""
    out = np.empty_like(body_kp)
    out[:, 0] = center[0] + body_kp[:, 0] * scale / aspect
    out[:, 1] = center[1] + body_kp[:, 1] * scale
    return out


def dance_angles(t: float, phase: float = 0.0) -> tuple[float, float, float]:
    """A simple looping dance: (left arm deg, right arm deg, bob) at time t."""
    left = 90 + 80 * np.sin(2 * np.pi * 0.5 * t + phase)
    right = 90 + 80 * np.sin(2 * np.pi * 0.5 * t + phase + np.pi / 2)
    bob = 0.08 * max(0.0, np.sin(2 * np.pi * 1.0 * t + phase))
    return float(left), float(right), float(bob)


class MockPoseSource:
    SCENARIOS = ("dance", "cross")

    def __init__(self, num_players: int = 2, scenario: str = "dance",
                 noise: float = 0.003, seed: int = 0, scale: float = 0.2):
        assert scenario in self.SCENARIOS
        self.num_players = num_players
        self.scenario = scenario
        self.noise = noise
        self.scale = scale
        self._rng = np.random.default_rng(seed)

    def generate(self, t: float, image_shape) -> tuple[list[PoseObs], list[FaceObs]]:
        h, w = image_shape[:2]
        aspect = w / h
        poses, faces = [], []
        for i in range(self.num_players):
            left, right, bob = dance_angles(t, phase=i * 2.2)
            kp = to_image(body_pose(left, right, bob), (self._center_x(i, t), 0.62), self.scale, aspect)
            kp += self._rng.normal(0.0, self.noise, kp.shape)
            conf = np.full(len(kp), 0.9, np.float32)
            poses.append(PoseObs(kp.astype(np.float32), conf, timestamp=t))
            faces.append(self._face(kp, aspect, t))
        return poses, faces

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
