"""Task 2: turn keypoints into motion/control signals used by Task 4.

All functions take a PoseObs (or its arrays) and must ignore keypoints below `min_conf`.
Nothing calls these yet; Task 4's scorer and move detector will.
"""
from __future__ import annotations

import numpy as np

from core.types import PoseObs


def normalize_pose(pose: PoseObs, min_conf: float = 0.3) -> np.ndarray:
    """TODO(T2): (17, 2) keypoints translated to the hip centre and scaled by torso length.

    Makes poses of a tall/short or near/far player comparable to the reference.
    Unreliable keypoints -> NaN.
    """
    raise NotImplementedError


def limb_vectors(pose: PoseObs, min_conf: float = 0.3) -> np.ndarray:
    """TODO(T2): unit direction vectors of the limbs (upper arm, forearm, thigh, shin, ...).

    Direction vectors are scale-invariant, which makes them a good input for pose similarity.
    """
    raise NotImplementedError


def joint_angles(pose: PoseObs, min_conf: float = 0.3) -> dict[str, float]:
    """TODO(T2): angles in degrees, e.g. {"left_elbow": 170.0, "left_shoulder": 90.0, ...}."""
    raise NotImplementedError


def keypoint_velocity(prev: PoseObs, curr: PoseObs, min_conf: float = 0.3) -> np.ndarray:
    """TODO(T2): (17, 2) velocity in normalized units per second (useful for claps, jumps, energy)."""
    raise NotImplementedError
