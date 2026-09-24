"""Task 2: turn keypoints into motion/control signals used by Task 4.

All functions ignore keypoints below `min_conf` and work in *isotropic* units (x * aspect),
because normalized image coordinates stretch x relative to y.
The array versions accept leading batch dimensions, so a whole reference window can be
processed in one call.
"""
from __future__ import annotations

import numpy as np

from core.types import KP, PoseObs

# Limbs used for pose similarity: (from, to, weight). The arms carry most of a Just Dance
# move; the legs are often cut off by the frame and barely move in most choreographies, so they
# weigh less (otherwise standing still would already score half the points).
LIMBS = (
    (KP["left_shoulder"], KP["left_elbow"], 1.0),
    (KP["left_elbow"], KP["left_wrist"], 1.0),
    (KP["right_shoulder"], KP["right_elbow"], 1.0),
    (KP["right_elbow"], KP["right_wrist"], 1.0),
    (KP["left_hip"], KP["left_knee"], 0.3),
    (KP["left_knee"], KP["left_ankle"], 0.3),
    (KP["right_hip"], KP["right_knee"], 0.3),
    (KP["right_knee"], KP["right_ankle"], 0.3),
)
LIMB_FROM = np.array([a for a, _, _ in LIMBS])
LIMB_TO = np.array([b for _, b, _ in LIMBS])
LIMB_WEIGHT = np.array([w for _, _, w in LIMBS], np.float32)

# (joint, neighbour a, neighbour b): the angle at `joint` between the segments to a and b.
JOINTS = {
    "left_elbow": (KP["left_elbow"], KP["left_shoulder"], KP["left_wrist"]),
    "right_elbow": (KP["right_elbow"], KP["right_shoulder"], KP["right_wrist"]),
    "left_shoulder": (KP["left_shoulder"], KP["left_hip"], KP["left_elbow"]),
    "right_shoulder": (KP["right_shoulder"], KP["right_hip"], KP["right_elbow"]),
    "left_knee": (KP["left_knee"], KP["left_hip"], KP["left_ankle"]),
    "right_knee": (KP["right_knee"], KP["right_hip"], KP["right_ankle"]),
}


def isotropic(keypoints: np.ndarray, aspect: float) -> np.ndarray:
    return keypoints * np.array([aspect, 1.0], np.float32)


def limb_vectors_arr(keypoints: np.ndarray, confidence: np.ndarray, aspect: float,
                     min_conf: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Unit limb direction vectors (..., L, 2) and weights (..., L); weight 0 = unreliable limb.

    Direction vectors do not depend on body size, distance or position in the frame, which
    makes them a good basis for comparing a player to the reference dancer.
    """
    iso = isotropic(keypoints, aspect)
    vec = iso[..., LIMB_TO, :] - iso[..., LIMB_FROM, :]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    unit = vec / np.maximum(norm, 1e-6)
    c = np.minimum(confidence[..., LIMB_FROM], confidence[..., LIMB_TO])
    ok = (c >= min_conf) & (norm[..., 0] > 1e-4)
    return unit.astype(np.float32), np.where(ok, c * LIMB_WEIGHT, 0.0).astype(np.float32)


def limb_vectors(pose: PoseObs, min_conf: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    return limb_vectors_arr(pose.keypoints, pose.confidence, pose.aspect, min_conf)


def torso_length(pose: PoseObs, min_conf: float = 0.5) -> float | None:
    """Shoulder-midpoint to hip-midpoint distance (isotropic units), or estimated from the
    shoulder width when the hips are not visible."""
    iso, ok = pose.iso(), pose.valid(min_conf)
    ls, rs, lh, rh = KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]
    if ok[[ls, rs, lh, rh]].all():
        return float(np.linalg.norm((iso[ls] + iso[rs]) / 2 - (iso[lh] + iso[rh]) / 2))
    if ok[[ls, rs]].all():
        return float(np.linalg.norm(iso[ls] - iso[rs]) / 0.8)  # shoulder width ~ 0.8 torso
    return None


def normalize_pose(pose: PoseObs, min_conf: float = 0.5) -> np.ndarray | None:
    """(17, 2) keypoints with the hip centre (or shoulder centre) as origin and torso length 1.

    Makes poses of a tall/short or near/far player comparable. Unreliable keypoints -> NaN.
    Returns None if the body is not visible enough to define the scale.
    """
    scale = torso_length(pose, min_conf)
    if scale is None or scale < 1e-4:
        return None
    iso, ok = pose.iso(), pose.valid(min_conf)
    hips = [KP["left_hip"], KP["right_hip"]]
    shoulders = [KP["left_shoulder"], KP["right_shoulder"]]
    if ok[hips].all():
        origin = iso[hips].mean(axis=0)
    else:  # hips out of view: shoulder centre shifted down by one torso length
        origin = iso[shoulders].mean(axis=0) + np.array([0.0, scale], np.float32)
    out = (iso - origin) / scale
    out[~ok] = np.nan
    return out.astype(np.float32)


def joint_angles(pose: PoseObs, min_conf: float = 0.5) -> dict[str, float]:
    """Joint angles in degrees (180 = straight). Joints with an unreliable keypoint are left out."""
    iso, ok = pose.iso(), pose.valid(min_conf)
    out = {}
    for name, (j, a, b) in JOINTS.items():
        if not (ok[j] and ok[a] and ok[b]):
            continue
        u, v = iso[a] - iso[j], iso[b] - iso[j]
        cos = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
        out[name] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return out


def keypoint_velocity(prev: PoseObs, curr: PoseObs, min_conf: float = 0.5) -> np.ndarray:
    """(17, 2) velocity in isotropic units per second; NaN where either keypoint is unreliable."""
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        return np.full_like(curr.keypoints, np.nan)
    v = (curr.iso() - prev.iso()) / dt
    v[~(prev.valid(min_conf) & curr.valid(min_conf))] = np.nan
    return v
