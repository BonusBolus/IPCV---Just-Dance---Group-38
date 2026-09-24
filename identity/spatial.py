"""Task 3: convert image positions into physical positions (metres) for Task 4.

Pinhole camera model with focal length f (pixels) from the horizontal field of view:
    depth   Z = f * M / m     (M = real size of a body measure in metres, m = its size in pixels)
    lateral X = (u - cx) * Z / f,   vertical Y = (v - cy) * Z / f
Body measures: shoulder width (~0.38 m) and torso length (~0.50 m). Turning sideways
foreshortens the shoulders and leaning foreshortens the torso, and foreshortening only makes
a measure *smaller*, i.e. Z too large. So we take the smallest Z of the available measures.
Z is smoothed per player (EMA), because it is noisier than the 2D position.

Accuracy depends on the camera FOV (config: tracking.camera_hfov_deg) and on body size, so
expect roughly 10-20% depth error. eval/eval_identity.py compares estimates to tape-measured
distances.
"""
from __future__ import annotations

import math

import numpy as np

from core.config import Config
from core.types import KP, Player


def focal_length_px(image_width_px: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from the horizontal field of view."""
    return (image_width_px / 2) / math.tan(math.radians(hfov_deg) / 2)


class SpatialEstimator:
    def __init__(self, cfg: Config):
        self.tc = cfg.tracking
        self.min_conf = cfg.pose.min_keypoint_conf
        self._depth: dict[int, float] = {}
        self._shape: tuple[int, int] | None = None

    def estimate(self, player: Player, frame_shape) -> tuple[float, float] | None:
        """(lateral x, depth z) in metres of the torso centre; None if the body is not visible."""
        self._shape = frame_shape[:2]
        pose = player.pose
        if pose is None:
            return None
        h, w = frame_shape[:2]
        f = focal_length_px(w, self.tc.camera_hfov_deg)
        px = pose.keypoints * np.array([w, h], np.float32)
        ok = pose.confidence >= self.min_conf
        ls, rs, lh, rh = KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]
        if not (ok[ls] and ok[rs]):
            return None

        depths = [f * self.tc.shoulder_width_m / max(np.linalg.norm(px[ls] - px[rs]), 1.0)]
        center = (px[ls] + px[rs]) / 2
        if ok[lh] and ok[rh]:
            hip_mid = (px[lh] + px[rh]) / 2
            depths.append(f * self.tc.torso_length_m / max(np.linalg.norm(center - hip_mid), 1.0))
            center = (center + hip_mid) / 2
        z = min(depths)
        prev = self._depth.get(player.pid)
        z = z if prev is None else 0.7 * prev + 0.3 * z
        self._depth[player.pid] = z
        return float((center[0] - w / 2) * z / f), float(z)

    def point_m(self, player: Player, xy_norm) -> np.ndarray | None:
        """3D position (X, Y, Z) in metres of a normalized image point on this player's body,
        assuming it is at the player's depth (fine for hands during a high five)."""
        z = self._depth.get(player.pid)
        if z is None or self._shape is None:
            return None
        h, w = self._shape
        f = focal_length_px(w, self.tc.camera_hfov_deg)
        u, v = xy_norm[0] * w, xy_norm[1] * h
        return np.array([(u - w / 2) * z / f, (v - h / 2) * z / f, z], np.float64)

    def reset(self) -> None:
        self._depth.clear()
