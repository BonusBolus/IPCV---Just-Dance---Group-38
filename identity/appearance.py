"""Task 3: appearance model for re-identification (who is who after crossing or re-entry).

Descriptor: 2D hue-saturation histogram of the torso (the quad between shoulders and hips,
shrunk towards its centre so background and arms are excluded). Clothing colour is cheap and
fairly distinctive between two players, and HS ignores most brightness changes. Very dark or
unsaturated pixels have an unreliable hue and are masked out.

Limitation: two players in similar clothes are hard to tell apart by colour. Then the tracker
falls back on motion prediction only.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.types import KP, FrameData, PoseObs

H_BINS, S_BINS = 16, 8
WORK_WIDTH = 320  # histograms are computed on a downscaled frame: plenty for colour statistics
_TORSO = [KP["left_shoulder"], KP["right_shoulder"], KP["right_hip"], KP["left_hip"]]


class AppearanceModel:
    def __init__(self):
        self._hsv: np.ndarray | None = None
        self._frame_index = -1

    def _hsv_for(self, frame: FrameData) -> np.ndarray:
        if frame.index != self._frame_index or self._hsv is None:  # once per frame, shared by all players
            h, w = frame.image.shape[:2]
            small = cv2.resize(frame.image, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)
            self._hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            self._frame_index = frame.index
        return self._hsv

    def torso_polygon(self, pose: PoseObs, min_conf: float = 0.5) -> np.ndarray | None:
        """Normalized torso quad (4, 2), shrunk by 20% towards its centre; None if not visible."""
        ok = pose.confidence >= min_conf
        ls, rs, lh, rh = (pose.keypoints[i] for i in _TORSO)
        if not (ok[KP["left_shoulder"]] and ok[KP["right_shoulder"]]):
            return None
        if ok[KP["left_hip"]] and ok[KP["right_hip"]]:
            quad = np.array([ls, rs, rh, lh], np.float32)
        else:  # hips out of view: a box below the shoulders
            down = np.array([0.0, np.linalg.norm((ls - rs) * [pose.aspect, 1.0])], np.float32)
            quad = np.array([ls, rs, rs + down, ls + down], np.float32)
        c = quad.mean(axis=0)
        return c + (quad - c) * 0.8

    def extract(self, frame: FrameData, pose: PoseObs) -> np.ndarray | None:
        quad = self.torso_polygon(pose)
        if quad is None:
            return None
        hsv = self._hsv_for(frame)
        h, w = hsv.shape[:2]
        pts = np.round(quad * [w, h]).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        mask &= cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))  # drop dark / grey pixels
        if cv2.countNonZero(mask) < 30:
            return None
        hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
        return cv2.normalize(hist, None, 1.0, 0.0, cv2.NORM_L1).flatten()


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two histograms: 0 = identical, 1 = no overlap."""
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
