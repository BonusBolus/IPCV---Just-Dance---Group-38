"""Task 3: appearance model for re-identification (who is who after crossing or re-entry)."""
from __future__ import annotations

import cv2
import numpy as np

from core.types import FrameData, PoseObs


class AppearanceModel:
    """TODO(T3): describe a player's look so identities survive crossings and leaving the frame.

    Suggested start: HSV colour histogram of the torso region (between shoulders and hips).
    Clothing colour is cheap and fairly distinctive between two players. Update it slowly
    (running average) while the track is confident, and freeze it during occlusion.
    """

    def extract(self, frame: FrameData, pose: PoseObs) -> np.ndarray | None:
        """TODO(T3): return a normalized feature vector for this person, or None if not visible."""
        return None


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two histograms: 0 = identical, 1 = no overlap."""
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
