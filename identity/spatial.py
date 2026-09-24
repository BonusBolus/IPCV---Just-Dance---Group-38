"""Task 3: convert image positions into physical positions (metres) for Task 4."""
from __future__ import annotations

import math

from core.config import Config
from core.types import Player


def focal_length_px(image_width_px: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from the horizontal field of view."""
    return (image_width_px / 2) / math.tan(math.radians(hfov_deg) / 2)


class SpatialEstimator:
    """TODO(T3): estimate each player's (lateral x, depth z) in metres.

    Pinhole model: Z = f * W_real / W_px, using a body measure of roughly known real size,
    e.g. shoulder width (~0.40 m) or inter-pupil distance (~0.063 m). Then X = (u - cx) * Z / f.
    Report how reliable this is (compare to tape-measured distances).
    Task 4 uses it for e.g. "high five": wrists of both players within 15 cm.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def estimate(self, player: Player, frame_shape) -> tuple[float, float] | None:
        return None
