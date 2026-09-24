"""Task 2: per-player keypoint smoothing and missing-keypoint handling.

Temporal filtering needs frame-to-frame correspondence, so it runs *after* Task 3 has assigned
poses to players: the main loop calls `update(pid, pose, t)` for each active player.
"""
from __future__ import annotations

from core.config import Config
from core.types import PoseObs


class PoseSmoother:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[int, object] = {}  # pid -> filter state / last good keypoints

    def update(self, pid: int, pose: PoseObs, t: float) -> PoseObs:
        """TODO(T2): return a smoothed copy of `pose`. Currently a passthrough.

        - Smooth each keypoint (core.filters.OneEuroFilter works on the whole (17, 2) array).
        - Keypoints below a confidence threshold: hold the last good value for a short time,
          then decay their confidence to 0 so gameplay ignores them (no uncontrolled actions).
        - Measure the lag the filter adds vs. the jitter it removes; this trade-off is the
          core of your evaluation.
        """
        return pose

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._state.clear()
        else:
            self._state.pop(pid, None)
