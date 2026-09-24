"""Task 1: temporal smoothing of each player's face.

Smoothing needs to know which face in frame t matches which in frame t-1, so it runs *after*
Task 3 has assigned faces to players: the main loop calls `update(pid, face)` per player.
"""
from __future__ import annotations

from core.config import Config
from core.types import FaceObs


class FaceSmoother:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._filters: dict[int, object] = {}  # pid -> filter state

    def update(self, pid: int, face: FaceObs, t: float) -> FaceObs:
        """TODO(T1): smooth bbox (and yaw/pitch/roll) per player, e.g. with core.filters.OneEuroFilter.

        Also decide what happens during short detection dropouts (hold the last face for N ms,
        fade out the effect, ...). Currently a passthrough.
        """
        return face

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._filters.clear()
        else:
            self._filters.pop(pid, None)
