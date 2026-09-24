"""Task 2: per-player keypoint smoothing and missing-keypoint handling.

Temporal filtering needs frame-to-frame correspondence, so it runs *after* Task 3 has assigned
poses to players: the main loop calls `update(pid, pose, t)` for each active player.

Per keypoint and per frame:
    confident (conf >= min_keypoint_conf) -> feed the raw position into the One Euro filter
    unreliable, but last seen < hold_s ago -> feed the last good position (the filter keeps
                                              running without jumping), confidence fades out
    unreliable for longer                   -> confidence 0: gameplay ignores the keypoint
So a keypoint that flickers for a frame or two never causes a jump or an accidental action.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import Config
from core.filters import OneEuroFilter
from core.types import PoseObs


@dataclass
class _Track:
    filt: OneEuroFilter
    last_good: np.ndarray      # (17, 2) last confident position
    last_good_t: np.ndarray    # (17,) time it was confident
    last_t: float = 0.0


class PoseSmoother:
    def __init__(self, cfg: Config):
        self.pc = cfg.pose
        self._tracks: dict[int, _Track] = {}

    def update(self, pid: int, pose: PoseObs, t: float) -> PoseObs:
        track = self._tracks.get(pid)
        if track is None or t - track.last_t > 0.5:  # new track or back after a gap: restart
            track = self._new_track(pose, t)
            self._tracks[pid] = track

        good = pose.confidence >= self.pc.min_keypoint_conf
        track.last_good[good] = pose.keypoints[good]
        track.last_good_t[good] = t

        # Held keypoints stay just below the reliability threshold and fade out: they keep the
        # skeleton drawing continuous, but gameplay (which requires min_keypoint_conf) ignores them.
        age = t - track.last_good_t
        fade = np.clip(1.0 - age / self.pc.hold_s, 0.0, 1.0)
        held_conf = 0.99 * self.pc.min_keypoint_conf * fade
        conf = np.where(good, pose.confidence, held_conf).astype(np.float32)
        target = np.where(good[:, None], pose.keypoints, track.last_good)

        smoothed = track.filt(target, t)
        track.last_t = t
        out = PoseObs(smoothed.astype(np.float32), conf, timestamp=t, aspect=pose.aspect, mask=pose.mask)
        return out

    def velocity(self, pid: int) -> np.ndarray | None:
        """Filtered keypoint velocity (normalized units / s) of the last update."""
        track = self._tracks.get(pid)
        return None if track is None else track.filt.velocity

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._tracks.clear()
        else:
            self._tracks.pop(pid, None)

    def _new_track(self, pose: PoseObs, t: float) -> _Track:
        return _Track(
            filt=OneEuroFilter(self.pc.min_cutoff, self.pc.beta),
            last_good=pose.keypoints.copy(),
            last_good_t=np.where(pose.confidence >= self.pc.min_keypoint_conf, t, -np.inf).astype(np.float64),
            last_t=t,
        )
