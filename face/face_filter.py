"""Task 1: temporal smoothing of each player's face, and controlled handling of dropouts.

Smoothing needs to know which face in frame t matches which in frame t-1, so it runs *after*
Task 3 has assigned faces to players: the main loop calls `update(pid, face_or_None, t)`.

- Box (centre, size): One Euro filter, so the effect is steady when the head is still but
  follows fast head movement without lag.
- Roll/yaw: a separate One Euro filter with angle-scaled parameters.
- Landmarks: moved along with the smoothed box (same offset/scale as the raw box), which
  removes the global jitter at almost no cost.
- Missed detection: the last face is held for `hold_s` with a fading confidence, so the effect
  fades out instead of flickering. Longer gaps reset the filter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from core.config import Config
from core.filters import OneEuroFilter
from core.types import FaceObs


@dataclass
class _Track:
    box: OneEuroFilter
    angles: OneEuroFilter
    last: FaceObs
    last_seen: float


class FaceSmoother:
    def __init__(self, cfg: Config):
        self.fc = cfg.face
        self._tracks: dict[int, _Track] = {}

    def update(self, pid: int, face: FaceObs | None, t: float) -> FaceObs | None:
        track = self._tracks.get(pid)
        if face is None:
            if track is None:
                return None
            age = t - track.last_seen
            if age > self.fc.hold_s:
                self._tracks.pop(pid, None)
                return None
            return replace(track.last, confidence=track.last.confidence * (1.0 - age / self.fc.hold_s))

        if track is None or t - track.last_seen > 0.5:
            track = _Track(OneEuroFilter(self.fc.min_cutoff, self.fc.beta),
                           OneEuroFilter(1.0, 0.02), face, t)
            self._tracks[pid] = track

        x, y, w, h = face.bbox
        cx, cy, sw, sh = track.box(np.array([x + w / 2, y + h / 2, w, h], np.float32), t)
        roll, yaw, pitch = track.angles(np.array([face.roll, face.yaw, face.pitch], np.float32), t)

        landmarks = face.landmarks
        if landmarks is not None and w > 0 and h > 0:
            rel = (landmarks - np.array([x + w / 2, y + h / 2], np.float32)) / np.array([w, h], np.float32)
            landmarks = (rel * np.array([sw, sh], np.float32) + np.array([cx, cy], np.float32)).astype(np.float32)

        out = replace(face, bbox=(float(cx - sw / 2), float(cy - sh / 2), float(sw), float(sh)),
                      landmarks=landmarks, roll=float(roll), yaw=float(yaw), pitch=float(pitch), timestamp=t)
        track.last, track.last_seen = out, t
        return out

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._tracks.clear()
        else:
            self._tracks.pop(pid, None)
