"""Reference choreography: what the players should be doing at each moment of the song.

File format (assets/choreography.json), produced by tools/extract_reference.py:
{
  "version": 1, "song": "song.mp3", "duration": 62.0,
  "frames": [{"t": 0.033, "kp": [[x, y], ... 17], "conf": [c, ... 17]}, ...],
  "moves":  [{"start": 4.0, "end": 6.0, "move": "ARMS_UP", "gold": false, "duet": false}, ...]
}
Keypoints are normalized to the reference video frame. Moves are annotated by hand.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mock_source import body_pose, dance_angles, to_image
from core.types import NUM_KEYPOINTS, MoveType, PoseObs

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoveSegment:
    start: float
    end: float
    move: MoveType
    gold: bool = False   # bonus move
    duet: bool = False   # both players must do it together (player-player interaction)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


class Choreography:
    def __init__(self, times, keypoints, confidence, moves: list[MoveSegment],
                 duration: float, song: str | None = None, is_placeholder: bool = False):
        self.times = np.asarray(times, np.float64)
        self.keypoints = np.asarray(keypoints, np.float32).reshape(-1, NUM_KEYPOINTS, 2)
        self.confidence = np.asarray(confidence, np.float32).reshape(-1, NUM_KEYPOINTS)
        self.moves = sorted(moves, key=lambda m: m.start)
        self.duration = float(duration)
        self.song = song
        self.is_placeholder = is_placeholder

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path: str | Path) -> Choreography:
        data = json.loads(Path(path).read_text())
        frames = data.get("frames", [])
        times = [f["t"] for f in frames]
        moves = [
            MoveSegment(m["start"], m["end"], MoveType[m["move"]], m.get("gold", False), m.get("duet", False))
            for m in data.get("moves", [])
        ]
        return cls(
            times,
            [f["kp"] for f in frames],
            [f.get("conf", [1.0] * NUM_KEYPOINTS) for f in frames],
            moves,
            data.get("duration", times[-1] if times else 0.0),
            data.get("song"),
        )

    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "song": self.song,
            "duration": self.duration,
            "frames": [
                {"t": round(float(t), 4), "kp": np.round(kp, 5).tolist(), "conf": np.round(c, 3).tolist()}
                for t, kp, c in zip(self.times, self.keypoints, self.confidence)
            ],
            "moves": [
                {"start": m.start, "end": m.end, "move": m.move.name, "gold": m.gold, "duet": m.duet}
                for m in self.moves
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def placeholder(cls, duration: float, fps: float = 30.0) -> Choreography:
        """Synthetic stick-figure dance so the game runs before the real reference exists."""
        times = np.arange(0.0, duration, 1.0 / fps)
        kps = [to_image(body_pose(*dance_angles(t)), (0.5, 0.55), 0.25, 1.0) for t in times]
        conf = np.ones((len(times), NUM_KEYPOINTS), np.float32)
        cycle = [MoveType.ARMS_UP, MoveType.T_POSE, MoveType.CLAP, MoveType.SQUAT,
                 MoveType.POINT_LEFT, MoveType.POINT_RIGHT]
        moves = [
            MoveSegment(s, s + 3.0, cycle[i % len(cycle)], gold=(i % 4 == 3), duet=(i % 5 == 4))
            for i, s in enumerate(np.arange(2.0, duration - 3.0, 4.0))
        ]
        return cls(times, kps, conf, moves, duration, is_placeholder=True)

    @classmethod
    def load_or_placeholder(cls, path: str | Path | None, duration: float) -> Choreography:
        if path and Path(path).exists():
            try:
                return cls.load(path)
            except Exception:
                log.exception("Could not read choreography %s", path)
        log.warning("No choreography at %s, using a placeholder dance", path)
        return cls.placeholder(duration)

    # ------------------------------------------------------------------ lookup
    def reference_at(self, t: float) -> PoseObs | None:
        """Reference pose nearest to song time t (None outside the dance)."""
        n = len(self.times)
        if n == 0 or t < self.times[0] or t > self.times[-1]:
            return None
        i = int(np.searchsorted(self.times, t))
        if i > 0 and (i == n or t - self.times[i - 1] < self.times[i] - t):
            i -= 1
        return PoseObs(self.keypoints[i], self.confidence[i], timestamp=float(self.times[i]))

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All reference frames with t0 <= t <= t1 as (times, keypoints, confidence).

        For scoring with a timing tolerance: compare the live pose against every frame in
        [t - tol, t + tol] and keep the best match.
        """
        i0 = int(np.searchsorted(self.times, t0, side="left"))
        i1 = int(np.searchsorted(self.times, t1, side="right"))
        return self.times[i0:i1], self.keypoints[i0:i1], self.confidence[i0:i1]

    def move_at(self, t: float) -> MoveSegment | None:
        for m in self.moves:
            if m.contains(t):
                return m
            if m.start > t:
                break
        return None

    def next_move(self, t: float) -> MoveSegment | None:
        return next((m for m in self.moves if m.start > t), None)

    def progress(self, t: float) -> float:
        return float(np.clip(t / self.duration, 0.0, 1.0)) if self.duration > 0 else 0.0
