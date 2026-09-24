"""Reference choreography: what the players should be doing at each moment of the song.

File format (assets/choreography.json):
{
  "version": 1, "song": "song.wav", "duration": 64.0, "bpm": 120, "aspect": 1.0,
  "frames": [{"t": 0.033, "kp": [[x, y], ... 17], "conf": [c, ... 17]}, ...],
  "moves":  [{"start": 4.0, "end": 6.0, "move": "ARMS_UP", "gold": false, "duet": false}, ...]
}
Keypoints are normalized to the reference frame, whose width/height ratio is `aspect`.
Two ways to make one:
  - `Choreography.synthetic(duration, bpm)`: a beat-aligned dance built from the move pose
    library below (default, used by tools/make_default_assets.py)
  - tools/extract_reference.py: from a video of a real model dancer (moves annotated by hand)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mock_source import body_pose, to_image
from core.types import NUM_KEYPOINTS, MoveType, PoseObs

log = logging.getLogger(__name__)

# Move pose library: body_pose() parameters. Arms are (upper-arm angle, forearm angle).
# "POINT_LEFT" = pointing to the screen's left: for the camera-facing dancer that is the
# right_* arm (the pose model labels by appearance).
MOVE_POSES: dict[MoveType, dict] = {
    MoveType.NONE: dict(left=(20, 20), right=(20, 20)),
    MoveType.ARMS_UP: dict(left=(170, 170), right=(170, 170)),
    MoveType.T_POSE: dict(left=(90, 90), right=(90, 90)),
    MoveType.CLAP: dict(left=(5, 265), right=(5, 265)),        # hands together at the chest
    MoveType.SQUAT: dict(left=(45, 70), right=(45, 70), squat=1.0),
    MoveType.POINT_LEFT: dict(left=(15, 15), right=(90, 90)),
    MoveType.POINT_RIGHT: dict(left=(90, 90), right=(15, 15)),
    MoveType.HIGH_FIVE: dict(left=(120, 150), right=(20, 20)),
    MoveType.SWAP: dict(left=(35, 60), right=(35, 60)),
}


def move_body(move: MoveType, bob: float = 0.0) -> np.ndarray:
    return body_pose(bob=bob, **MOVE_POSES[move])


@dataclass(frozen=True)
class MoveSegment:
    start: float
    end: float
    move: MoveType
    gold: bool = False   # bonus move: recognising the move gives extra points
    duet: bool = False   # both players must hit it together (player-player interaction)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


class Choreography:
    def __init__(self, times, keypoints, confidence, moves: list[MoveSegment], duration: float,
                 song: str | None = None, is_placeholder: bool = False, bpm: float = 120.0,
                 aspect: float = 1.0):
        self.times = np.asarray(times, np.float64)
        self.keypoints = np.asarray(keypoints, np.float32).reshape(-1, NUM_KEYPOINTS, 2)
        self.confidence = np.asarray(confidence, np.float32).reshape(-1, NUM_KEYPOINTS)
        self.moves = sorted(moves, key=lambda m: m.start)
        self.duration = float(duration)
        self.song = song
        self.is_placeholder = is_placeholder
        self.bpm = float(bpm)
        self.aspect = float(aspect)

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
            bpm=data.get("bpm", 120.0),
            aspect=data.get("aspect", 1.0),
        )

    def save(self, path: str | Path) -> None:
        data = {
            "version": 1,
            "song": self.song,
            "duration": self.duration,
            "bpm": self.bpm,
            "aspect": self.aspect,
            "frames": [
                {"t": round(float(t), 4), "kp": np.round(kp, 4).tolist(), "conf": np.round(c, 3).tolist()}
                for t, kp, c in zip(self.times, self.keypoints, self.confidence)
            ],
            "moves": [
                {"start": round(m.start, 4), "end": round(m.end, 4), "move": m.move.name,
                 "gold": m.gold, "duet": m.duet}
                for m in self.moves
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def synthetic(cls, duration: float, bpm: float = 120.0, fps: float = 30.0, song: str | None = None) -> Choreography:
        """Beat-aligned dance from the move pose library.

        Every move lasts one bar (4 beats): the dancer moves into the pose on beat 1 and holds
        it for beats 2-4 (the scored segment), bobbing on the beat. HIGH_FIVE lasts 2 bars and
        SWAP 3 bars (walking takes time). Every 7th pose move is gold, every 5th a duet.
        """
        beat = 60.0 / bpm
        bar = 4 * beat
        pose_cycle = [MoveType.ARMS_UP, MoveType.T_POSE, MoveType.CLAP, MoveType.POINT_LEFT,
                      MoveType.POINT_RIGHT, MoveType.SQUAT, MoveType.T_POSE, MoveType.CLAP,
                      MoveType.ARMS_UP, MoveType.POINT_RIGHT, MoveType.POINT_LEFT, MoveType.SQUAT]
        moves: list[MoveSegment] = []
        t, n_pose, placed = bar, 0, set()   # one bar of groove as intro
        while t + bar <= duration - bar:    # and at least one bar of outro
            frac = t / duration
            if frac > 0.33 and "high_five" not in placed and t + 2 * bar <= duration - bar:
                moves.append(MoveSegment(t + beat, t + 2 * bar, MoveType.HIGH_FIVE))
                placed.add("high_five")
                t += 2 * bar
                continue
            if frac > 0.62 and "swap" not in placed and t + 3 * bar <= duration - bar:
                moves.append(MoveSegment(t + beat, t + 3 * bar, MoveType.SWAP))
                placed.add("swap")
                t += 3 * bar
                continue
            move = pose_cycle[n_pose % len(pose_cycle)]
            n_pose += 1
            moves.append(MoveSegment(t + beat, t + bar, move, gold=(n_pose % 7 == 0), duet=(n_pose % 5 == 0)))
            t += bar

        times = np.arange(0.0, duration, 1.0 / fps)
        seg_starts = [m.start - beat for m in moves]  # the pose transition starts one beat early
        kps = []
        for ts in times:
            i = int(np.searchsorted(seg_starts, ts, side="right")) - 1
            target = moves[i].move if i >= 0 and ts < moves[i].end else MoveType.NONE
            prev = moves[i - 1].move if i >= 1 else MoveType.NONE
            bob = 0.06 * abs(np.sin(np.pi * ts / beat))
            if i >= 0 and ts < seg_starts[i] + beat:   # transition beat: blend previous -> target
                s = (ts - seg_starts[i]) / beat
                s = s * s * (3 - 2 * s)                 # smoothstep
                body = (1 - s) * move_body(prev, bob) + s * move_body(target, bob)
            else:
                body = move_body(target, bob)
            kps.append(to_image(body, (0.5, 0.55), 0.25, 1.0))
        conf = np.ones((len(times), NUM_KEYPOINTS), np.float32)
        return cls(times, kps, conf, moves, duration, song=song, bpm=bpm, aspect=1.0)

    @classmethod
    def placeholder(cls, duration: float, fps: float = 30.0) -> Choreography:
        c = cls.synthetic(duration, fps=fps)
        c.is_placeholder = True
        return c

    @classmethod
    def load_or_placeholder(cls, path: str | Path | None, duration: float) -> Choreography:
        if path and Path(path).exists():
            try:
                return cls.load(path)
            except Exception:
                log.exception("Could not read choreography %s", path)
        log.warning("No choreography at %s, using a generated dance", path)
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
        return PoseObs(self.keypoints[i], self.confidence[i], timestamp=float(self.times[i]), aspect=self.aspect)

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All reference frames with t0 <= t <= t1 as (times, keypoints, confidence).

        For scoring with a timing tolerance: compare the live pose against every frame in
        the window and keep the best match.
        """
        i0 = int(np.searchsorted(self.times, t0, side="left"))
        i1 = int(np.searchsorted(self.times, t1, side="right"))
        return self.times[i0:i1], self.keypoints[i0:i1], self.confidence[i0:i1]

    def segment_at(self, t: float) -> tuple[int, MoveSegment] | tuple[None, None]:
        for i, m in enumerate(self.moves):
            if m.contains(t):
                return i, m
            if m.start > t:
                break
        return None, None

    def move_at(self, t: float) -> MoveSegment | None:
        return self.segment_at(t)[1]

    def next_move(self, t: float) -> MoveSegment | None:
        return next((m for m in self.moves if m.start > t), None)

    def progress(self, t: float) -> float:
        return float(np.clip(t / self.duration, 0.0, 1.0)) if self.duration > 0 else 0.0

    def beat_phase(self, t: float) -> float:
        """0 on the beat, rising to 1 just before the next beat."""
        beat = 60.0 / self.bpm
        return (t % beat) / beat if t >= 0 else 0.0
