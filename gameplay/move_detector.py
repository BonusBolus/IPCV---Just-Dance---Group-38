"""Task 4: recognise discrete moves/gestures (the ">= 2 distinguishable actions" requirement).

Rule-based classification on the normalized pose (origin = hip centre, 1 unit = torso length,
y down, isotropic), so it works for any body size or distance. Rules are checked in priority
order, and the first match wins:
    ARMS_UP      both wrists more than 0.6 above the shoulder line
    CLAP         wrists closer than 0.35 to each other, above the hips
    T_POSE       both arms horizontal (elbows and wrists within 0.3 of shoulder height),
                 wrists more than 0.6 outside the shoulders
    POINT_LEFT   one arm horizontal to the screen's left, the other hand low (and mirrored)
    SQUAT        hip-to-ankle height below 1.3 torso lengths (standing ~1.6), or both knees
                 bent below 140 deg when the ankles are out of view
Unreliable keypoints (NaN) make a rule fail: missing data never triggers an action.

Temporal logic against accidental actions:
    confirmation  the same move must be classified for `move_confirm_s`
    cooldown      a confirmed move fires again at most once per `move_cooldown_s` per player
Uses: gold moves during the song (bonus), and the "raise both arms" menu gesture.
"""
from __future__ import annotations

import numpy as np

from core.config import Config
from core.types import KP, EventType, GameEvent, MoveType, Player, PoseObs
from gameplay.choreography import Choreography
from pose.features import joint_angles, normalize_pose


class Cooldown:
    """Prevents an action from firing repeatedly. Keys are anything hashable, e.g. (pid, move)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict = {}

    def ready(self, key, t: float) -> bool:
        return t - self._last.get(key, float("-inf")) >= self.seconds

    def trigger(self, key, t: float) -> None:
        self._last[key] = t

    def try_trigger(self, key, t: float) -> bool:
        """Trigger and return True if ready, else return False."""
        if self.ready(key, t):
            self.trigger(key, t)
            return True
        return False

    def reset(self) -> None:
        self._last.clear()


def _nanmean(*values: float) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def classify_pose(pose: PoseObs, min_conf: float = 0.5) -> MoveType:
    n = normalize_pose(pose, min_conf)
    if n is None:
        return MoveType.NONE
    g = lambda name: n[KP[name]]  # noqa: E731
    ls, rs, le, re, lw, rw = (g(k) for k in ("left_shoulder", "right_shoulder", "left_elbow",
                                             "right_elbow", "left_wrist", "right_wrist"))
    with np.errstate(invalid="ignore"):
        sh_y = _nanmean(ls[1], rs[1])
        if lw[1] < sh_y - 0.6 and rw[1] < sh_y - 0.6:
            return MoveType.ARMS_UP
        if np.linalg.norm(lw - rw) < 0.35 and lw[1] < -0.3 and rw[1] < -0.3:
            return MoveType.CLAP

        def horizontal(s, e, w) -> bool:
            return abs(e[1] - s[1]) < 0.3 and abs(w[1] - s[1]) < 0.3

        def low(s, w) -> bool:
            return w[1] > s[1] + 0.4

        l_out = horizontal(ls, le, lw) and lw[0] > ls[0] + 0.6   # left_* arm is on the image right
        r_out = horizontal(rs, re, rw) and rw[0] < rs[0] - 0.6
        if l_out and r_out:
            return MoveType.T_POSE
        if r_out and low(ls, lw):
            return MoveType.POINT_LEFT
        if l_out and low(rs, rw):
            return MoveType.POINT_RIGHT

        la, ra, lh, rh = g("left_ankle"), g("right_ankle"), g("left_hip"), g("right_hip")
        leg = _nanmean(la[1] - lh[1], ra[1] - rh[1])
        if not np.isnan(leg):
            if leg < 1.3:
                return MoveType.SQUAT
        else:
            angles = joint_angles(pose, min_conf)
            if angles.get("left_knee", 180) < 140 and angles.get("right_knee", 180) < 140:
                return MoveType.SQUAT
    return MoveType.NONE


class MoveDetector:
    def __init__(self, cfg: Config):
        self.gc = cfg.gameplay
        self.min_conf = cfg.pose.min_keypoint_conf
        self.cooldown = Cooldown(self.gc.move_cooldown_s)
        self._candidate: dict[int, tuple[MoveType, float]] = {}   # pid -> (move, since)
        self._gold_awarded: set[tuple[int, int]] = set()

    def update(self, players: dict[int, Player], now: float, song_t: float,
               choreo: Choreography | None = None) -> list[GameEvent]:
        events = []
        for pid, p in players.items():
            if not p.is_active or p.pose is None:
                self._candidate.pop(pid, None)
                continue
            move = self.classify(p.pose)
            cand = self._candidate.get(pid)
            if cand is None or cand[0] is not move:
                self._candidate[pid] = cand = (move, now)
            if move is MoveType.NONE or now - cand[1] < self.gc.move_confirm_s:
                continue
            if not self.cooldown.try_trigger((pid, move), now):
                continue
            data = {"pos": _hands_center(p.pose)}
            if choreo is not None:
                seg_i, seg = choreo.segment_at(song_t - self.gc.timing_tolerance_s / 2)
                if seg is not None and seg.gold and seg.move is move and (pid, seg_i) not in self._gold_awarded:
                    self._gold_awarded.add((pid, seg_i))
                    data.update(points=self.gc.gold_bonus, label="GOLD MOVE!")
            events.append(GameEvent(EventType.MOVE_DETECTED, song_t, pid=pid, move=move, data=data))
        return events

    def classify(self, pose: PoseObs) -> MoveType:
        return classify_pose(pose, self.min_conf)

    def current(self, pid: int) -> MoveType:
        cand = self._candidate.get(pid)
        return MoveType.NONE if cand is None else cand[0]

    def held_for(self, pid: int, move: MoveType, now: float) -> float:
        """How long the player has been doing `move` (0 if they are doing something else)."""
        cand = self._candidate.get(pid)
        return now - cand[1] if cand is not None and cand[0] is move else 0.0

    def reset(self) -> None:
        self.cooldown.reset()
        self._candidate.clear()
        self._gold_awarded.clear()


def _hands_center(pose: PoseObs) -> tuple[float, float] | None:
    w = pose.keypoints[[KP["left_wrist"], KP["right_wrist"]]]
    ok = pose.confidence[[KP["left_wrist"], KP["right_wrist"]]] >= 0.3
    if not ok.any():
        return None
    x, y = w[ok].mean(axis=0)
    return float(x), float(y)
