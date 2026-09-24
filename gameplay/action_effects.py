"""Task 4: visual feedback on recognised actions, aligned with the player's body.

  - grade popups (PERFECT/GOOD/OK/MISS) float up from the player's head
  - recognised moves: small label at the player's hands; gold moves in gold
  - wrist trails: the last ~0.35 s of both wrists, in the player's colour (gold during a gold
    segment), so players see their own motion being tracked
  - interactions (high five, swap, duet): an expanding ring at the contact point + label
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from core.config import Config
from core.image_utils import GOLD, put_text, to_px
from core.types import KP, EventType, GameEvent, Grade, Player

GRADE_COLORS = {
    Grade.PERFECT: GOLD,
    Grade.GOOD: (0, 200, 0),
    Grade.OK: (255, 170, 0),
    Grade.MISS: (80, 80, 255),
}
TRAIL_S = 0.35
WRISTS = (KP["left_wrist"], KP["right_wrist"])


@dataclass
class _Popup:
    text: str
    color: tuple[int, int, int]
    pos: tuple[float, float]  # normalized
    t0: float
    duration: float = 1.0
    scale: float = 1.1
    rise: bool = True


@dataclass
class _Ring:
    pos: tuple[float, float]
    color: tuple[int, int, int]
    t0: float
    duration: float = 0.6


class ActionEffects:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.min_conf = cfg.pose.min_keypoint_conf
        self._popups: list[_Popup] = []
        self._rings: list[_Ring] = []
        self._trails: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=30))

    def add_events(self, events: list[GameEvent], players: dict[int, Player], now: float) -> None:
        for ev in events:
            player = players.get(ev.pid) if ev.pid is not None else None
            head = player.head_top() if player is not None else None
            head = head or (0.5, 0.4)
            pos = ev.data.get("pos") or head
            label = ev.data.get("label")
            if ev.type is EventType.GRADE and ev.grade is not None:
                self._popups.append(_Popup(ev.grade.name, GRADE_COLORS[ev.grade], head, now))
            elif ev.type is EventType.MOVE_DETECTED and ev.move is not None:
                gold = label is not None
                self._popups.append(_Popup(label or ev.move.label, GOLD if gold else (255, 255, 255), pos, now,
                                           duration=1.2 if gold else 0.6, scale=1.0 if gold else 0.55, rise=gold))
                if gold:
                    self._rings.append(_Ring(pos, GOLD, now))
            elif ev.type is EventType.INTERACTION:
                self._popups.append(_Popup(label or "COMBO!", GOLD, head, now, duration=1.3))
                if "pos" in ev.data:
                    self._rings.append(_Ring(ev.data["pos"], (255, 255, 255), now, 0.8))
            elif ev.type is EventType.PLAYER_RETURNED and player is not None:
                self._popups.append(_Popup("welcome back!", player.color, head, now, scale=0.7))

    def update(self, players: dict[int, Player], now: float) -> None:
        """Record wrist positions for the trails (call once per frame)."""
        for pid, p in players.items():
            for k in WRISTS:
                trail = self._trails[(pid, k)]
                if p.is_active and p.pose is not None and p.pose.confidence[k] >= self.min_conf:
                    trail.append((now, tuple(p.pose.keypoints[k])))
                while trail and now - trail[0][0] > TRAIL_S:
                    trail.popleft()

    def draw(self, canvas: np.ndarray, players: dict[int, Player], now: float, gold: bool = False) -> None:
        for (pid, _), trail in self._trails.items():
            player = players.get(pid)
            if player is None or not player.is_active or len(trail) < 2:
                continue
            color = GOLD if gold else player.color
            pts = [to_px(xy, canvas.shape) for _, xy in trail]
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, 2 + int(8 * i / len(pts)), cv2.LINE_AA)

        self._rings = [r for r in self._rings if now - r.t0 < r.duration]
        for r in self._rings:
            age = (now - r.t0) / r.duration
            radius = int(20 + 140 * age)
            cv2.circle(canvas, to_px(r.pos, canvas.shape), radius, r.color, max(1, int(8 * (1 - age))), cv2.LINE_AA)

        self._popups = [p for p in self._popups if now - p.t0 < p.duration]
        for p in self._popups:
            age = (now - p.t0) / p.duration
            x, y = to_px(p.pos, canvas.shape)
            dy = int(40 + 60 * age) if p.rise else 10
            put_text(canvas, p.text, (x, y - dy), p.scale, p.color, 2, anchor="center")

    def clear(self) -> None:
        self._popups.clear()
        self._rings.clear()
        self._trails.clear()
