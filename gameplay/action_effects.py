"""Task 4: visual feedback for recognised actions (grade popups, body-aligned effects)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import Config
from core.image_utils import GOLD, put_text, to_px
from core.types import EventType, GameEvent, Grade, Player

GRADE_COLORS = {
    Grade.PERFECT: GOLD,
    Grade.GOOD: (0, 200, 0),
    Grade.OK: (255, 170, 0),
    Grade.MISS: (80, 80, 255),
}


@dataclass
class _Popup:
    text: str
    color: tuple[int, int, int]
    pos: tuple[float, float]  # normalized
    t0: float
    duration: float = 1.0


class ActionEffects:
    """Floating text popups at the player that triggered an event.

    TODO(T4): body-aligned effects, e.g. sparkles/trails on the wrists during gold moves,
    a flash between both players on a successful duet move.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._popups: list[_Popup] = []

    def add_events(self, events: list[GameEvent], players: dict[int, Player], now: float) -> None:
        for ev in events:
            player = players.get(ev.pid) if ev.pid is not None else None
            anchor = player.head_top() if player is not None else None
            if anchor is None:
                anchor = (0.5, 0.4)
            if ev.type is EventType.GRADE and ev.grade is not None:
                self._popups.append(_Popup(ev.grade.name, GRADE_COLORS[ev.grade], anchor, now))
            elif ev.type is EventType.MOVE_DETECTED and ev.move is not None:
                self._popups.append(_Popup(ev.move.name.replace("_", " "), (255, 255, 255), anchor, now, 0.7))
            elif ev.type is EventType.INTERACTION:
                self._popups.append(_Popup(ev.data.get("label", "COMBO!"), GOLD, anchor, now))

    def draw(self, canvas: np.ndarray, players: dict[int, Player], now: float) -> None:
        self._popups = [p for p in self._popups if now - p.t0 < p.duration]
        for p in self._popups:
            age = (now - p.t0) / p.duration
            x, y = to_px(p.pos, canvas.shape)
            put_text(canvas, p.text, (x, int(y - 40 - 60 * age)), 1.1, p.color, 2, anchor="center")

    def clear(self) -> None:
        self._popups.clear()
