"""Scores and combos per player. Updated only through GameEvents, so the game logic stays traceable."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from core.types import EventType, GameEvent, Grade

GRADE_POINTS = {Grade.PERFECT: 100, Grade.GOOD: 60, Grade.OK: 30, Grade.MISS: 0}


@dataclass
class PlayerStats:
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    grades: Counter = field(default_factory=Counter)
    last_grade: Grade | None = None
    last_grade_t: float = float("-inf")

    def register_grade(self, grade: Grade, t: float, bonus: int = 0) -> None:
        self.grades[grade] += 1
        self.last_grade, self.last_grade_t = grade, t
        self.combo = self.combo + 1 if grade in (Grade.PERFECT, Grade.GOOD) else 0
        self.max_combo = max(self.max_combo, self.combo)
        self.score += GRADE_POINTS[grade] + bonus


class GameState:
    def __init__(self):
        self.stats: dict[int, PlayerStats] = {}
        self.log: list[GameEvent] = []

    def reset(self, pids) -> None:
        self.stats = {pid: PlayerStats() for pid in pids}
        self.log.clear()

    def stats_for(self, pid: int) -> PlayerStats:
        return self.stats.setdefault(pid, PlayerStats())

    def apply(self, events: list[GameEvent]) -> None:
        for ev in events:
            self.log.append(ev)
            if ev.pid is None:
                continue
            stats = self.stats_for(ev.pid)
            points = int(ev.data.get("points", 0))
            if ev.type is EventType.GRADE and ev.grade is not None:
                stats.register_grade(ev.grade, ev.t, bonus=points)
            else:
                stats.score += points

    def leader(self) -> int | None:
        """pid with the highest score; None when empty or tied."""
        if not self.stats:
            return None
        ranked = sorted(self.stats.items(), key=lambda kv: kv[1].score, reverse=True)
        if len(ranked) > 1 and ranked[0][1].score == ranked[1][1].score:
            return None
        return ranked[0][0]

    def winner(self) -> int | None:
        return self.leader()

    def balance(self, pid_a: int = 1, pid_b: int = 2) -> float:
        """-1 (all points to a) .. +1 (all to b): e.g. for a tug-of-war bar."""
        a = self.stats.get(pid_a, PlayerStats()).score
        b = self.stats.get(pid_b, PlayerStats()).score
        return 0.0 if a + b == 0 else (b - a) / (a + b)
