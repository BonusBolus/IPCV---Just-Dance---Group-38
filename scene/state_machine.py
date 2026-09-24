"""Game flow: START -> LOBBY -> COUNTDOWN -> PLAYING -> RESULTS -> START."""
from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Callable

from core.config import GameConfig
from core.types import Player

log = logging.getLogger(__name__)


class Phase(Enum):
    START = "start"          # title screen, waits for SPACE (TODO(T4): or an ARMS_UP gesture)
    LOBBY = "lobby"          # waits until all players are visible for lobby_confirm_s
    COUNTDOWN = "countdown"  # 3-2-1
    PLAYING = "playing"      # song + scoring
    RESULTS = "results"      # winner screen

    def next(self) -> Phase:
        order = list(Phase)
        return order[(order.index(self) + 1) % len(order)]


class PhaseController:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.phase = Phase.START
        self.phase_start = 0.0
        self._ready_since: float | None = None
        self._hooks: dict[Phase, list[Callable[[], None]]] = defaultdict(list)

    def on_enter(self, phase: Phase, fn: Callable[[], None]) -> None:
        self._hooks[phase].append(fn)

    def go(self, phase: Phase, now: float) -> None:
        log.info("Phase %s -> %s", self.phase.name, phase.name)
        self.phase = phase
        self.phase_start = now
        self._ready_since = None
        for fn in self._hooks[phase]:
            fn()

    def advance(self, now: float) -> None:
        """Manual skip to the next phase (keyboard)."""
        self.go(self.phase.next(), now)

    def time_in_phase(self, now: float) -> float:
        return now - self.phase_start

    def update(self, players: dict[int, Player], now: float, song_finished: bool) -> None:
        n_active = sum(p.is_active for p in players.values())

        if self.phase is Phase.LOBBY:
            if n_active >= self.cfg.num_players:
                if self._ready_since is None:
                    self._ready_since = now
                elif now - self._ready_since >= self.cfg.lobby_confirm_s:
                    self.go(Phase.COUNTDOWN, now)
            else:
                self._ready_since = None
        elif self.phase is Phase.COUNTDOWN:
            if self.time_in_phase(now) >= self.cfg.countdown_s:
                self.go(Phase.PLAYING, now)
        elif self.phase is Phase.PLAYING:
            if song_finished:
                self.go(Phase.RESULTS, now)
        elif self.phase is Phase.RESULTS:
            if self.time_in_phase(now) >= self.cfg.results_s:
                self.go(Phase.START, now)

    # ------------------------------------------------------------------ for the renderer
    def lobby_progress(self, now: float) -> float:
        if self._ready_since is None:
            return 0.0
        return min(1.0, (now - self._ready_since) / self.cfg.lobby_confirm_s)

    def countdown_remaining(self, now: float) -> float:
        return max(0.0, self.cfg.countdown_s - self.time_in_phase(now))
