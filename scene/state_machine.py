"""Game flow: START -> LOBBY -> COUNTDOWN -> PLAYING -> RESULTS -> START (or straight to LOBBY).

Vision-based menu control: "raise both arms" (ARMS_UP held for start_gesture_s) replaces the
keyboard. SPACE still works as a fallback.
    START      any player raises both arms              -> LOBBY
    LOBBY      each player raises both arms = ready;
               all `num_players` visible and ready,
               held for lobby_confirm_s                  -> COUNTDOWN
               (a player who disappears is un-readied)
    COUNTDOWN  after countdown_s                         -> PLAYING
    PLAYING    song finished                             -> RESULTS
               (a lost player keeps the song running; their segments are graded MISS)
    RESULTS    arms up (after 3 s)                       -> LOBBY (play again)
               or results_s timeout                      -> START
"""
from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Callable

from core.config import GameConfig, GameplayConfig
from core.types import Player

log = logging.getLogger(__name__)


class Phase(Enum):
    START = "start"
    LOBBY = "lobby"
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    RESULTS = "results"

    def next(self) -> Phase:
        order = list(Phase)
        return order[(order.index(self) + 1) % len(order)]


class PhaseController:
    def __init__(self, cfg: GameConfig, gameplay: GameplayConfig | None = None):
        self.cfg = cfg
        self.gesture_s = (gameplay or GameplayConfig()).start_gesture_s
        self.phase = Phase.START
        self.phase_start = 0.0
        self.ready: set[int] = set()
        self._ready_since: float | None = None
        self._hooks: dict[Phase, list[Callable[[], None]]] = defaultdict(list)

    def on_enter(self, phase: Phase, fn: Callable[[], None]) -> None:
        self._hooks[phase].append(fn)

    def go(self, phase: Phase, now: float) -> None:
        log.info("Phase %s -> %s", self.phase.name, phase.name)
        self.phase = phase
        self.phase_start = now
        self._ready_since = None
        self.ready.clear()
        for fn in self._hooks[phase]:
            fn()

    def advance(self, now: float) -> None:
        """Manual skip to the next phase (keyboard)."""
        self.go(self.phase.next(), now)

    def time_in_phase(self, now: float) -> float:
        return now - self.phase_start

    def update(self, players: dict[int, Player], now: float, song_finished: bool,
               arms_up: dict[int, float] | None = None) -> None:
        """`arms_up`: pid -> how long that player has been holding both arms up."""
        arms_up = arms_up or {}
        raised = {pid for pid, held in arms_up.items() if held >= self.gesture_s}
        active = {pid for pid, p in players.items() if p.is_active}

        if self.phase is Phase.START:
            if raised & active:
                self.go(Phase.LOBBY, now)
        elif self.phase is Phase.LOBBY:
            self.ready = (self.ready | (raised & active)) & active
            if len(active) >= self.cfg.num_players and self.ready >= active:
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
            if self.time_in_phase(now) >= 3.0 and raised & active:
                self.go(Phase.LOBBY, now)
            elif self.time_in_phase(now) >= self.cfg.results_s:
                self.go(Phase.START, now)

    # ------------------------------------------------------------------ for the renderer
    def lobby_progress(self, now: float) -> float:
        if self._ready_since is None:
            return 0.0
        return min(1.0, (now - self._ready_since) / self.cfg.lobby_confirm_s)

    def countdown_remaining(self, now: float) -> float:
        return max(0.0, self.cfg.countdown_s - self.time_in_phase(now))
