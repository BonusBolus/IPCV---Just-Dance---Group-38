"""Song playback. The playback position is the master clock for choreography and scoring.

Without a song file (or without pygame / an audio device) the game still runs: time then
comes from time.perf_counter() and the song length from the choreography.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


class SongPlayer:
    def __init__(self, path: str | Path | None, fallback_duration: float):
        self.path = Path(path) if path else None
        self.duration = float(fallback_duration)
        self._mixer = None
        self._t0: float | None = None

        if self.path is None:
            log.info("Audio disabled")
        elif not self.path.exists():
            log.warning("Song not found at %s, running silently (timing still works)", self.path)
        else:
            self._init_pygame()

    def _init_pygame(self) -> None:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(str(self.path))
            try:
                self.duration = pygame.mixer.Sound(str(self.path)).get_length()
            except pygame.error:
                log.warning("Could not read song length, using %.0f s", self.duration)
            self._mixer = pygame.mixer
            log.info("Loaded song %s (%.1f s)", self.path.name, self.duration)
        except Exception as exc:  # noqa: BLE001 - missing pygame, no audio device, bad file...
            log.warning("Audio unavailable (%s), running silently", exc)
            self._mixer = None

    @property
    def has_audio(self) -> bool:
        return self._mixer is not None

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def play(self) -> None:
        self._t0 = time.perf_counter()
        if self._mixer is not None:
            self._mixer.music.play()

    def stop(self) -> None:
        if self._mixer is not None:
            self._mixer.music.stop()
        self._t0 = None

    def time(self) -> float:
        """Seconds since the song started (0 before it starts).

        TODO(T5): measure the drift between get_pos() and perf_counter() and the audio output
        latency; this determines how precisely moves can be scored.
        """
        if self._t0 is None:
            return 0.0
        if self._mixer is not None:
            pos_ms = self._mixer.music.get_pos()
            if pos_ms >= 0:
                return pos_ms / 1000.0
        return time.perf_counter() - self._t0

    @property
    def finished(self) -> bool:
        return self._t0 is not None and self.time() >= self.duration

    def close(self) -> None:
        if self._mixer is not None:
            self._mixer.music.stop()
            self._mixer.quit()
            self._mixer = None
