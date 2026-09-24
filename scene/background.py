"""Task 5: stage background, background replacement and scene reactions to gameplay.

Background replacement: the per-person segmentation masks from the pose model (one per
detected body, at inference resolution) are merged with a pixel-wise max, smoothed over time
(EMA, against flicker), given a soft edge (contrast stretch + Gaussian blur) and used to
composite the players onto a procedural disco stage. Masks come for free with the pose model,
so no second segmentation network is needed.

Scene reactions:
  - spotlights sweeping on the beat, flashing on every beat (brighter on the downbeat)
  - the stage glows in the colour of the leading player; the stronger the lead, the brighter
  - PERFECT -> short white flash; MISS -> red pulse at the edges
  - gold moves, duets, high fives, swaps -> confetti burst at the player / contact point
All effects are drawn on low-resolution layers or cached full-resolution layers and added
with a single saturating cv2.add, which keeps the cost at a few ms per frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from core.config import Config
from core.image_utils import to_px, vertical_gradient
from core.types import EventType, FrameData, GameEvent, Grade, Player, PoseObs
from gameplay.game_state import GameState

BEAM_COLORS = ((255, 0, 200), (255, 200, 0), (0, 200, 255), (120, 255, 0))
CONFETTI = ((0, 215, 255), (255, 0, 200), (255, 200, 0), (60, 60, 255), (0, 220, 0), (255, 255, 255))


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    t0: float
    life: float


class SceneBackground:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._stage_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._vignette: tuple[tuple[int, int], np.ndarray] | None = None
        self._tint_cache: dict[tuple, np.ndarray] = {}
        self._mask_ema: np.ndarray | None = None
        self._particles: list[_Particle] = []
        self._flash_t = -1.0
        self._miss_t = -1.0
        self._last_fx_t: float | None = None
        self._rng = np.random.default_rng(1)

    # ------------------------------------------------------------------ background
    def stage(self, w: int, h: int) -> np.ndarray:
        """Procedural disco stage (cached per size)."""
        if self._stage_cache is None or self._stage_cache[0] != (w, h):
            img = vertical_gradient(w, h, top=(90, 20, 70), bottom=(25, 10, 30)).copy()
            horizon, vanish = int(h * 0.62), (w // 2, int(h * 0.35))
            for i in range(-12, 13):  # floor lines towards the vanishing point
                x = w // 2 + i * w // 10
                p0 = (int(vanish[0] + (x - vanish[0]) * (horizon - vanish[1]) / (h - vanish[1])), horizon)
                cv2.line(img, p0, (x, h), (170, 60, 200), 1, cv2.LINE_AA)
            for k in range(1, 9):     # horizontal floor lines, denser towards the horizon
                y = int(horizon + (h - horizon) * (k / 8) ** 2)
                cv2.line(img, (0, y), (w, y), (170, 60, 200), 1, cv2.LINE_AA)
            self._stage_cache = ((w, h), img)
        return self._stage_cache[1]

    def person_mask(self, frame: FrameData, poses: list[PoseObs]) -> np.ndarray | None:
        """(h, w) float mask in [0, 1] of all players (1 = person), or None = no replacement."""
        masks = [p.mask for p in poses if p.mask is not None]
        if not masks:
            self._mask_ema = None
            return None
        m = np.maximum.reduce([mk.astype(np.float32) for mk in masks]) if len(masks) > 1 else masks[0].astype(np.float32)
        if self._mask_ema is not None and self._mask_ema.shape == m.shape:
            m = 0.5 * self._mask_ema + 0.5 * m
        self._mask_ema = m
        m = np.clip((m - 0.3) / 0.4, 0.0, 1.0)          # sharpen the soft model output a bit
        return cv2.GaussianBlur(m, (5, 5), 0)          # ...and feather the edge

    def compose(self, image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Put the players (mask) in front of the stage background."""
        if mask is None:
            return image
        h, w = image.shape[:2]
        m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        # one C call instead of numpy float math on the full frame (measured 34 ms -> ~3 ms at 1280x960)
        return cv2.blendLinear(image, self.stage(w, h), m, 1.0 - m)

    # ------------------------------------------------------------------ reactions
    def on_events(self, events: list[GameEvent], players: dict[int, Player], now: float) -> None:
        for ev in events:
            if ev.type is EventType.GRADE:
                if ev.grade is Grade.PERFECT:
                    self._flash_t = now
                elif ev.grade is Grade.MISS:
                    self._miss_t = now
            points = ev.data.get("points", 0)
            if points and ev.type in (EventType.INTERACTION, EventType.MOVE_DETECTED):
                player = players.get(ev.pid)
                pos = ev.data.get("pos") or (player.head_top() if player else None) or (0.5, 0.4)
                self._burst(pos, now, n=60 if ev.type is EventType.INTERACTION else 35)

    def _burst(self, pos, now: float, n: int) -> None:
        for _ in range(n):
            a = self._rng.uniform(0, 2 * math.pi)
            s = self._rng.uniform(0.2, 0.7)
            self._particles.append(_Particle(pos[0], pos[1], s * math.cos(a) * 0.6, s * math.sin(a) - 0.3,
                                             CONFETTI[self._rng.integers(len(CONFETTI))], now,
                                             self._rng.uniform(0.8, 1.5)))
        del self._particles[:-400]  # cap

    def draw_fx(self, canvas: np.ndarray, now: float, game_state: GameState,
                song_t: float | None = None, bpm: float = 120.0, playing: bool = False) -> None:
        h, w = canvas.shape[:2]
        if playing and song_t is not None:
            cv2.add(canvas, self._lights(w, h, song_t, bpm), dst=canvas)
            leader = game_state.leader()
            if leader is not None:
                strength = min(1.0, abs(game_state.balance()) * 3)
                color = self.cfg.player_colors[(leader - 1) % len(self.cfg.player_colors)]
                cv2.add(canvas, self._tint(w, h, color, round(strength * 10) / 10 * 0.5), dst=canvas)
        miss_age = now - self._miss_t
        if 0 <= miss_age < 0.4:
            cv2.add(canvas, self._tint(w, h, (40, 40, 255), round((1 - miss_age / 0.4) * 10) / 10 * 0.7), dst=canvas)
        flash_age = now - self._flash_t
        if 0 <= flash_age < 0.15:
            a = 0.25 * (1 - flash_age / 0.15)
            cv2.convertScaleAbs(canvas, dst=canvas, alpha=1 - a, beta=255 * a)
        self._draw_particles(canvas, now)

    def _lights(self, w: int, h: int, song_t: float, bpm: float) -> np.ndarray:
        """Four spotlights on a 1/8-resolution layer, upscaled (the blur makes them soft anyway)."""
        lw, lh = max(8, w // 8), max(8, h // 8)
        layer = np.zeros((lh, lw, 3), np.uint8)
        beat = 60.0 / bpm
        phase = (song_t % beat) / beat
        downbeat = int(song_t / beat) % 4 == 0
        intensity = 0.25 + (0.75 if downbeat else 0.45) * math.exp(-4 * phase)
        bar = int(song_t / (4 * beat))
        for i, x in enumerate((0.15, 0.38, 0.62, 0.85)):
            ang = math.radians(28 * math.sin(song_t * 0.9 + i * 1.7))
            ox, length, spread = x * lw, 1.3 * lh, math.radians(9)
            p1 = (ox + length * math.sin(ang - spread), length * math.cos(ang - spread))
            p2 = (ox + length * math.sin(ang + spread), length * math.cos(ang + spread))
            color = BEAM_COLORS[(i + bar) % len(BEAM_COLORS)]
            pts = np.array([(ox, -2), p1, p2], np.int32)
            cv2.fillConvexPoly(layer, pts, tuple(int(c * 0.35 * intensity) for c in color), cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (5, 5), 0)
        return cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR)

    def _tint(self, w: int, h: int, color, strength: float) -> np.ndarray:
        """Coloured edge glow (vignette), cached per colour and strength step."""
        key = (w, h, tuple(color), strength)
        if key not in self._tint_cache:
            if self._vignette is None or self._vignette[0] != (w, h):
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
                self._vignette = ((w, h), np.clip((d - 0.55) / 0.8, 0, 1) ** 1.5)
            v = self._vignette[1][..., None]
            if len(self._tint_cache) > 64:
                self._tint_cache.clear()
            self._tint_cache[key] = (v * np.array(color, np.float32) * strength).astype(np.uint8)
        return self._tint_cache[key]

    def _draw_particles(self, canvas: np.ndarray, now: float) -> None:
        dt = 0.0 if self._last_fx_t is None else min(0.1, now - self._last_fx_t)
        self._last_fx_t = now
        alive = []
        for p in self._particles:
            if now - p.t0 > p.life:
                continue
            p.vy += 0.9 * dt   # gravity (normalized units / s^2)
            p.x += p.vx * dt
            p.y += p.vy * dt
            x, y = to_px((p.x, p.y), canvas.shape)
            cv2.rectangle(canvas, (x - 3, y - 3), (x + 3, y + 3), p.color, -1)
            alive.append(p)
        self._particles = alive

    def reset(self) -> None:
        self._particles.clear()
        self._mask_ema = None
