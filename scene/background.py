"""Task 5: stage background, background replacement and scene reactions to gameplay."""
from __future__ import annotations

import cv2
import numpy as np

from core.config import Config
from core.image_utils import vertical_gradient
from core.types import FrameData, GameEvent, PoseObs
from gameplay.game_state import GameState


class SceneBackground:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._stage_cache: tuple[tuple[int, int], np.ndarray] | None = None

    def stage(self, w: int, h: int) -> np.ndarray:
        """Procedural disco stage (cached per size). TODO(T5): replace/extend with a nicer scene."""
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
        """TODO(T5): (h, w) float mask in [0, 1] of all players (1 = person).

        E.g. combine the PoseObs.mask of each player (MediaPipe PoseLandmarker can output
        segmentation masks) or run a separate selfie-segmentation model. Consider running it at
        low resolution and smoothing the mask edge / over time. None = show the raw camera image.
        """
        return None

    def compose(self, image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Put the players (mask) in front of the stage background."""
        if mask is None:
            return image
        h, w = image.shape[:2]
        m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        bg = self.stage(w, h)
        return (image * m + bg * (1.0 - m)).astype(np.uint8)

    def on_events(self, events: list[GameEvent], now: float) -> None:
        """TODO(T5): react to Task 4's events (flash on PERFECT, confetti on gold moves, ...)."""

    def draw_fx(self, canvas: np.ndarray, now: float, game_state: GameState) -> None:
        """TODO(T5): environment effects every frame: lights pulsing on the beat, stage tint
        towards the leading player's colour, particles."""
