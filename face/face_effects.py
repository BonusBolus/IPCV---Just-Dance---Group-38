"""Task 1: augmented face effects and the per-player tag shown above the head."""
from __future__ import annotations

import cv2
import numpy as np

from core.config import Config
from core.image_utils import put_text, to_px
from core.types import Player
from gameplay.game_state import GameState, PlayerStats


class FaceEffects:
    """Draws, for every active player:
      1. a face effect aligned with the head (it has to mean something in the game,
         e.g. a crown for the leader, a mask whose glow shows the combo streak)
      2. an extra non-face visual: name tag, score/combo bar, buffs...

    TODO(T1): load stickers here, e.g. cv2.imread("assets/effects/crown.png", cv2.IMREAD_UNCHANGED),
    and draw them with core.image_utils.overlay_rgba (scale from face size, rotate by face roll).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def draw(self, canvas: np.ndarray, players: dict[int, Player], game_state: GameState) -> None:
        leader = game_state.leader()
        for p in players.values():
            if not p.is_active:
                continue
            stats = game_state.stats.get(p.pid)
            if p.face is not None:
                self.draw_face_effect(canvas, p, stats, is_leader=(p.pid == leader))
            self.draw_player_tag(canvas, p, stats)

    def draw_face_effect(self, canvas: np.ndarray, player: Player,
                         stats: PlayerStats | None, is_leader: bool) -> None:
        """TODO(T1): the real effect. Placeholder: outline the face box in the player's colour."""
        x, y, w, h = player.face.bbox
        p0 = to_px((x, y), canvas.shape)
        p1 = to_px((x + w, y + h), canvas.shape)
        cv2.rectangle(canvas, p0, p1, player.color, 2, cv2.LINE_AA)

    def draw_player_tag(self, canvas: np.ndarray, player: Player, stats: PlayerStats | None) -> None:
        """TODO(T1): make this a proper game-state visual. Placeholder: name + combo."""
        anchor = player.head_top()
        if anchor is None:
            return
        x, y = to_px(anchor, canvas.shape)
        label = player.name if stats is None or stats.combo < 2 else f"{player.name}  x{stats.combo}"
        put_text(canvas, label, (x, y - 14), 0.6, anchor="center", bg=player.color, bg_alpha=0.8)
