"""Task 1: augmented face effects and the per-player tag shown above the head.

The effects show the game state on the face, so players read their performance off each
other's heads:
  - sunglasses, affine-warped onto both eye corners + the nose tip (follow turns and tilts);
    lens style shows the combo: player colour -> orange "on fire" (combo >= 3) -> gold (>= 6)
  - star eyes for 0.6 s after a PERFECT
  - a crown on the current leader, placed above the forehead along the face's up-direction
Extra non-face visual (required): name tag, combo counter and a live "groove meter" bar showing
how well the player matches the model right now.
Effects fade out with the face confidence during short dropouts (see face_filter.py) and are
drawn per pid, so they cannot jump to the other player unless Task 3 swaps identities.
"""
from __future__ import annotations

import numpy as np

from core.config import Config
from core.image_utils import GOLD, draw_bar, overlay_rgba, put_text, to_px, warp_rgba_affine
from core.types import FaceObs, Grade, Player
from face import stickers
from face.face_tracker import CHIN, EYE_OUTER, FOREHEAD, NOSE_TIP
from gameplay.game_state import GameState, PlayerStats

PERFECT_FLASH_S = 0.6
FIRE = (0, 120, 255)


def face_anchor_points(face: FaceObs, shape) -> dict[str, np.ndarray]:
    """Pixel positions of the points effects attach to. Uses landmarks when available,
    otherwise proportions of the face box (e.g. mock faces)."""
    h, w = shape[:2]
    scale = np.array([w, h], np.float32)
    if face.landmarks is not None and len(face.landmarks) > CHIN:
        lm = face.landmarks
        a, b = lm[EYE_OUTER[0]] * scale, lm[EYE_OUTER[1]] * scale
        left, right = (a, b) if a[0] <= b[0] else (b, a)
        return {"eye_l": left, "eye_r": right, "nose": lm[NOSE_TIP] * scale,
                "forehead": lm[FOREHEAD] * scale, "chin": lm[CHIN] * scale}
    x, y, bw, bh = face.bbox
    p = lambda fx, fy: np.array([(x + fx * bw) * w, (y + fy * bh) * h], np.float32)  # noqa: E731
    return {"eye_l": p(0.18, 0.42), "eye_r": p(0.82, 0.42), "nose": p(0.5, 0.65),
            "forehead": p(0.5, 0.05), "chin": p(0.5, 1.0)}


class FaceEffects:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._glasses: dict[tuple, stickers.Sticker] = {}
        self._star = stickers.star()
        self._crown = stickers.crown()

    def _glasses_for(self, color, combo: int) -> stickers.Sticker:
        if combo >= 6:
            style = (GOLD, (40, 180, 255))
        elif combo >= 3:
            style = ((0, 60, 200), FIRE)
        else:
            style = (color, (30, 30, 30))
        if style not in self._glasses:
            self._glasses[style] = stickers.sunglasses(*style)
        return self._glasses[style]

    def draw(self, canvas: np.ndarray, players: dict[int, Player], game_state: GameState,
             now: float | None = None, song_t: float | None = None) -> None:
        leader = game_state.leader()
        for p in players.values():
            if not p.is_active:
                continue
            stats = game_state.stats.get(p.pid)
            if p.face is not None:
                self.draw_face_effect(canvas, p, stats, is_leader=(p.pid == leader), song_t=song_t)
            self.draw_player_tag(canvas, p, stats)

    def draw_face_effect(self, canvas: np.ndarray, player: Player, stats: PlayerStats | None,
                         is_leader: bool, song_t: float | None = None) -> None:
        face = player.face
        opacity = float(np.clip(face.confidence / 0.9, 0.0, 1.0))
        pts = face_anchor_points(face, canvas.shape)
        eye_span = float(np.linalg.norm(pts["eye_r"] - pts["eye_l"]))
        if eye_span < 4:
            return

        perfect = (stats is not None and stats.last_grade is Grade.PERFECT and song_t is not None
                   and song_t - stats.last_grade_t < PERFECT_FLASH_S)
        if perfect:
            span = pts["eye_r"] - pts["eye_l"]
            star_scale = 0.45 * eye_span / self._star.rgba.shape[1]
            for center in (pts["eye_l"] + 0.22 * span, pts["eye_r"] - 0.22 * span):  # eye centres
                overlay_rgba(canvas, self._star.rgba, tuple(center), scale=star_scale,
                             angle_deg=face.roll, opacity=opacity)
        else:
            g = self._glasses_for(player.color, stats.combo if stats else 0)
            warp_rgba_affine(canvas, g.rgba, g.anchors, [pts["eye_l"], pts["eye_r"], pts["nose"]], opacity)

        if is_leader:
            up = pts["forehead"] - pts["chin"]
            face_h = float(np.linalg.norm(up))
            if face_h > 1:
                up /= face_h
                center = pts["forehead"] + up * 0.32 * face_h
                c = self._crown.rgba
                overlay_rgba(canvas, c, tuple(center), scale=1.5 * eye_span / c.shape[1],
                             angle_deg=face.roll, opacity=opacity)

    def draw_player_tag(self, canvas: np.ndarray, player: Player, stats: PlayerStats | None) -> None:
        anchor = player.head_top()
        if anchor is None:
            return
        x, y = to_px(anchor, canvas.shape)
        y -= 60 if player.face is not None else 20  # leave room for the crown
        label = player.name if stats is None or stats.combo < 2 else f"{player.name}  x{stats.combo}"
        box = put_text(canvas, label, (x, y), 0.6, anchor="center", bg=player.color, bg_alpha=0.8)
        if stats is not None and stats.live_match is not None:
            m = stats.live_match
            color = (0, 200, 0) if m >= 0.65 else (0, 200, 255) if m >= 0.45 else (60, 60, 255)
            draw_bar(canvas, (box[0], box[1] + box[3] + 2, box[2], 7), m, color)
