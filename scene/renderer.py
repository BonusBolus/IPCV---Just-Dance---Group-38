"""Task 5: composes all layers into the final frame.

Layer order (bottom -> top): background/camera -> scene fx -> face effects (T1) ->
action effects (T4) -> model dancer panel -> HUD -> debug overlay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import Config
from core.guard import ModuleGuard
from core.image_utils import (
    GOLD, WHITE, darken, draw_bar, draw_skeleton, fill_rect_alpha, put_text, resize_to_width,
)
from core.profiler import Profiler
from core.types import FrameData, Player, TrackState
from face.face_effects import FaceEffects
from gameplay.action_effects import GRADE_COLORS, ActionEffects
from gameplay.choreography import Choreography
from gameplay.game_state import GameState
from scene.background import SceneBackground
from scene.state_machine import Phase, PhaseController


@dataclass
class RenderContext:
    """Everything the renderer needs for one frame."""
    frame: FrameData
    players: dict[int, Player]
    phases: PhaseController
    game_state: GameState
    choreo: Choreography
    song_t: float
    now: float
    debug: bool = False
    debug_lines: list[str] = field(default_factory=list)


class Renderer:
    def __init__(self, cfg: Config, background: SceneBackground, face_effects: FaceEffects,
                 action_effects: ActionEffects, guard: ModuleGuard, profiler: Profiler):
        self.cfg = cfg
        self.background = background
        self.face_effects = face_effects
        self.action_effects = action_effects
        self.guard = guard
        self.profiler = profiler

    # ------------------------------------------------------------------ entry point
    def render(self, ctx: RenderContext) -> np.ndarray:
        cam = resize_to_width(ctx.frame.image, self.cfg.display.width, cv2.INTER_LINEAR)
        phase = ctx.phases.phase

        if phase is Phase.START:
            canvas = self.background.stage(cam.shape[1], cam.shape[0]).copy()
            self._start_screen(canvas, cam, ctx)
        else:
            canvas = self._scene(cam, ctx)
            if ctx.debug:
                self._draw_skeletons(canvas, ctx.players)
            if phase is Phase.RESULTS:
                canvas = darken(canvas, 0.35)
                self._results_screen(canvas, ctx)
            else:
                self.guard.call("face_fx", self.face_effects.draw, canvas, ctx.players, ctx.game_state)
                if phase is Phase.LOBBY:
                    self._lobby_screen(canvas, ctx)
                elif phase is Phase.COUNTDOWN:
                    self._countdown_screen(canvas, ctx)
                elif phase is Phase.PLAYING:
                    self._playing_screen(canvas, ctx)

        if ctx.debug:
            self._debug_overlay(canvas, ctx)
        return canvas

    def no_signal(self, last_canvas: np.ndarray | None, error: str | None) -> np.ndarray:
        if last_canvas is None:
            w = self.cfg.display.width
            canvas = np.zeros((w * 9 // 16, w, 3), np.uint8)
        else:
            canvas = darken(last_canvas, 0.3)
        h, w = canvas.shape[:2]
        put_text(canvas, "No camera signal", (w // 2, h // 2), 1.4, (80, 80, 255), 3, anchor="center")
        if error:
            put_text(canvas, error, (w // 2, h // 2 + 50), 0.7, anchor="center")
        return canvas

    # ------------------------------------------------------------------ layers
    def _scene(self, cam: np.ndarray, ctx: RenderContext) -> np.ndarray:
        poses = [p.pose for p in ctx.players.values() if p.is_active and p.pose is not None]
        mask = self.guard.call("segmentation", self.background.person_mask, ctx.frame, poses, default=None)
        canvas = self.guard.call("compose", self.background.compose, cam, mask, default=cam)
        if canvas is cam:
            canvas = cam.copy()
        self.guard.call("scene_fx", self.background.draw_fx, canvas, ctx.now, ctx.game_state)
        return canvas

    def _draw_skeletons(self, canvas: np.ndarray, players: dict[int, Player]) -> None:
        for p in players.values():
            if p.pose is None:
                continue
            color = p.color if p.is_active else (128, 128, 128)
            draw_skeleton(canvas, p.pose.keypoints, p.pose.confidence, color, thickness=2)

    def _start_screen(self, canvas: np.ndarray, cam: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        pulse = 1.0 + 0.04 * math.sin(ctx.now * 4)
        put_text(canvas, "JUST DANCE", (w // 2, int(h * 0.38)), 3.2 * pulse, GOLD, 6, anchor="center")
        put_text(canvas, "IPCV Group 38 - webcam edition", (w // 2, int(h * 0.48)), 0.9, anchor="center")
        blink = int(ctx.now * 2) % 2 == 0
        if blink:
            put_text(canvas, "Press SPACE to start", (w // 2, int(h * 0.62)), 1.1, anchor="center")
        # small camera preview so players can check they are in frame
        pw = w // 4
        preview = resize_to_width(cam, pw)
        ph = preview.shape[0]
        x0, y0 = w - pw - 20, h - ph - 20
        canvas[y0:y0 + ph, x0:x0 + pw] = preview
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), WHITE, 1)

    def _lobby_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        need = self.cfg.game.num_players
        n = sum(p.is_active for p in ctx.players.values())
        put_text(canvas, f"Waiting for players: {n}/{need}", (w // 2, 60), 1.2, anchor="center", bg=(0, 0, 0))
        put_text(canvas, "Stand ~2 m from the camera, upper body fully visible",
                 (w // 2, h - 30), 0.7, anchor="center", bg=(0, 0, 0))
        progress = ctx.phases.lobby_progress(ctx.now)
        if progress > 0:
            put_text(canvas, "Get ready!", (w // 2, 120), 1.0, GOLD, anchor="center")
            draw_bar(canvas, (w // 2 - 150, 140, 300, 14), progress, GOLD)

    def _countdown_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        n = math.ceil(ctx.phases.countdown_remaining(ctx.now))
        put_text(canvas, str(max(n, 1)), (w // 2, h // 2 + 60), 6.0, GOLD, 10, anchor="center")

    def _playing_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        self.guard.call("action_fx", self.action_effects.draw, canvas, ctx.players, ctx.now)
        self._model_panel(canvas, ctx)
        self._score_boxes(canvas, ctx)

        move, nxt = ctx.choreo.move_at(ctx.song_t), ctx.choreo.next_move(ctx.song_t)
        if move is not None:
            label = move.move.name.replace("_", " ") + ("  (GOLD)" if move.gold else "") + ("  (DUET)" if move.duet else "")
            put_text(canvas, label, (w // 2, h - 60), 1.2, GOLD if move.gold else WHITE, anchor="center", bg=(0, 0, 0))
        elif nxt is not None:
            put_text(canvas, f"next: {nxt.move.name.replace('_', ' ')}", (w // 2, h - 60), 0.8,
                     anchor="center", bg=(0, 0, 0))
        draw_bar(canvas, (20, h - 24, w - 40, 10), ctx.choreo.progress(ctx.song_t), (255, 0, 200))

        for p in ctx.players.values():
            if p.state is TrackState.LOST and ctx.now - p.last_seen > self.cfg.game.lost_warning_s:
                put_text(canvas, f"{p.name}: come back into view!", (w // 2, h // 2), 1.0, p.color,
                         anchor="center", bg=(0, 0, 0))

    def _model_panel(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        """The dancer to copy. TODO(T5/T4): nicer look (glow, silhouette, trail)."""
        h, w = canvas.shape[:2]
        pw, ph = int(w * 0.24), int(h * 0.62)
        rect = (w - pw - 20, int(h * 0.16), pw, ph)
        fill_rect_alpha(canvas, rect, (20, 0, 30), 0.55)
        put_text(canvas, "FOLLOW ME", (rect[0] + pw // 2, rect[1] + 28), 0.7, GOLD, anchor="center")
        ref = ctx.choreo.reference_at(ctx.song_t)
        if ref is not None:
            draw_skeleton(canvas, ref.keypoints, ref.confidence, GOLD, thickness=4, rect=rect)

    def _score_boxes(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        pids = sorted(ctx.game_state.stats) or sorted(ctx.players)
        box_w = 250
        for i, pid in enumerate(pids):
            player = ctx.players.get(pid)
            stats = ctx.game_state.stats_for(pid)
            color = player.color if player else WHITE
            name = player.name if player else f"Player {pid}"
            x = 20 + i * (box_w + 20)
            fill_rect_alpha(canvas, (x, 16, box_w, 70), color, 0.55)
            put_text(canvas, name, (x + 10, 42), 0.7)
            put_text(canvas, f"{stats.score:>6}", (x + 10, 76), 1.0, WHITE, 2)
            if stats.last_grade is not None:
                put_text(canvas, stats.last_grade.name, (x + box_w - 10, 76), 0.6,
                         GRADE_COLORS[stats.last_grade], anchor="right")

    def _results_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        winner = ctx.game_state.winner()
        if winner is None:
            put_text(canvas, "IT'S A DRAW!", (w // 2, int(h * 0.25)), 2.4, GOLD, 5, anchor="center")
        else:
            p = ctx.players.get(winner)
            name = p.name if p else f"Player {winner}"
            put_text(canvas, f"{name.upper()} WINS!", (w // 2, int(h * 0.25)), 2.4,
                     p.color if p else GOLD, 5, anchor="center")

        y = int(h * 0.40)
        for pid in sorted(ctx.game_state.stats):
            s = ctx.game_state.stats[pid]
            p = ctx.players.get(pid)
            grades = "  ".join(f"{g.name[:4]} {s.grades[g]}" for g in GRADE_COLORS)
            line = f"{p.name if p else pid}:  {s.score} pts   max combo {s.max_combo}   {grades}"
            put_text(canvas, line, (w // 2, y), 0.8, p.color if p else WHITE, anchor="center")
            y += 50
        put_text(canvas, "Press SPACE to play again", (w // 2, int(h * 0.85)), 0.9, anchor="center")

    def _debug_overlay(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        lines = self.profiler.summary() + ctx.debug_lines
        errors = [f"! {name} failed x{n}: {self.guard.last_error.get(name, '')[:60]}"
                  for name, n in self.guard.failures.items()]
        h = canvas.shape[0]
        y = h - 40 - 22 * (len(lines) + len(errors))
        fill_rect_alpha(canvas, (10, y - 22, 460, 22 * (len(lines) + len(errors)) + 12), (0, 0, 0), 0.5)
        for line in lines:
            put_text(canvas, line, (18, y), 0.5, (200, 255, 200), 1, outline=False)
            y += 22
        for line in errors:
            put_text(canvas, line, (18, y), 0.5, (80, 80, 255), 1, outline=False)
            y += 22
