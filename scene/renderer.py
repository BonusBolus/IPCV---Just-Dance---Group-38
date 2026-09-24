"""Task 5: composes all layers into the final frame.

Layer order (bottom -> top): stage + players (background replacement) -> scene fx ->
wrist trails / popups (T4) -> face effects + tags (T1) -> model dancer panel -> HUD -> debug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.config import Config
from core.guard import ModuleGuard
from core.image_utils import (
    GOLD, WHITE, darken, draw_bar, draw_skeleton, fill_rect_alpha, put_text, resize_to_width, to_px,
)
from core.profiler import Profiler
from core.types import KP, FrameData, MoveType, Player, TrackState
from face.face_effects import FaceEffects
from gameplay.action_effects import GRADE_COLORS, ActionEffects
from gameplay.choreography import Choreography
from gameplay.game_state import GameState
from scene.background import SceneBackground
from scene.state_machine import Phase, PhaseController

INSTRUCTIONS = {
    MoveType.HIGH_FIVE: "HIGH FIVE your partner!",
    MoveType.SWAP: "SWAP PLACES!",
}


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
    arms_up: dict[int, float] = field(default_factory=dict)


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
                if phase is Phase.PLAYING:
                    seg = ctx.choreo.move_at(ctx.song_t)
                    self.guard.call("action_fx", self.action_effects.draw, canvas, ctx.players, ctx.now,
                                    gold=bool(seg and seg.gold))
                self.guard.call("face_fx", self.face_effects.draw, canvas, ctx.players, ctx.game_state,
                                ctx.now, ctx.song_t)
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
        playing = ctx.phases.phase in (Phase.PLAYING, Phase.COUNTDOWN)
        self.guard.call("scene_fx", self.background.draw_fx, canvas, ctx.now, ctx.game_state,
                        ctx.song_t, ctx.choreo.bpm, playing)
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
        put_text(canvas, "JUST DANCE", (w // 2, int(h * 0.36)), 3.2 * pulse, GOLD, 6, anchor="center")
        put_text(canvas, "IPCV Group 38 - webcam edition", (w // 2, int(h * 0.46)), 0.9, anchor="center")
        if int(ctx.now * 2) % 2 == 0:
            put_text(canvas, "Raise both arms to start", (w // 2, int(h * 0.60)), 1.1, anchor="center")
        put_text(canvas, "(or press SPACE)", (w // 2, int(h * 0.60) + 36), 0.6, anchor="center")
        held = max(ctx.arms_up.values(), default=0.0)
        if held > 0:
            draw_bar(canvas, (w // 2 - 150, int(h * 0.68), 300, 12), held / ctx.phases.gesture_s, GOLD)
        # small camera preview so players can check that they are in view
        pw = w // 4
        preview = resize_to_width(cam, pw)
        ph = preview.shape[0]
        x0, y0 = w - pw - 20, h - ph - 20
        canvas[y0:y0 + ph, x0:x0 + pw] = preview
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), WHITE, 1)

    def _lobby_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        need = self.cfg.game.num_players
        active = [p for p in ctx.players.values() if p.is_active]
        put_text(canvas, f"Players: {len(active)}/{need}", (w // 2, 50), 1.2, anchor="center", bg=(0, 0, 0))
        msg = ("Raise both arms when you are ready!" if len(active) >= need
               else "Step into view, ~2 m from the camera, upper body visible")
        put_text(canvas, msg, (w // 2, h - 30), 0.8, anchor="center", bg=(0, 0, 0))
        for p in active:
            anchor = p.head_top()
            if anchor is None:
                continue
            x, y = to_px(anchor, canvas.shape)
            if p.pid in ctx.phases.ready:
                put_text(canvas, "READY!", (x, max(30, y - 110)), 1.0, (0, 220, 0), 3, anchor="center")
            else:
                held = ctx.arms_up.get(p.pid, 0.0)
                if held > 0:
                    draw_bar(canvas, (x - 50, max(20, y - 120), 100, 10), held / ctx.phases.gesture_s, p.color)
        progress = ctx.phases.lobby_progress(ctx.now)
        if progress > 0:
            put_text(canvas, "Get ready!", (w // 2, 110), 1.0, GOLD, anchor="center")
            draw_bar(canvas, (w // 2 - 150, 125, 300, 14), progress, GOLD)

    def _countdown_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        n = math.ceil(ctx.phases.countdown_remaining(ctx.now))
        put_text(canvas, str(max(n, 1)), (w // 2, h // 2 + 60), 6.0, GOLD, 10, anchor="center")
        self._model_panel(canvas, ctx)

    def _playing_screen(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        h, w = canvas.shape[:2]
        self._model_panel(canvas, ctx)
        self._score_boxes(canvas, ctx)
        self._tug_of_war(canvas, ctx)

        move, nxt = ctx.choreo.move_at(ctx.song_t), ctx.choreo.next_move(ctx.song_t)
        if move is not None:
            label = INSTRUCTIONS.get(move.move, move.move.label)
            tags = ("  GOLD MOVE" if move.gold else "") + ("  DUET" if move.duet else "")
            color = GOLD if move.gold or move.move.is_interaction else WHITE
            put_text(canvas, label + tags, (w // 2, h - 52), 1.2, color, anchor="center", bg=(0, 0, 0))
        elif nxt is not None:
            put_text(canvas, f"next: {INSTRUCTIONS.get(nxt.move, nxt.move.label)}", (w // 2, h - 52), 0.8,
                     anchor="center", bg=(0, 0, 0))
        draw_bar(canvas, (20, h - 22, w - 40, 10), ctx.choreo.progress(ctx.song_t), (255, 0, 200))

        y = h // 2
        for p in ctx.players.values():
            if p.state is TrackState.LOST and ctx.now - p.last_seen > self.cfg.game.lost_warning_s:
                put_text(canvas, f"{p.name}: come back into view!", (w // 2, y), 1.0, p.color,
                         anchor="center", bg=(0, 0, 0))
                y += 45

    def _model_panel(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        """The dancer to copy, letterboxed to the reference aspect ratio, with a glow.

        Centred between the players (they stand left and right of the middle, like in Just
        Dance) and semi-transparent, so it never hides a player completely.
        """
        h, w = canvas.shape[:2]
        pw, ph = int(w * 0.20), int(h * 0.58)
        px, py = (w - pw) // 2, int(h * 0.12)
        fill_rect_alpha(canvas, (px, py, pw, ph), (20, 0, 30), 0.45)
        put_text(canvas, "FOLLOW ME", (px + pw // 2, py + 26), 0.7, GOLD, anchor="center")
        ref = ctx.choreo.reference_at(max(0.0, ctx.song_t))
        if ref is None:
            return
        inner_h = ph - 40
        dw = min(pw, int(inner_h * ctx.choreo.aspect))
        dh = int(dw / ctx.choreo.aspect)
        rect = (px + (pw - dw) // 2, py + 36 + (inner_h - dh) // 2, dw, dh)
        draw_skeleton(canvas, ref.keypoints, ref.confidence, (60, 120, 160), thickness=9, rect=rect)  # glow
        draw_skeleton(canvas, ref.keypoints, ref.confidence, GOLD, thickness=4, rect=rect)
        nx, ny = ref.keypoints[KP["nose"]]
        cv2.circle(canvas, (int(rect[0] + nx * dw), int(rect[1] + ny * dh)), max(6, dh // 18), GOLD, -1, cv2.LINE_AA)
        nxt = ctx.choreo.next_move(ctx.song_t)
        if nxt is not None and nxt.start - ctx.song_t < 1.5:
            put_text(canvas, f"next: {nxt.move.label}", (px + pw // 2, py + ph - 10), 0.55, WHITE, 1, anchor="center")

    def _score_boxes(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        pids = sorted(ctx.game_state.stats) or sorted(ctx.players)
        box_w = 250
        for i, pid in enumerate(pids):
            player = ctx.players.get(pid)
            stats = ctx.game_state.stats_for(pid)
            color = player.color if player else WHITE
            name = player.name if player else f"Player {pid}"
            x = 20 + i * (box_w + 20) if i == 0 else canvas.shape[1] - (box_w + 20) * (len(pids) - i)
            fill_rect_alpha(canvas, (x, 16, box_w, 70), color, 0.55)
            put_text(canvas, name, (x + 10, 40), 0.65)
            put_text(canvas, f"{stats.score:>6}", (x + 10, 76), 1.0, WHITE, 2)
            if stats.combo >= 2:
                put_text(canvas, f"combo x{stats.combo}", (x + box_w - 10, 40), 0.5, GOLD, 1, anchor="right")
            if stats.last_grade is not None:
                put_text(canvas, stats.last_grade.name, (x + box_w - 10, 76), 0.6,
                         GRADE_COLORS[stats.last_grade], anchor="right")

    def _tug_of_war(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        """Shared bar between the scores: who is winning, and by how much."""
        if len(ctx.game_state.stats) < 2:
            return
        w = canvas.shape[1]
        a, b = sorted(ctx.game_state.stats)[:2]
        ca = ctx.players[a].color if a in ctx.players else WHITE
        cb = ctx.players[b].color if b in ctx.players else WHITE
        bw, x0, y0 = 360, w // 2 - 180, 30
        split = int(bw * (1 - ctx.game_state.balance(a, b)) / 2)  # balance -1 = everything to a
        cv2.rectangle(canvas, (x0, y0), (x0 + split, y0 + 22), ca, -1)
        cv2.rectangle(canvas, (x0 + split, y0), (x0 + bw, y0 + 22), cb, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + 22), WHITE, 2)
        cv2.line(canvas, (w // 2, y0 - 4), (w // 2, y0 + 26), WHITE, 1)

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
            line = f"{p.name if p else pid}:  {s.score} pts   max combo {s.max_combo}   bonus {s.bonuses}   {grades}"
            put_text(canvas, line, (w // 2, y), 0.75, p.color if p else WHITE, anchor="center")
            y += 50
        if ctx.phases.time_in_phase(ctx.now) >= 3.0:
            put_text(canvas, "Raise both arms (or SPACE) to play again", (w // 2, int(h * 0.85)), 0.9,
                     anchor="center")

    def _debug_overlay(self, canvas: np.ndarray, ctx: RenderContext) -> None:
        lines = self.profiler.summary() + ctx.debug_lines
        errors = [f"! {name} failed x{n}: {self.guard.last_error.get(name, '')[:60]}"
                  for name, n in self.guard.failures.items()]
        h = canvas.shape[0]
        y = h - 40 - 20 * (len(lines) + len(errors))
        fill_rect_alpha(canvas, (10, y - 20, 430, 20 * (len(lines) + len(errors)) + 10), (0, 0, 0), 0.5)
        for line in lines:
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
            y += 20
        for line in errors:
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)
            y += 20
