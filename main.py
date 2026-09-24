"""Just Dance, IPCV Group 38: entry point.

Pipeline, once per camera frame:

    Camera
      -> PoseEstimator (T2), FaceTracker (T1)          raw, anonymous detections
      -> PlayerTracker (T3)                            who is who
      -> PoseSmoother (T2), FaceSmoother (T1),
         SpatialEstimator (T3)                         smooth per-player signals
      -> PhaseController (T5)                          game flow
      -> Scorer, MoveDetector, InteractionDetector (T4) -> GameState
      -> Renderer (T5)                                 final frame

Every module call goes through ModuleGuard, so a failing module degrades the game instead of
crashing it. Run `python main.py --help` for options.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from core.camera import Camera
from core.config import Config
from core.guard import ModuleGuard
from core.mock_source import MockPoseSource
from core.profiler import Profiler
from core.types import FrameData, GameEvent, Player
from face.face_effects import FaceEffects
from face.face_filter import FaceSmoother
from face.face_tracker import FaceTracker
from gameplay.action_effects import ActionEffects
from gameplay.choreography import Choreography
from gameplay.game_state import GameState
from gameplay.interactions import InteractionDetector
from gameplay.move_detector import MoveDetector
from gameplay.scorer import Scorer
from identity.player_tracker import PlayerTracker
from identity.spatial import SpatialEstimator
from pose.keypoint_filter import PoseSmoother
from pose.pose_estimator import PoseEstimator
from scene.audio import SongPlayer
from scene.background import SceneBackground
from scene.renderer import RenderContext, Renderer
from scene.state_machine import Phase, PhaseController

log = logging.getLogger("main")

CONTROLS = (
    ("SPACE", "start / play again"),
    ("N", "force next phase (debug)"),
    ("R", "back to start screen"),
    ("M", "toggle mock players (synthetic dancers)"),
    ("C", "toggle mock scenario dance / cross"),
    ("D", "toggle debug overlay"),
    ("F", "toggle fullscreen"),
    ("Q / ESC", "quit"),
)


class App:
    def __init__(self, cfg: Config, use_mock: bool = False):
        self.cfg = cfg
        self.use_mock = use_mock
        self.debug = cfg.display.show_debug
        self.running = True
        self.camera_fps = 0.0
        self.guard = ModuleGuard()
        self.profiler = Profiler()

        # Task 1 / 2 / 3: perception
        self.pose_estimator = PoseEstimator(cfg)
        self.face_tracker = FaceTracker(cfg)
        self.pose_smoother = PoseSmoother(cfg)
        self.face_smoother = FaceSmoother(cfg)
        self.tracker = PlayerTracker(cfg)
        self.spatial = SpatialEstimator(cfg)
        self.mock = MockPoseSource(num_players=cfg.game.num_players)

        # Task 4: gameplay
        self.game_state = GameState()
        self.scorer = Scorer(cfg)
        self.move_detector = MoveDetector(cfg)
        self.interactions = InteractionDetector(cfg)
        self.action_effects = ActionEffects(cfg)

        # Task 5: scene & integration
        song_path = cfg.game.song_path if cfg.game.enable_audio else None
        self.song = SongPlayer(song_path, fallback_duration=cfg.game.placeholder_song_s)
        self.choreo = Choreography.load_or_placeholder(cfg.game.choreography_path, self.song.duration)
        if not self.song.has_audio:
            self.song.duration = self.choreo.duration
        self.phases = PhaseController(cfg.game)
        self.background = SceneBackground(cfg)
        self.renderer = Renderer(cfg, self.background, FaceEffects(cfg), self.action_effects,
                                 self.guard, self.profiler)
        self._register_phase_hooks()

    def _register_phase_hooks(self) -> None:
        self.phases.on_enter(Phase.START, self.song.stop)
        self.phases.on_enter(Phase.COUNTDOWN, self._reset_round)
        self.phases.on_enter(Phase.PLAYING, self.song.play)
        self.phases.on_enter(Phase.RESULTS, self.song.stop)

    def _reset_round(self) -> None:
        pids = [p.pid for p in self.tracker.active_players()] or list(self.tracker.players)
        self.game_state.reset(pids)
        for module in (self.scorer, self.move_detector, self.interactions):
            module.reset()
        self.action_effects.clear()

    # ------------------------------------------------------------------ per-frame pipeline
    def process_frame(self, frame: FrameData, now: float | None = None):
        """Run the whole pipeline on one frame and return the canvas to display."""
        now = time.perf_counter() if now is None else now
        g, prof = self.guard, self.profiler
        prof.tick()

        if self.use_mock:
            poses, faces = self.mock.generate(now, frame.image.shape)
        else:
            with prof.section("pose"):
                poses = g.call("pose", self.pose_estimator.process, frame, default=[])
            with prof.section("face"):
                faces = g.call("face", self.face_tracker.process, frame, poses, default=[])

        with prof.section("identity"):
            players = g.call("identity", self.tracker.update, poses, faces, now, default=self.tracker.players)
        with prof.section("smoothing"):
            self._smooth_players(players, frame, now)

        song_t = self.song.time() - self.cfg.game.camera_latency_s
        self.phases.update(players, now, song_finished=self.song.finished)

        if self.phases.phase is Phase.PLAYING:
            with prof.section("gameplay"):
                events: list[GameEvent] = []
                events += g.call("scorer", self.scorer.update, players, song_t, self.choreo, default=[])
                events += g.call("moves", self.move_detector.update, players, song_t, default=[])
                events += g.call("interact", self.interactions.update, players, song_t, self.choreo, default=[])
                self.game_state.apply(events)
                self.action_effects.add_events(events, players, now)
                g.call("scene_events", self.background.on_events, events, now)

        ctx = RenderContext(frame, players, self.phases, self.game_state, self.choreo, song_t, now,
                            debug=self.debug, debug_lines=self._debug_lines(frame, song_t))
        with prof.section("render"):
            canvas = g.call("render", self.renderer.render, ctx, default=None)
        return canvas if canvas is not None else frame.image

    def _smooth_players(self, players: dict[int, Player], frame: FrameData, now: float) -> None:
        g = self.guard
        for pid, p in players.items():
            if not p.is_active:
                continue
            if p.pose is not None:
                p.pose = g.call("pose_filter", self.pose_smoother.update, pid, p.pose, now, default=p.pose)
            if p.face is not None:
                p.face = g.call("face_filter", self.face_smoother.update, pid, p.face, now, default=p.face)
            p.position_m = g.call("spatial", self.spatial.estimate, p, frame.image.shape, default=None)

    def _debug_lines(self, frame: FrameData, song_t: float) -> list[str]:
        h, w = frame.image.shape[:2]
        lines = [
            f"camera {w}x{h} @ {self.camera_fps:4.1f} fps",
            f"phase {self.phases.phase.name}   song t {song_t:6.2f}/{self.song.duration:.0f} s"
            + ("" if self.song.has_audio else " (silent)"),
        ]
        if self.use_mock:
            lines.append(f"MOCK PLAYERS ON ({self.mock.scenario})")
        if self.choreo.is_placeholder:
            lines.append("placeholder choreography")
        return lines

    # ------------------------------------------------------------------ input
    def handle_key(self, key: int, window: str | None = None) -> None:
        now = time.perf_counter()
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or ch == "q":
            self.running = False
        elif ch == " ":
            if self.phases.phase in (Phase.START, Phase.RESULTS):
                self.phases.go(Phase.LOBBY if self.phases.phase is Phase.START else Phase.START, now)
        elif ch == "n":
            self.phases.advance(now)
        elif ch == "r":
            self.phases.go(Phase.START, now)
        elif ch == "m":
            self.use_mock = not self.use_mock
            log.info("Mock players %s", "on" if self.use_mock else "off")
        elif ch == "c":
            self.mock.scenario = "cross" if self.mock.scenario == "dance" else "dance"
        elif ch == "d":
            self.debug = not self.debug
        elif ch == "f" and window is not None:
            self.cfg.display.fullscreen = not self.cfg.display.fullscreen
            self._apply_fullscreen(window)

    def _apply_fullscreen(self, window: str) -> None:
        mode = cv2.WINDOW_FULLSCREEN if self.cfg.display.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, mode)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        win = self.cfg.display.window_name
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.cfg.display.width, self.cfg.display.width * 9 // 16)
        self._apply_fullscreen(win)
        canvas = None
        try:
            with Camera(self.cfg.camera) as cam:
                while self.running:
                    frame = cam.read(timeout=1.0)
                    if frame is not None:
                        self.camera_fps = cam.fps
                        canvas = self.process_frame(frame)
                    else:
                        canvas = self.renderer.no_signal(canvas, cam.error)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(1)
                    if key != -1:
                        self.handle_key(key & 0xFF, win)
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break  # window closed with the mouse
        finally:
            self.close()
            cv2.destroyAllWindows()

    def close(self) -> None:
        self.song.close()
        self.pose_estimator.close()
        self.face_tracker.close()
        self.profiler.close()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Just Dance, IPCV Group 38",
        epilog="Keys: " + ", ".join(f"{k} = {v}" for k, v in CONTROLS),
    )
    p.add_argument("--camera", default="0", help="webcam index (default 0)")
    p.add_argument("--video", help="replay a recorded video file instead of the webcam")
    p.add_argument("--width", type=int, help="requested camera width")
    p.add_argument("--height", type=int, help="requested camera height")
    p.add_argument("--mock", action="store_true", help="start with synthetic mock players")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    p.add_argument("--no-audio", action="store_true", help="run without sound")
    p.add_argument("--no-debug", action="store_true", help="hide the debug overlay")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--profile-csv", help="write per-stage timings to this CSV file")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = Config()
    cfg.camera.source = args.video if args.video else args.camera
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    cfg.camera.mirror = not args.no_mirror
    cfg.game.enable_audio = not args.no_audio
    cfg.display.show_debug = not args.no_debug
    cfg.display.fullscreen = args.fullscreen

    app = App(cfg, use_mock=args.mock)
    if args.profile_csv:
        app.profiler.open_csv(args.profile_csv)
    try:
        app.run()
    except RuntimeError as exc:  # camera could not be opened
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
