"""Just Dance, IPCV Group 38: entry point.

Pipeline, once per camera frame:

    Camera (own thread, newest frame only)
      -> PoseEstimator (T2), FaceTracker (T1, head crops from the poses)   anonymous detections
      -> PlayerTracker (T3)                                                who is who
      -> PoseSmoother (T2), FaceSmoother (T1), SpatialEstimator (T3)       per-player signals
      -> MoveDetector (T4)                          gestures (menus) and moves (gold bonus)
      -> PhaseController (T5)                       game flow
      -> Scorer, InteractionDetector (T4) -> GameState
      -> Renderer (T5)                              final frame

Every module call goes through ModuleGuard, so a failing module degrades the game instead of
crashing it. Run `python main.py --help` for options.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from core.camera import Camera
from core.config import Config
from core.guard import ModuleGuard
from core.mock_source import MockPoseSource
from core.profiler import Profiler
from core.types import EventType, FrameData, GameEvent, MoveType, Player
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
from tools.make_default_assets import ensure_default_assets

log = logging.getLogger("main")

CONTROLS = (
    ("arms up", "start / ready / play again (hold both arms up)"),
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
    def __init__(self, cfg: Config, use_mock: bool = False, load_models: bool = True):
        self.cfg = cfg
        self.use_mock = use_mock
        self.debug = cfg.display.show_debug
        self.running = True
        self.camera_fps = 0.0
        self.guard = ModuleGuard()
        self.profiler = Profiler()
        ensure_default_assets(cfg.game)

        # Task 1 / 2 / 3: perception
        self.pose_estimator = PoseEstimator(cfg) if load_models else None
        self.face_tracker = FaceTracker(cfg) if load_models else None
        self.pose_smoother = PoseSmoother(cfg)
        self.face_smoother = FaceSmoother(cfg)
        self.tracker = PlayerTracker(cfg)
        self.spatial = SpatialEstimator(cfg)
        self.mock = MockPoseSource(num_players=cfg.game.num_players)

        # Task 4: gameplay
        self.game_state = GameState()
        self.scorer = Scorer(cfg)
        self.move_detector = MoveDetector(cfg)
        self.interactions = InteractionDetector(cfg, self.spatial)
        self.action_effects = ActionEffects(cfg)

        # Task 5: scene & integration
        song_path = cfg.game.song_path if cfg.game.enable_audio else None
        self.choreo = Choreography.load_or_placeholder(cfg.game.choreography_path, cfg.game.default_song_s)
        self.song = SongPlayer(song_path, fallback_duration=self.choreo.duration)
        self.phases = PhaseController(cfg.game, cfg.gameplay)
        self.background = SceneBackground(cfg)
        self.renderer = Renderer(cfg, self.background, FaceEffects(cfg), self.action_effects,
                                 self.guard, self.profiler)
        self._register_phase_hooks()

    def _register_phase_hooks(self) -> None:
        self.phases.on_enter(Phase.START, self.song.stop)
        self.phases.on_enter(Phase.LOBBY, self._new_game)
        self.phases.on_enter(Phase.COUNTDOWN, self._reset_round)
        self.phases.on_enter(Phase.PLAYING, self.song.play)
        self.phases.on_enter(Phase.RESULTS, self.song.stop)

    def _new_game(self) -> None:
        """Entering the lobby: forget old identities, so P1/P2 are numbered left to right again."""
        self.song.stop()
        self.tracker.reset()
        self.pose_smoother.reset()
        self.face_smoother.reset()
        self.spatial.reset()
        self.move_detector.reset()
        self.game_state.reset([])

    def _reset_round(self) -> None:
        pids = [p.pid for p in self.tracker.active_players()] or list(self.tracker.players)
        self.game_state.reset(pids)
        for module in (self.scorer, self.move_detector, self.interactions, self.background):
            module.reset()
        self.action_effects.clear()

    # ------------------------------------------------------------------ per-frame pipeline
    def process_frame(self, frame: FrameData, now: float | None = None):
        """Run the whole pipeline on one frame and return the canvas to display."""
        now = time.perf_counter() if now is None else now
        g, prof = self.guard, self.profiler
        prof.tick()

        if self.use_mock or self.pose_estimator is None:
            poses, faces = self.mock.generate(now, frame.image.shape) if self.use_mock else ([], [])
        else:
            with prof.section("pose"):
                poses = g.call("pose", self.pose_estimator.process, frame, default=[])
            with prof.section("face"):
                faces = g.call("face", self.face_tracker.process, frame, poses, default=[])

        with prof.section("identity"):
            players = g.call("identity", self.tracker.update, frame, poses, faces, now, default=self.tracker.players)
        with prof.section("smoothing"):
            self._smooth_players(players, frame, now)

        song_t = self.song.time() - self.cfg.game.camera_latency_s
        playing = self.phases.phase is Phase.PLAYING
        with prof.section("gameplay"):
            events: list[GameEvent] = [GameEvent(kind, song_t, pid=pid) for kind, pid in self.tracker.pop_events()]
            moves = g.call("moves", self.move_detector.update, players, now, song_t,
                           self.choreo if playing else None, default=[])
            arms_up = {pid: self.move_detector.held_for(pid, MoveType.ARMS_UP, now) for pid in players}
            self.phases.update(players, now, song_finished=self.song.finished, arms_up=arms_up)

            if playing:
                grades = g.call("scorer", self.scorer.update, players, song_t, self.choreo, default=[])
                inter = g.call("interact", self.interactions.update, players, song_t, self.choreo, grades, default=[])
                events += moves + grades + inter
                for pid, sim in self.scorer.live_similarity.items():
                    if pid in self.game_state.stats:
                        self.game_state.stats[pid].live_match = sim
                self.game_state.apply(events)
                g.call("scene_events", self.background.on_events, events, players, now)
            self.action_effects.add_events(events, players, now)
            self.action_effects.update(players, now)

        ctx = RenderContext(frame, players, self.phases, self.game_state, self.choreo, song_t, now,
                            debug=self.debug, debug_lines=self._debug_lines(frame, song_t), arms_up=arms_up)
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
            # also called without a detection: the smoother holds the face briefly (fade-out)
            p.face = g.call("face_filter", self.face_smoother.update, pid, p.face, now, default=p.face)
            p.position_m = g.call("spatial", self.spatial.estimate, p, frame.image.shape, default=None)

    def _debug_lines(self, frame: FrameData, song_t: float) -> list[str]:
        h, w = frame.image.shape[:2]
        lines = [
            f"camera {w}x{h} @ {self.camera_fps:4.1f} fps",
            f"phase {self.phases.phase.name}   song {song_t:6.2f}/{self.song.duration:.0f} s"
            + ("" if self.song.has_audio else " (silent)"),
        ]
        for p in self.tracker.players.values():
            pos = f"x {p.position_m[0]:+.2f} m, z {p.position_m[1]:.2f} m" if p.position_m else "-"
            sim = self.scorer.live_similarity.get(p.pid)
            lines.append(f"P{p.pid} {p.state.name:<6} {pos}  match {sim if sim is None else round(sim, 2)}"
                         f"  move {self.move_detector.current(p.pid).name}")
        if self.use_mock:
            lines.append(f"MOCK PLAYERS ON ({self.mock.scenario})")
        for module in (self.pose_estimator, self.face_tracker):
            if module is not None and module.error:
                lines.append(f"! {module.error}")
        return lines

    # ------------------------------------------------------------------ input
    def handle_key(self, key: int, window: str | None = None) -> None:
        now = time.perf_counter()
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or ch == "q":
            self.running = False
        elif ch == " ":
            if self.phases.phase is Phase.START:
                self.phases.go(Phase.LOBBY, now)
            elif self.phases.phase is Phase.RESULTS:
                self.phases.go(Phase.LOBBY, now)
        elif ch == "n":
            self.phases.advance(now)
        elif ch == "r":
            self.phases.go(Phase.START, now)
        elif ch == "m":
            self.use_mock = not self.use_mock
            self.tracker.reset()
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
        for module in (self.pose_estimator, self.face_tracker):
            if module is not None:
                module.close()
        self.profiler.close()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Just Dance, IPCV Group 38",
        epilog="Controls: " + ", ".join(f"{k} = {v}" for k, v in CONTROLS),
    )
    p.add_argument("--camera", default="0", help="webcam index (default 0)")
    p.add_argument("--video", help="replay a recorded video file instead of the webcam")
    p.add_argument("--width", type=int, help="requested camera width")
    p.add_argument("--height", type=int, help="requested camera height")
    p.add_argument("--song", help="audio file to play (default: generated assets/song.wav)")
    p.add_argument("--choreography", help="choreography JSON (default: generated assets/choreography.json)")
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
    if args.song:
        cfg.game.song_path = Path(args.song)
    if args.choreography:
        cfg.game.choreography_path = Path(args.choreography)
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
