"""Game: owns the shared parts and runs the main loop.

Each frame:  camera -> pose -> identity (players) -> state.update() -> head effect -> state.draw() -> screen
The states (game_logic/states/, one file per state) decide what happens. This file only connects the parts.
"""
import cv2

import config
from camera import FPSCounter
from face_tracking.face_overlay import draw_face_outline
from face_tracking.face_tracking import get_face_properties, smooth_face_properties
from face_tracking.head_filter import enlarge_heads
from game_logic.pose_grading import PoseGrader
from game_logic.song import Song
from game_logic.states.countdown import CountdownState
from game_logic.states.playing import PlayingState
from game_logic.states.registration import RegistrationState
from game_logic.states.results import ResultsState
from identity_tracking.identity import IdentityTracker, draw_labels, people_from_pose
from pose_tracking.pose_processing import PoseEstimator
from scene.scene import Scene

WINDOW_NAME = "Just Dance"

# Only used by the debug key "n": jump to the next state
NEXT_STATE = {"registration": "countdown", "countdown": "playing",
              "playing": "results", "results": "registration"}


class Game:
    def __init__(self, camera, debug=False):
        self.camera = camera
        self.debug = debug
        self.pose_estimator = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)
        self.tracker = IdentityTracker(max_players=2)
        self.pose_grader = PoseGrader()
        self.song = Song(config.SONG_PATH)
        ungraded = sorted({move.key for move in self.song.moves} - set(self.pose_grader.processed_poses))
        if ungraded:
            print(f"Warning: the PoseGrader has no pose {', '.join(ungraded)}, these moves give no points")
        self.scene = Scene(color_key=config.PLAYER_COLOR)
        self.fps = FPSCounter()
        self.states = {
            "registration": RegistrationState(),
            "countdown": CountdownState(),
            "playing": PlayingState(),
            "results": ResultsState(),
        }
        self.players = self.tracker.entries   # {1: {...}, 2: {...}}, see identity.new_player_entry()
        self.now = 0.0                        # time of the current frame (seconds)
        self.state_name = None
        self.state = None

    def change_state(self, name):
        self.state_name = name
        self.state = self.states[name]
        self.state.enter(self)

    def step(self, frame):
        """Everything that happens with one camera frame. Returns the image to show."""
        self.now = frame.time
        if self.state is None:
            self.change_state("registration")

        keypoints, _ = self.pose_estimator.process(frame.image)
        self.players = self.tracker.update(frame.image, people_from_pose(keypoints, min_confidence=0.1))

        next_state = self.state.update(self)
        if next_state:
            self.change_state(next_state)

        image = frame.image.copy()
        faces = smooth_face_properties(get_face_properties(image))
        enlarge_heads(image, faces, config.HEAD_ENLARGEMENT)
        self.scene.draw_edge_effects(image, list(self.players.values()), self.now)   # glow + sparkles, every state
        if self.debug:
            draw_labels(image, self.players)
            draw_face_outline(image, faces)

        image = self.state.draw(self, image)
        if self.debug:
            cv2.putText(image, f"FPS: {self.fps.fps:.1f}  state: {self.state_name}  (n = next state)",
                        (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def run(self):
        while True:
            frame = self.camera.read()
            if frame is None:
                break

            image = self.step(frame)
            self.fps.update()
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # safety exit, not a game control
                break
            if self.debug and key == ord("n"):
                self.change_state(NEXT_STATE[self.state_name])

        self.song.stop()
        self.camera.release()
        cv2.destroyAllWindows()
