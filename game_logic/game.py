"""Game: owns the shared parts and runs the main loop.

Each frame:  camera -> pose -> identity (players) -> state.update() -> head effect -> state.draw() -> screen
Face tracking runs in a second thread, at the same time as the pose (YOLO). Both only read the
camera image, and both spend most time outside Python, so this saves about 7 ms per frame.
The states (game_logic/states/, one file per state) decide what happens. This file only connects the parts.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from camera import FPSCounter
from face_tracking.face_tracking import get_face_properties, smooth_face_properties
from face_tracking.head_filter import enlarge_heads
from background.segmentation import PersonSegmenter, PlayerMasker
from background.stage import Stage
from face_tracking.game_effects import add_crown, add_player_label, faces_for_players
from game_logic.pose_grading import PoseGrader
from game_logic.song import find_songs
from game_logic.states.countdown import CountdownState
from game_logic.states.playing import PlayingState
from game_logic.states.registration import RegistrationState
from game_logic.states.results import ResultsState
from game_logic.states.song_select import SongSelectState
from identity_tracking.identity import IdentityTracker, draw_labels, people_from_pose
from pose_tracking.pose_processing import PoseEstimator
from scene.functions import player_color
from scene.scene import Scene
import cv2
import config

WINDOW_NAME = "Just Dance"


def find_faces(image):
    """Face tracking for one image. Runs in the face thread (see Game.step)."""
    return smooth_face_properties(get_face_properties(image))


def crown_holder(players):
    """Id of the player who leads by at least config.CROWN_LEAD (10%), or None.
    Nobody gets the crown while both scores are 0 (for example during Registration)."""
    (id_1, player_1), (id_2, player_2) = sorted(players.items())
    if player_1["score"] > player_2["score"]:
        return id_1
    if player_2["score"] > player_1["score"]:
        return id_2
    return None


# Only used by the debug key "n": jump to the next state
NEXT_STATE = {"song_select": "registration", "registration": "countdown", "countdown": "playing",
              "playing": "results", "results": "song_select"}


class Game:
    def __init__(self, camera, debug=False, fullscreen=True):
        self.camera = camera
        self.debug = debug
        self.fullscreen = fullscreen
        self.pose_estimator = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)
        self.tracker = IdentityTracker(max_players=2)
        self.pose_grader = PoseGrader()
        self.songs = find_songs()                      # all songs, for the song selection
        first = [i for i, song in enumerate(self.songs) if song.path.resolve() == config.SONG_PATH.resolve()]
        self.song_index = first[0] if first else 0
        self.song = self.songs[self.song_index]        # the selected song
        for song in self.songs:
            ungraded = sorted({move.key for move in song.moves} - set(self.pose_grader.processed_poses))
            if ungraded:
                print(f"Warning: {song.path.name}: the PoseGrader has no pose {', '.join(ungraded)}, these moves give no points")
        self.scene = Scene(color_key=config.PLAYER_COLOR)
        self.fps = FPSCounter()
        self.face_thread = ThreadPoolExecutor(max_workers=1)   # one worker: frames stay in order
        self.step_thread = ThreadPoolExecutor(max_workers=1)   # calculates the next frame, see run()
        # the Just Dance stage instead of the room: person segmentation in its own thread, like the faces
        self.segmenter = PersonSegmenter() if config.BACKGROUND else None
        self.segment_thread = ThreadPoolExecutor(max_workers=1) if config.BACKGROUND else None
        self.stage = Stage()
        self.masker = PlayerMasker()                           # which people stay on the stage
        self.skip_state = False                                # debug key n: next state, done in step()
        self.states = {
            "song_select": SongSelectState(),
            "registration": RegistrationState(),
            "countdown": CountdownState(),
            "playing": PlayingState(),
            "results": ResultsState(),
        }
        self.players = self.tracker.entries   # {1: {...}, 2: {...}}, see identity.new_player_entry()
        self.now = 0.0                        # time of the current frame (seconds)
        self.heads = {}                       # player id -> face of that player, or None
        self.state_name = None
        self.state = None

    def change_state(self, name):
        if self.state is not None:
            self.state.exit(self)
        self.state_name = name
        self.state = self.states[name]
        self.state.enter(self)

    def step(self, frame):
        """Everything that happens with one camera frame. Returns the image to show."""
        self.now = frame.time
        if self.skip_state:                            # debug key n (set in run(), done here between frames)
            self.skip_state = False
            self.change_state(NEXT_STATE[self.state_name])
        if self.state is None:
            self.change_state("song_select")

        faces_future = self.face_thread.submit(find_faces, frame.image)    # starts now, in the second thread
        people_future = self.segment_thread.submit(self.segmenter.find_people, frame.image) if self.segmenter else None
        keypoints, _ = self.pose_estimator.process(frame.image)
        self.players = self.tracker.update(frame.image, people_from_pose(keypoints, min_confidence=0.1))

        next_state = self.state.update(self)
        if next_state:
            self.change_state(next_state)

        image = frame.image.copy()
        faces = faces_future.result()                  # wait for the face thread (usually done already)
        # Only faces that belong to a player get effects: no crowns on bystanders or false detections
        self.heads = faces_for_players(self.players, faces)
        player_faces = [face for face in self.heads.values() if face is not None]
        # only the players stay, the room becomes the stage (mask: 1 = player, 0 = room)
        # (other people are only removed while the players are fixed, see PlayerMasker)
        players_fixed = self.state_name in ("countdown", "playing", "results")
        mask = (self.masker.mask(people_future.result(), self.players.values(), image.shape, self.now, players_fixed)
                if people_future else None)
        # the big heads come from the sharpest camera image (1080p: Frame.detail); frame.image is not drawn on
        enlarge_heads(image, player_faces, config.HEAD_ENLARGEMENT, detail=getattr(frame, "detail", frame.image),
                      mask=mask)
        if mask is not None:
            self.stage.place_players(image, mask, self.now)
        leader = crown_holder(self.players)
        if leader is not None and self.heads[leader] is not None:
            add_crown(image, [self.heads[leader]])
        # The player label sits on the crown band (on the forehead when there is no crown)
        for player_id, face in self.heads.items():
            if face is not None:
                add_player_label(image, f"Player {player_id}", face, player_color(self.players[player_id]))
        self.scene.draw_edge_effects(image, list(self.players.values()), self.now)   # glow + sparkles, every state
        if self.debug:
            draw_labels(image, self.players)
            self.scene.draw_player(image, self.players)
            t = self.song.time()
            current_move = self.song.current_move(t) or self.song.grading_move(t) or self.song.next_move(t)
            if current_move is not None:
                self.scene.draw_target_pose_overlay(image, self.players, current_move.pose)

        image = self.state.draw(self, image)
        if self.debug:
            cv2.putText(image, f"FPS: {self.fps.fps:.1f}  state: {self.state_name}  (n = next state)",
                        (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def run(self):
        # A window without the toolbar and status bar: showing an image is about 2.5 ms faster.
        # The window can be made bigger, the image keeps its shape (black bars at the sides if needed).
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        window_sized = self.fullscreen                 # a full-screen window already has its size
        frame_time = 1.0 / config.MAX_FPS
        # The next frame is calculated in a second thread while the window shows the current one.
        # Painting a full-screen window takes about 25 ms (it is scaled up in software), and this way
        # that time is no longer added to every frame. The game itself still runs one frame at a
        # time: everything happens in step(), and step() never runs twice at the same time.
        frame = self.camera.read()
        pending = self.step_thread.submit(self.step, frame) if frame is not None else None
        while pending is not None:
            frame_start = time.perf_counter()
            image = pending.result()                   # the frame that was calculated meanwhile
            frame = self.camera.read()
            pending = self.step_thread.submit(self.step, frame) if frame is not None else None
            self.fps.update()
            if not window_sized:                       # start with the size of the camera image
                cv2.resizeWindow(WINDOW_NAME, image.shape[1], image.shape[0])
                window_sized = True
            cv2.imshow(WINDOW_NAME, image)

            # Wait for the rest of this frame's time, so the game never runs faster than MAX_FPS.
            # waitKey does the waiting, so the window stays responsive (and q / ESC still work).
            wait_ms = int((frame_time - (time.perf_counter() - frame_start)) * 1000)
            key = cv2.waitKey(max(1, wait_ms)) & 0xFF
            if key in (ord("q"), 27):          # safety exit, not a game control
                break
            if self.debug and key == ord("n"):
                self.skip_state = True                 # step() switches the state before the next frame
        if pending is not None:
            pending.result()                           # let the frame that is still running finish

        self.song.stop()
        self.step_thread.shutdown()
        self.face_thread.shutdown()
        if self.segment_thread:
            self.segment_thread.shutdown()
        self.camera.release()
        cv2.destroyAllWindows()
