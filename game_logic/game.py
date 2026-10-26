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
from game_logic.measurements import NoMeasurements, timed
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
import numpy as np
import config

WINDOW_NAME = "Just Dance"


def find_faces(image):
    """Face tracking for one image. Runs in the face thread (see Game.step)."""
    return smooth_face_properties(get_face_properties(image))


def warm_up_models(pose_estimator, segmenter=None):
    """Prepare the YOLO models one after the other, before they run in their own threads.

    OpenVINO prepares (compiles) a model the first time it is used. When the pose model and the
    segmentation model do that at the same moment in two threads, the program can hang (seen in a test:
    all threads waiting for each other). After this, both are ready, and the first frame is not slow."""
    blank = np.zeros((config.GAME_SIZE[1], config.GAME_SIZE[0], 3), np.uint8)
    pose_estimator.model.predict(blank, verbose=False)
    if segmenter is not None:
        segmenter.find_people(blank)


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
    def __init__(self, camera, debug=False, fullscreen=True, measure=None, duration=None, recorder=None):
        self.camera = camera
        self.debug = debug
        self.show_debug = debug                        # debug drawings this frame: --debug, or see debug_shown()
        self.fullscreen = fullscreen
        self.measure = measure or NoMeasurements()    # --measure: times of every stage (measurements.py)
        self.duration = duration                       # --duration: stop after this many seconds
        self.recorder = recorder                       # --record: webcam and game as videos (recorder.py)
        self.song_deadline = None                      # --record: time (perf_counter) the song must stop
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
        warm_up_models(self.pose_estimator, self.segmenter)   # one after the other, see there
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
        if name == "playing" and self.recorder is not None and self.song.has_audio:
            self.recorder.song_started(self.song.audio_path)    # the music of the video starts here

    def step(self, frame):
        """Everything that happens with one camera frame. Returns the image to show."""
        self.now = frame.time
        if self.skip_state:                            # debug key n (set in run(), done here between frames)
            self.skip_state = False
            self.change_state(NEXT_STATE[self.state_name])
        if self.state is None:
            self.change_state("song_select")

        m = self.measure
        m.start_frame(frame, self.state_name)
        step_start = time.perf_counter()
        # face detection and segmentation start now, in their own threads (timed() also gives their time)
        faces_future = self.face_thread.submit(timed, find_faces, frame.image)
        people_future = (self.segment_thread.submit(timed, self.segmenter.find_people, frame.image)
                         if self.segmenter else None)
        with m.stage("t2_pose"):
            keypoints, _ = self.pose_estimator.process(frame.image)
        with m.stage("t3_identity"):
            self.players = self.tracker.update(frame.image, people_from_pose(keypoints, min_confidence=0.1))

        with m.stage("t4_logic"):
            next_state = self.state.update(self)
            if next_state:
                self.change_state(next_state)

        image = frame.image.copy()
        with m.stage("wait_threads"):
            faces, faces_ms = faces_future.result()    # wait for the face thread (usually done already)
        m.add("t1_faces", faces_ms)
        with m.stage("t3_identity"):
            # Only faces that belong to a player get effects: no crowns on bystanders or false detections
            self.heads = faces_for_players(self.players, faces)
        player_faces = [face for face in self.heads.values() if face is not None]
        m.count("people", len(keypoints))
        m.count("faces", len(faces))
        m.count("players_visible", sum(player["visible"] for player in self.players.values()))
        m.count("heads", len(player_faces))
        mask = None
        if people_future:
            with m.stage("wait_threads"):
                people, segment_ms = people_future.result()
            m.add("segmentation", segment_ms)
            with m.stage("t5_render"):
                # only the players stay, the room becomes the stage (mask: 1 = player, 0 = room)
                # (other people are only removed while the players are fixed, see PlayerMasker)
                players_fixed = self.state_name in ("countdown", "playing", "results")
                mask = self.masker.mask(people, self.players.values(), image.shape, self.now, players_fixed)
        with m.stage("t1_effects"):
            # the big heads come from the sharpest camera image (1080p: Frame.detail); frame.image is not drawn on
            enlarge_heads(image, player_faces, config.HEAD_ENLARGEMENT, detail=getattr(frame, "detail", frame.image),
                          mask=mask)
        with m.stage("t5_render"):
            if mask is not None:
                self.stage.place_players(image, mask, self.now)
        with m.stage("t1_effects"):
            leader = crown_holder(self.players)
            if leader is not None and self.heads[leader] is not None:
                add_crown(image, [self.heads[leader]])
            # The player label sits on the crown band (on the forehead when there is no crown)
            for player_id, face in self.heads.items():
                if face is not None:
                    add_player_label(image, f"Player {player_id}", face, player_color(self.players[player_id]))
        with m.stage("t5_render"):
            self.scene.draw_edge_effects(image, list(self.players.values()), self.now)   # glow + sparkles, every state
        self.show_debug = self.debug or self.debug_shown()
        with m.stage("t5_render"):
            image = self.state.draw(self, image)
        if self.show_debug:
            # on top of the state's screen, so they also show on the (darkened) results screen
            draw_labels(image, self.players)
            self.scene.draw_player(image, self.players)
            t = self.song.time()
            current_move = self.song.current_move(t) or self.song.grading_move(t) or self.song.next_move(t)
            if current_move is not None and self.state_name == "playing":
                self.scene.draw_target_pose_overlay(image, self.players, current_move.pose)
            cv2.putText(image, f"FPS: {self.fps.fps:.1f}  state: {self.state_name}  (n = next state)",
                        (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        m.add("step", (time.perf_counter() - step_start) * 1000)
        return image

    def debug_shown(self):
        """The debug drawings (skeletons, boxes, FPS) show by themselves on the results screen (DEBUG_IN_RESULTS),
        and from DEBUG_SHOW_AT to DEBUG_SHOW_AT + DEBUG_SHOW_FOR seconds into the song, so the demo video
        shows the tracking. The n key (next state) still only works with --debug."""
        if config.DEBUG_IN_RESULTS and self.state_name == "results":
            return True
        if config.DEBUG_SHOW_AT is None or self.state_name != "playing":
            return False
        return config.DEBUG_SHOW_AT <= self.song.time() < config.DEBUG_SHOW_AT + config.DEBUG_SHOW_FOR

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
        self.measure.game_ready = time.perf_counter()
        run_start = time.perf_counter()
        shown_size = None
        if self.recorder is not None and self.duration is not None:
            # the results fill the last RECORD_RESULTS_TIME of the video, however long the start took
            self.song_deadline = run_start + self.duration - config.RECORD_RESULTS_TIME
        frame = self.camera.read()
        pending = self.step_thread.submit(self.step, frame) if frame is not None else None
        while pending is not None:
            frame_start = time.perf_counter()
            image = pending.result()                   # the frame that was calculated meanwhile
            shown_index, shown_frame = frame.index, frame
            frame = self.camera.read()
            pending = self.step_thread.submit(self.step, frame) if frame is not None else None
            self.fps.update()
            if not window_sized:                       # start with the size of the camera image
                cv2.resizeWindow(WINDOW_NAME, image.shape[1], image.shape[0])
                window_sized = True
            display_start = time.perf_counter()
            shown_size = f"{image.shape[1]}x{image.shape[0]}"
            cv2.imshow(WINDOW_NAME, image)
            self.measure.frame_shown(shown_index, (time.perf_counter() - display_start) * 1000)
            if self.recorder is not None:
                self.recorder.add(shown_frame.image, image)    # the webcam and the game, the same moment
            if self.duration is not None and time.perf_counter() - run_start >= self.duration:
                break                                  # --duration: the measurement is long enough

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
        self.measure.finish(f"song: {self.song.title}, game image {shown_size}, "
                            f"camera image {config.CAMERA_SIZE[0]}x{config.CAMERA_SIZE[1]} asked")

        self.song.stop()
        self.step_thread.shutdown()
        self.face_thread.shutdown()
        if self.segment_thread:
            self.segment_thread.shutdown()
        self.camera.release()
        cv2.destroyAllWindows()
        if self.recorder is not None:
            self.recorder.finish()                     # makes the MP4 files (some seconds)
