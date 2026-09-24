"""All tunable settings in one place. Command-line flags in main.py override some of these.

Every threshold the report has to justify lives here, grouped per task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


@dataclass
class CameraConfig:
    source: int | str = 0        # webcam index, or a video file path for replay mode
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True          # selfie view: players see themselves like in a mirror
    loop_video: bool = True      # replay mode: restart the file when it ends


@dataclass
class DisplayConfig:
    window_name: str = "Just Dance - IPCV Group 38"
    width: int = 1280            # canvas width; height follows the camera aspect ratio
    fullscreen: bool = False
    show_debug: bool = True


@dataclass
class FaceConfig:  # Task 1
    model: Path = ASSETS / "models" / "face_landmarker.task"
    crop_size: int = 192             # head crops are resized to this before the face model
    crop_scale: float = 2.2          # crop side = crop_scale * ear-to-ear distance
    every_n_frames: int = 1          # run the face model every N frames (filter bridges the gap)
    hold_s: float = 0.35             # keep a face this long after a missed detection (fading)
    min_cutoff: float = 1.0          # One Euro filter on the face box
    beta: float = 10.0


@dataclass
class PoseConfig:  # Task 2
    model: Path = ASSETS / "models" / "pose_landmarker_full.task"
    inference_width: int = 640       # frames are downscaled to this width before inference
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5
    segmentation: bool = True        # person masks for background replacement (Task 5)
    min_keypoint_conf: float = 0.5   # below this a keypoint counts as unreliable
    hold_s: float = 0.3              # hold an unreliable keypoint this long, then drop it
    min_cutoff: float = 1.0          # One Euro filter on keypoints (normalized units); tuned with
    beta: float = 30.0               # eval/eval_pose.py --mock (min error vs. ground truth)
    min_body_height: float = 0.12    # ignore detections smaller than this (spectators far away)


@dataclass
class TrackingConfig:  # Task 3
    max_cost: float = 1.0            # assignment cost above this = no match
    w_position: float = 1.0
    w_appearance: float = 0.8
    w_size: float = 0.4
    w_shape: float = 0.6            # limb positions vs. last frame (separates overlapping bodies)
    position_gate: float = 0.25      # normalized distance that costs 1.0
    reentry_max_appearance: float = 0.45  # Bhattacharyya distance for re-identification
    appearance_alpha: float = 0.05   # running-average rate of the appearance model
    lost_predict_s: float = 1.0      # keep predicting a lost track's position this long
    camera_hfov_deg: float = 65.0    # horizontal field of view, for pixel -> metre conversion
    shoulder_width_m: float = 0.38
    torso_length_m: float = 0.50


@dataclass
class GameplayConfig:  # Task 4
    timing_tolerance_s: float = 0.35     # humans lag the model; compare within +- this window
    score_percentile: float = 75.0       # segment score = this percentile of frame similarities
    gold_bonus: int = 50
    duet_bonus: int = 80
    high_five_bonus: int = 100
    swap_bonus: int = 150
    high_five_distance_m: float = 0.30
    move_confirm_s: float = 0.2          # a move must be held this long before it counts
    move_cooldown_s: float = 1.0
    start_gesture_s: float = 0.8         # hold both arms up this long to start / get ready


@dataclass
class GameConfig:  # Task 5
    num_players: int = 2
    lobby_confirm_s: float = 1.0     # after everybody is ready, wait this long, then count down
    countdown_s: float = 3.0
    results_s: float = 25.0          # results screen returns to start after this
    lost_warning_s: float = 0.5      # show "come back" warning after a player is lost this long
    camera_latency_s: float = 0.10   # webcam delay compensation when comparing to the song
    enable_audio: bool = True
    song_path: Path = ASSETS / "song.wav"
    choreography_path: Path = ASSETS / "choreography.json"
    bpm: float = 120.0               # used when generating the default song + choreography
    default_song_s: float = 64.0


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gameplay: GameplayConfig = field(default_factory=GameplayConfig)
    game: GameConfig = field(default_factory=GameConfig)
    player_names: tuple[str, ...] = ("Player 1", "Player 2", "Player 3", "Player 4")
    player_colors: tuple[tuple[int, int, int], ...] = (  # BGR
        (255, 150, 0),   # blue
        (60, 60, 255),   # red
        (0, 200, 0),     # green
        (0, 200, 255),   # yellow
    )
