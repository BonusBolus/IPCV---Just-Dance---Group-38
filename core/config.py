"""All tunable settings in one place. Command-line flags in main.py override some of these."""
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
class ModelConfig:
    inference_width: int = 640   # frames are downscaled to this width before CV inference
    pose_model: Path = ASSETS / "models" / "pose_landmarker.task"
    face_model: Path = ASSETS / "models" / "face_landmarker.task"
    camera_hfov_deg: float = 65.0  # horizontal field of view, for pixel -> metre conversion


@dataclass
class GameConfig:
    num_players: int = 2
    lobby_confirm_s: float = 1.5     # all players must be visible this long before the countdown
    countdown_s: float = 3.0
    results_s: float = 20.0          # results screen returns to start after this
    lost_warning_s: float = 0.5      # show "come back" warning after a player is lost this long
    camera_latency_s: float = 0.10   # webcam delay compensation when comparing to the song
    enable_audio: bool = True
    song_path: Path = ASSETS / "song.mp3"
    choreography_path: Path = ASSETS / "choreography.json"
    placeholder_song_s: float = 60.0  # game length when there is no song file


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    game: GameConfig = field(default_factory=GameConfig)
    player_names: tuple[str, ...] = ("Player 1", "Player 2", "Player 3", "Player 4")
    player_colors: tuple[tuple[int, int, int], ...] = (  # BGR
        (255, 150, 0),   # blue
        (60, 60, 255),   # red
        (0, 200, 0),     # green
        (0, 200, 255),   # yellow
    )
