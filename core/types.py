"""Shared data contracts between the five tasks.

Every module talks to the others only through these types. Changing a field here affects
everyone, so discuss it with the group (and the Task 5 integrator) first.

Conventions
-----------
- Images are BGR uint8 numpy arrays (OpenCV convention), already mirrored when mirroring is on.
- Image coordinates are *normalized* to [0, 1] relative to the camera frame (x to the right,
  y downwards). They stay valid whatever resolution a module runs at or the renderer draws at.
- Body keypoints use the COCO-17 layout below, whichever pose backend Task 2 picks.
- `timestamp` = time.perf_counter() at capture. "Song time" = seconds since the song started
  (see scene/audio.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

# --------------------------------------------------------------------------- keypoints (COCO-17)
KP_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP = {name: i for i, name in enumerate(KP_NAMES)}
NUM_KEYPOINTS = len(KP_NAMES)

SKELETON_EDGES = (
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]), (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]), (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]), (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]), (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]), (KP["right_knee"], KP["right_ankle"]),
    (KP["nose"], KP["left_eye"]), (KP["nose"], KP["right_eye"]),
    (KP["left_eye"], KP["left_ear"]), (KP["right_eye"], KP["right_ear"]),
)

# Index pairs that swap when a pose is mirrored horizontally.
MIRROR_PAIRS = tuple(
    (KP[n], KP[n.replace("left_", "right_")]) for n in KP_NAMES if n.startswith("left_")
)


# --------------------------------------------------------------------------- perception
@dataclass
class FrameData:
    image: np.ndarray   # (H, W, 3) BGR uint8
    timestamp: float    # time.perf_counter() at capture
    index: int          # increasing frame counter


@dataclass
class PoseObs:
    """One detected body in one frame (Task 2 output, anonymous until Task 3 assigns it)."""
    keypoints: np.ndarray   # (17, 2) float32, normalized image coords
    confidence: np.ndarray  # (17,) float32 in [0, 1]
    timestamp: float
    mask: np.ndarray | None = None  # optional (h, w) float32 person segmentation in [0, 1]

    def valid(self, min_conf: float = 0.3) -> np.ndarray:
        return self.confidence >= min_conf

    def bbox(self, min_conf: float = 0.3) -> tuple[float, float, float, float] | None:
        """(x, y, w, h) around the confident keypoints, normalized; None if none are confident."""
        pts = self.keypoints[self.valid(min_conf)]
        if len(pts) == 0:
            return None
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    def center(self, min_conf: float = 0.3) -> tuple[float, float] | None:
        pts = self.keypoints[self.valid(min_conf)]
        if len(pts) == 0:
            return None
        cx, cy = pts.mean(axis=0)
        return float(cx), float(cy)


@dataclass
class FaceObs:
    """One detected face in one frame (Task 1 output, anonymous until Task 3 assigns it)."""
    bbox: tuple[float, float, float, float]  # (x, y, w, h), normalized
    confidence: float
    timestamp: float
    landmarks: np.ndarray | None = None       # (N, 2) normalized, backend-specific layout
    yaw: float = 0.0                          # head orientation in degrees
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2


# --------------------------------------------------------------------------- players (Task 3)
class TrackState(Enum):
    ACTIVE = auto()  # seen in the current frame
    LOST = auto()    # temporarily not seen; last observation kept


@dataclass
class Player:
    pid: int
    name: str
    color: tuple[int, int, int]                 # BGR, used by every effect for this player
    state: TrackState = TrackState.ACTIVE
    pose: PoseObs | None = None
    face: FaceObs | None = None
    position_m: tuple[float, float] | None = None  # (lateral x, depth z) in metres, Task 3
    last_seen: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.state is TrackState.ACTIVE

    def head_top(self) -> tuple[float, float] | None:
        """Normalized point just above the head: anchor for name tags and popups."""
        if self.face is not None:
            x, y, w, _ = self.face.bbox
            return x + w / 2, y
        if self.pose is not None:
            box = self.pose.bbox()
            if box is not None:
                return box[0] + box[2] / 2, box[1]
        return None


# --------------------------------------------------------------------------- gameplay (Task 4)
class MoveType(Enum):
    NONE = auto()
    ARMS_UP = auto()
    T_POSE = auto()
    CLAP = auto()
    SQUAT = auto()
    POINT_LEFT = auto()
    POINT_RIGHT = auto()


class Grade(Enum):
    PERFECT = auto()
    GOOD = auto()
    OK = auto()
    MISS = auto()


class EventType(Enum):
    GRADE = auto()            # a choreography move was scored
    MOVE_DETECTED = auto()    # a discrete move/gesture was recognised
    INTERACTION = auto()      # player-player interaction (duet move, high five, ...)
    PLAYER_LOST = auto()
    PLAYER_RETURNED = auto()


@dataclass
class GameEvent:
    type: EventType
    t: float                     # song time
    pid: int | None = None
    grade: Grade | None = None
    move: MoveType | None = None
    data: dict = field(default_factory=dict)  # e.g. {"points": 50, "similarity": 0.83}
