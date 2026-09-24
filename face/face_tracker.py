"""Task 1: face detection and head orientation."""
from __future__ import annotations

from core.config import Config
from core.types import FaceObs, FrameData, PoseObs


class FaceTracker:
    """Detects every face in a frame.

    Output contract: one FaceObs per visible face, bbox normalized to [0, 1], with yaw/pitch/roll
    in degrees if available. Order carries no meaning: Task 3 decides which player a face belongs to.

    TODO(T1):
      - Pick a detector that handles >= 2 faces (e.g. MediaPipe Tasks FaceLandmarker, num_faces=2;
        its facial transformation matrix gives head rotation for aligning effects).
      - At ~2.5 m from a laptop webcam faces are only ~30-50 px. Consider cropping the head region
        around the pose's nose/ear keypoints (the `poses` argument), upscaling the crop, and
        running the face model on the crop instead of on the full frame.
      - Consider running every 2nd frame and relying on the filter in between, if it is slow.
      - Measure time per call (the main loop profiles this as section "face").
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None  # TODO(T1): load model from cfg.models.face_model

    def process(self, frame: FrameData, poses: list[PoseObs]) -> list[FaceObs]:
        return []

    def close(self) -> None:
        pass
