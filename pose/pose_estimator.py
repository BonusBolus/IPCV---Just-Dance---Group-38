"""Task 2: multi-person body keypoint estimation."""
from __future__ import annotations

from core.config import Config
from core.types import FrameData, PoseObs

# MediaPipe's 33 landmarks -> our COCO-17 layout (index i of COCO = MEDIAPIPE_TO_COCO[i]).
MEDIAPIPE_TO_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


class PoseEstimator:
    """Estimates body keypoints for every person in the frame.

    Output contract: one PoseObs per visible person: COCO-17 keypoints normalized to [0, 1] plus
    per-keypoint confidence. Order carries no meaning: identity is Task 3.

    TODO(T2):
      - Pick a backend that supports >= 2 people: MediaPipe *Tasks* PoseLandmarker with
        num_poses=2, or YOLO-pose. NB: the legacy `mp.solutions.pose` tracks ONE person only.
      - Downscale to cfg.models.inference_width before inference (core.image_utils.resize_to_width).
        Normalized outputs stay valid at any resolution.
      - Map the backend's layout to COCO-17 (see MEDIAPIPE_TO_COCO) and its visibility/score
        to `confidence`.
      - Optional: fill PoseObs.mask with the person segmentation for background replacement (T5).
      - The frame is mirrored; left/right labels are in mirrored-image terms. That matches the
        un-mirrored reference video (see tools/extract_reference.py), so no swapping is needed.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None  # TODO(T2): load model from cfg.models.pose_model

    def process(self, frame: FrameData) -> list[PoseObs]:
        return []

    def close(self) -> None:
        pass
