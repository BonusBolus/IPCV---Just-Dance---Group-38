"""Task 2: multi-person body keypoint estimation with MediaPipe PoseLandmarker.

Pipeline per frame:
    BGR frame (mirrored) -> downscale to `inference_width` -> RGB -> PoseLandmarker (VIDEO mode,
    num_poses = number of players, optional segmentation masks) -> 33 landmarks per person
    -> COCO-17 subset + confidence (= visibility, 0 outside the image) -> list[PoseObs]

Design notes
- The legacy `mp.solutions.pose` tracks only one person. The Tasks PoseLandmarker handles
  several: a BlazePose detector finds people and a landmark model refines each of them. In
  VIDEO mode, poses found in the previous frame are tracked instead of re-detected, which is
  faster and more stable.
- Landmarks are normalized to the input image, so downscaling does not change the output.
  The landmark model runs on a fixed 256x256 crop per person, so inference cost barely depends
  on the input resolution (measured: 640 px 23 ms vs 480 px 22 ms, full model, masks on).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from core.config import Config
from core.image_utils import resize_to_width
from core.types import FrameData, PoseObs

log = logging.getLogger(__name__)

# MediaPipe's 33 landmarks -> our COCO-17 layout (index i of COCO = MEDIAPIPE_TO_COCO[i]).
MEDIAPIPE_TO_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


class PoseEstimator:
    """Estimates body keypoints for every person in the frame.

    Output contract: one PoseObs per visible person, with COCO-17 keypoints normalized to
    [0, 1] plus per-keypoint confidence. The list order carries no meaning: identity is Task 3.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pc = cfg.pose
        self.model = None
        self.error: str | None = None
        self._last_ts_ms = -1
        try:
            self.model = self._create()
        except Exception as exc:  # noqa: BLE001 - missing model file / mediapipe
            self.error = f"pose model unavailable: {exc}"
            log.error("%s. Run `python -m tools.download_models`.", self.error)

    def _create(self):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        if not self.pc.model.exists():
            raise FileNotFoundError(self.pc.model)
        options = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(self.pc.model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.cfg.game.num_players,
            min_pose_detection_confidence=self.pc.min_detection_conf,
            min_pose_presence_confidence=self.pc.min_detection_conf,
            min_tracking_confidence=self.pc.min_tracking_conf,
            output_segmentation_masks=self.pc.segmentation,
        )
        log.info("Pose model: %s", self.pc.model.name)
        return vision.PoseLandmarker.create_from_options(options)

    def process(self, frame: FrameData) -> list[PoseObs]:
        if self.model is None:
            return []
        import mediapipe as mp

        small = resize_to_width(frame.image, self.pc.inference_width)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        # VIDEO mode needs strictly increasing integer timestamps
        ts_ms = max(int(frame.timestamp * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self.model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        h, w = frame.image.shape[:2]
        masks = result.segmentation_masks or []
        poses = []
        for i, landmarks in enumerate(result.pose_landmarks):
            pts = np.array([(landmarks[j].x, landmarks[j].y) for j in MEDIAPIPE_TO_COCO], np.float32)
            conf = np.array([landmarks[j].visibility or 0.0 for j in MEDIAPIPE_TO_COCO], np.float32)
            inside = (pts[:, 0] >= 0) & (pts[:, 0] <= 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
            conf = np.where(inside, conf, 0.0).astype(np.float32)  # extrapolated off-screen points
            mask = masks[i].numpy_view().copy() if i < len(masks) else None
            if mask is not None and mask.ndim == 3:
                mask = mask[..., 0]
            pose = PoseObs(pts, conf, timestamp=frame.timestamp, aspect=w / h, mask=mask)
            box = pose.bbox(self.pc.min_keypoint_conf)
            if box is None or box[3] < self.pc.min_body_height:
                continue  # tiny/uncertain detection: a spectator far away or a false positive
            poses.append(pose)
        return poses

    def close(self) -> None:
        if self.model is not None:
            self.model.close()
            self.model = None
