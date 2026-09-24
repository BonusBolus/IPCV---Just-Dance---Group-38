"""Task 1: face detection, landmarks and head orientation with MediaPipe FaceLandmarker.

Pipeline per frame:
    for every body pose: head region from nose/eye/ear keypoints -> square crop, upscaled to
    crop_size (one affine warp, handles image borders) -> FaceLandmarker (1 face) -> 478
    landmarks mapped back to full-frame coordinates -> bbox, roll (eye line), yaw/pitch
    (facial transformation matrix) -> FaceObs
    no poses available -> full-frame FaceLandmarker with num_faces = number of players

Why head crops: two players standing ~2.5 m from a laptop webcam have faces of only ~30-50 px.
The face detector inside FaceLandmarker is a short-range model and misses such faces in the
full frame. On an upscaled crop the face fills the image, so detection works at distance, and
each crop has a known owner, which prevents faces being swapped between players.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from core.config import Config
from core.image_utils import bbox_iou, resize_to_width
from core.types import KP, FaceObs, FrameData, PoseObs

log = logging.getLogger(__name__)

# FaceMesh landmark indices
EYE_OUTER = (33, 263)
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
HEAD_KPS = [KP[n] for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]


class FaceTracker:
    """Output contract: one FaceObs per visible face, normalized coordinates. The list order
    carries no meaning; Task 3 assigns faces to players."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fc = cfg.face
        self.error: str | None = None
        self._crop_model = None
        self._full_model = None
        self._cache: list[FaceObs] = []
        try:
            self._crop_model = self._create(num_faces=1)
            self._full_model = self._create(num_faces=cfg.game.num_players)
        except Exception as exc:  # noqa: BLE001
            self.error = f"face model unavailable: {exc}"
            log.error("%s. Run `python -m tools.download_models`.", self.error)

    def _create(self, num_faces: int):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        if not self.fc.model.exists():
            raise FileNotFoundError(self.fc.model)
        options = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(self.fc.model)),
            running_mode=vision.RunningMode.IMAGE,  # crops of different people: no temporal tracking
            num_faces=num_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ public API
    def process(self, frame: FrameData, poses: list[PoseObs]) -> list[FaceObs]:
        if self._crop_model is None:
            return []
        if self.fc.every_n_frames > 1 and frame.index % self.fc.every_n_frames != 0:
            return self._cache
        crops = [c for c in (self.head_crop(p, frame.image.shape) for p in poses) if c is not None]
        if crops:
            faces = [f for c in crops if (f := self._detect_in_crop(frame, c)) is not None]
        else:
            faces = self._detect_full(frame)
        self._cache = _dedupe(faces)
        return self._cache

    def head_crop(self, pose: PoseObs, shape) -> tuple[float, float, float] | None:
        """Square head region (cx, cy, side) in pixels from the body keypoints, or None."""
        h, w = shape[:2]
        ok = pose.confidence >= 0.3
        pts = pose.keypoints * np.array([w, h], np.float32)
        head = [i for i in HEAD_KPS if ok[i]]
        if not head:
            return None
        cx, cy = pts[head].mean(axis=0)
        le, re = KP["left_ear"], KP["right_ear"]
        ls, rs = KP["left_shoulder"], KP["right_shoulder"]
        if ok[le] and ok[re]:
            span = np.linalg.norm(pts[le] - pts[re])
        elif ok[ls] and ok[rs]:
            span = 0.45 * np.linalg.norm(pts[ls] - pts[rs])
        else:
            span = 0.08 * h
        side = float(np.clip(self.fc.crop_scale * span, 40, 0.8 * h))
        return float(cx), float(cy - 0.1 * side), side  # shift up: include the forehead

    # ------------------------------------------------------------------ internals
    def _run(self, model, rgb: np.ndarray):
        import mediapipe as mp

        return model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))

    def _detect_in_crop(self, frame: FrameData, crop) -> FaceObs | None:
        cx, cy, side = crop
        s = self.fc.crop_size
        scale = s / side
        m = np.array([[scale, 0, s / 2 - cx * scale], [0, scale, s / 2 - cy * scale]], np.float32)
        patch = cv2.warpAffine(frame.image, m, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result = self._run(self._crop_model, cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        if not result.face_landmarks:
            return None
        lm = np.array([(p.x, p.y) for p in result.face_landmarks[0]], np.float32)
        h, w = frame.image.shape[:2]
        px = (lm * s - np.array([s / 2, s / 2], np.float32)) / scale + np.array([cx, cy], np.float32)
        matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
        return _make_face(px / np.array([w, h], np.float32), matrix, frame.timestamp, w / h)

    def _detect_full(self, frame: FrameData) -> list[FaceObs]:
        small = resize_to_width(frame.image, self.cfg.pose.inference_width)
        result = self._run(self._full_model, cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        h, w = frame.image.shape[:2]
        mats = result.facial_transformation_matrixes or []
        return [
            _make_face(np.array([(p.x, p.y) for p in lms], np.float32), mats[i] if i < len(mats) else None,
                       frame.timestamp, w / h)
            for i, lms in enumerate(result.face_landmarks)
        ]

    def close(self) -> None:
        for model in (self._crop_model, self._full_model):
            if model is not None:
                model.close()
        self._crop_model = self._full_model = None


def _make_face(landmarks: np.ndarray, matrix, timestamp: float, aspect: float) -> FaceObs:
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    roll = eye_roll(landmarks, aspect)
    yaw = pitch = 0.0
    if matrix is not None:
        r = np.asarray(matrix)[:3, :3]
        yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
        pitch = math.degrees(math.atan2(-r[1, 2], math.hypot(r[0, 2], r[2, 2])))
    return FaceObs(bbox=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)), confidence=0.9,
                   timestamp=timestamp, landmarks=landmarks, yaw=yaw, pitch=pitch, roll=roll)


def eye_roll(landmarks: np.ndarray, aspect: float) -> float:
    """In-plane head rotation in degrees from the eye line; counter-clockwise positive (OpenCV)."""
    a, b = landmarks[EYE_OUTER[0]], landmarks[EYE_OUTER[1]]
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    dx, dy = (right[0] - left[0]) * aspect, right[1] - left[1]
    return -math.degrees(math.atan2(dy, dx))


def _dedupe(faces: list[FaceObs], iou: float = 0.5) -> list[FaceObs]:
    """Two overlapping head crops can find the same face: keep one."""
    out: list[FaceObs] = []
    for f in faces:
        if all(bbox_iou(f.bbox, g.bbox) < iou for g in out):
            out.append(f)
    return out
