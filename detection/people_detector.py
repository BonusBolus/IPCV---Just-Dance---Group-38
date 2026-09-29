"""People detector: finds every person in a frame with a box and 17 body keypoints.

Uses Ultralytics YOLO-pose (the same kind of model as the pose estimation task). The network
detects people and their keypoints in one pass. The keypoints follow the COCO layout below.
There is no identity yet: the order of the people in the list can change every frame.

The model (~6 MB) is downloaded automatically into detection/models/ the first time.

Usage:
    detector = PeopleDetector()
    people = detector.detect(frame.image)
    for person in people:
        person.box                          # (x, y, w, h) in pixels
        person.keypoint(LEFT_SHOULDER)      # (x, y) in pixels, or None if not visible
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "yolo11n-pose.pt"

# COCO keypoint order used by YOLO-pose
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class Person:
    box: tuple              # (x, y, w, h) in pixels
    score: float            # how sure the model is that this is a person (0..1)
    keypoints: np.ndarray   # (17, 2) x, y in pixels
    confidence: np.ndarray  # (17,) how sure the model is about each keypoint (0..1)

    def keypoint(self, index, min_confidence=0.5):
        """(x, y) of one keypoint, or None if the model is not sure about it
        (e.g. the hips are outside the image)."""
        if self.confidence[index] < min_confidence:
            return None
        x, y = self.keypoints[index]
        return int(x), int(y)

    def torso(self, min_confidence=0.5):
        """The four torso corners (left shoulder, right shoulder, right hip, left hip),
        or None if one of them is not visible."""
        points = [self.keypoint(i, min_confidence) for i in (LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP)]
        if any(p is None for p in points):
            return None
        return points


class PeopleDetector:
    def __init__(self, min_score=0.5, image_size=640):
        MODEL_PATH.parent.mkdir(exist_ok=True)
        self.model = YOLO(str(MODEL_PATH))  # downloads the weights if the file is missing
        self.min_score = min_score
        self.image_size = image_size  # the network input size; smaller = faster, less precise

    def detect(self, image):
        """Return a list of Person, one per detected person, in pixels of `image` (BGR)."""
        result = self.model(image, imgsz=self.image_size, conf=self.min_score, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        confidence = result.keypoints.conf.cpu().numpy()
        people = []
        for (x1, y1, x2, y2), score, kp, conf in zip(boxes, scores, keypoints, confidence):
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            people.append(Person(box, float(score), kp, conf))
        return people


def draw_people(image, people, color=(0, 255, 255)):
    """Draw the skeleton of every person, with the torso (shoulders + hips) highlighted."""
    for person in people:
        for a, b in SKELETON:
            pa, pb = person.keypoint(a), person.keypoint(b)
            if pa is not None and pb is not None:
                cv2.line(image, pa, pb, color, 2)
        torso = person.torso()
        if torso is not None:
            cv2.polylines(image, [np.array(torso)], True, (255, 0, 255), 2)
