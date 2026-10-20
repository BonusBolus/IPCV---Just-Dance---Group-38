from pathlib import Path
import cv2
import numpy as np

import config


# YuNet detects faces.
MODEL_PATH = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.6

if not hasattr(cv2, "FaceDetectorYN"):
    raise RuntimeError("YuNet requires OpenCV 4.5.4 or newer.")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"YuNet model not found: {MODEL_PATH}")

# The initial size is required by OpenCV; it is replaced with each frame's
# native dimensions in get_face_properties so distant faces are not downscaled.
face_detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH), "", (320, 320),
    score_threshold=SCORE_THRESHOLD,
    nms_threshold=0.3,
    top_k=5000,
)

_previous_faces = []
_missed_frames = []


def _orientation_from_keypoints(keypoints):
    """Estimate roll from YuNet's five facial keypoints."""
    right_eye, left_eye, *_ = keypoints
    eye_vector = right_eye - left_eye
    # Roll is measured from the eye line.
    roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0])) + 180
    if roll < 180:
        roll += 360
    return {
        "roll": float(roll),
    }


def get_face_properties(frame, write_results=False):
    """Detect faces and return their location, size, and estimated orientation."""
    height, width = frame.shape[:2]
    # The players' faces are big, so YuNet finds them just as well on a smaller copy of the image
    # (FACE_DETECT_WIDTH, 640 px wide). At 1280x720 that is about 3x faster, and the result is even
    # steadier. The box and the five points are scaled back to the full image.
    scale = min(1.0, config.FACE_DETECT_WIDTH / width)
    small = frame if scale == 1.0 else cv2.resize(
        frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    face_detector.setInputSize((small.shape[1], small.shape[0]))
    _, detections = face_detector.detect(small)

    faces = []
    if detections is None:
        return faces
    if scale != 1.0:
        detections = detections.copy()
        detections[:, :14] /= scale                   # box (x, y, w, h) and 5 points (x, y): back to full size

    for detection in detections:
        x, y, face_width, face_height = np.rint(detection[:4]).astype(int)
        if face_width <= 0 or face_height <= 0:
            continue

        # YuNet outputs: box, then right eye, left eye, nose, right mouth,
        # left mouth, followed by the confidence score.
        keypoints = detection[4:14].reshape(5, 2)
        faces.append({
            "location": (x + face_width // 2, y + face_height // 2),
            "size": (face_width, face_height),
            "orientation": _orientation_from_keypoints(keypoints),
        })

    if write_results:
        for index, face in enumerate(faces):
            print(
                f"Face {index}: Location={face['location']}, "
                f"Size={face['size']}, Orientation={face['orientation']}"
            )

    return faces


def smooth_face_properties(faces, alpha=0.5, max_missed_frames=5):
    """Smooth face properties and retain brief detector misses.

    A detected face is paired with the nearest unmatched face from the previous
    frame. ``alpha`` controls the balance between the new and previous value."""
    global _previous_faces, _missed_frames # Global variables preserving information between calls.

    smoothed_faces = []
    next_missed_frames = []
    unmatched_previous = set(range(len(_previous_faces)))

    for face in faces:
        x, y = face["location"]
        face_width, face_height = face["size"]
        match_index = None

        # Calculating Euclidian distances between current and old face locations.
        if unmatched_previous:
            distances = {
                index: np.hypot(
                    x - _previous_faces[index]["location"][0],
                    y - _previous_faces[index]["location"][1],
                )
                for index in unmatched_previous
            }
            candidate = min(distances, key=distances.get) # Selecting smallest distance face.

            # Avoid blending a newly arrived player with another player.
            if distances[candidate] <= max(face_width, face_height) * 1.5:
                match_index = candidate
                unmatched_previous.remove(candidate)

        if match_index is None:
            smoothed_face = {
                "location": (x, y),
                "size": (face_width, face_height),
                "orientation": face["orientation"].copy(),
            }
        else:
            previous = _previous_faces[match_index]

            def blend(current, old):
                return alpha * current + (1.0 - alpha) * old

            smoothed_face = {
                "location": tuple(
                    int(round(blend(current, old)))
                    for current, old in zip(face["location"], previous["location"])
                ),
                "size": tuple(
                    int(round(blend(current, old)))
                    for current, old in zip(face["size"], previous["size"])
                ),
                "orientation": {
                    axis: blend(face["orientation"][axis], previous["orientation"][axis])
                    for axis in ("roll",)
                },
            }

        smoothed_faces.append(smoothed_face)
        next_missed_frames.append(0)

    # Keep a last-known face briefly through blur or a one-frame detection miss.
    for index in unmatched_previous:
        missed_frames = _missed_frames[index] + 1
        if missed_frames <= max_missed_frames:
            smoothed_faces.append(_previous_faces[index])
            next_missed_frames.append(missed_frames)

    _previous_faces = smoothed_faces
    _missed_frames = next_missed_frames
    return smoothed_faces
