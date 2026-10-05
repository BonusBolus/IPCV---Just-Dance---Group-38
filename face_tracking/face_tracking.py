from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"

if not hasattr(cv2, "FaceDetectorYN"):
    raise RuntimeError(
        "YuNet requires OpenCV 4.5.4 or newer with cv2.FaceDetectorYN support."
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YuNet model not found: {MODEL_PATH}. Download "
        "face_detection_yunet_2023mar.onnx from OpenCV Zoo and place it here."
    )

# YuNet returns a face box, five facial keypoints, and a confidence score.
# The input size is updated to the camera frame before every detection, which
# preserves the available detail in small, distant faces.
landmarker = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)

_previous_faces = []
_missed_frames = []


def get_face_properties(frame, write_results=False):
    """
    Detect all faces in a frame and return their:

    - location: (center_x, center_y)
    - size: (width, height)
    - orientation: yaw, pitch, roll
    """

    height, width, _ = frame.shape

    faces = []
    landmarker.setInputSize((width, height))
    _, detections = landmarker.detect(frame)

    if detections is None:
        return faces

    for detection in detections:
        x1, y1, face_width, face_height = detection[:4].astype(int)
        x2 = x1 + face_width
        y2 = y1 + face_height
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # YuNet keypoints: right eye, left eye, nose, right mouth, left mouth.
        right_eye, left_eye, nose, right_mouth, left_mouth = (
            detection[4:14].reshape(5, 2)
        )

        # Roll
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        roll = np.degrees(np.arctan2(dy, dx))

        # Yaw
        eye_center = (left_eye + right_eye) / 2

        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        yaw = 0.0 if eye_distance == 0 else (
            (nose[0] - eye_center[0])
            / eye_distance
        ) * 90

        # Approximate pitch from the nose position between eye and mouth lines.
        mouth_center = (left_mouth + right_mouth) / 2
        face_height_pixels = np.linalg.norm(
            mouth_center - eye_center
        )

        nose_vertical = nose[1] - eye_center[1]

        pitch = 0.0 if face_height_pixels == 0 else (
            (nose_vertical / face_height_pixels) - 0.5
        ) * 90

        faces.append({
            "location": (center_x, center_y),

            "size": (
                face_width,
                face_height
            ),

            "orientation": {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll)
            }
        })

        if write_results:
            for i, face in enumerate(faces):
            
                    location = face["location"]
                    size = face["size"]
                    orientation = face["orientation"]
            
                    print(
                        f"Face {i}: "
                        f"Location={location}, "
                        f"Size={size}, "
                        f"Orientation={orientation}"
                    )

    return faces


def smooth_face_properties(faces, alpha=0.6, max_missed_frames=5):
    """Reduce frame-to-frame jitter in detected face properties.

    Each face is matched to the closest face from the preceding frame. Matched
    values are filtered with an exponential moving average: smaller ``alpha``
    values look steadier but respond more slowly to movement. When detection
    briefly fails, the last known properties remain available for up to
    ``max_missed_frames`` frames (without fading) before that face is removed.
    """
    global _previous_faces, _missed_frames

    smoothed_faces = []
    next_missed_frames = []
    unmatched_previous = set(range(len(_previous_faces)))

    for face in faces:
        x, y = face["location"]
        face_width, face_height = face["size"]

        # Find the nearest previous face. Limit the match distance so a face
        # that newly appears is not blended with a distant, different player.
        match_index = None
        if unmatched_previous:
            distances = {
                index: np.hypot(x - _previous_faces[index]["location"][0],
                                y - _previous_faces[index]["location"][1])
                for index in unmatched_previous
            }
            candidate = min(distances, key=distances.get)
            max_distance = max(face_width, face_height) * 1.5
            if distances[candidate] <= max_distance:
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
            previous_x, previous_y = previous["location"]
            previous_width, previous_height = previous["size"]

            def blend(current, old):
                return alpha * current + (1.0 - alpha) * old

            smoothed_face = {
                "location": (
                    int(round(blend(x, previous_x))),
                    int(round(blend(y, previous_y))),
                ),
                "size": (
                    int(round(blend(face_width, previous_width))),
                    int(round(blend(face_height, previous_height))),
                ),
                "orientation": {
                    axis: blend(face["orientation"][axis], previous["orientation"][axis])
                    for axis in ("yaw", "pitch", "roll")
                },
            }

        smoothed_faces.append(smoothed_face)
        next_missed_frames.append(0)

    # A detector can miss a face for a few frames because of motion blur or an
    # occlusion. Keep its last filtered properties during this grace period so
    # the outline does not disappear and immediately reappear.
    for index in unmatched_previous:
        missed_frames = _missed_frames[index] + 1
        if missed_frames <= max_missed_frames:
            smoothed_faces.append(_previous_faces[index])
            next_missed_frames.append(missed_frames)

    _previous_faces = smoothed_faces
    _missed_frames = next_missed_frames
    return smoothed_faces
