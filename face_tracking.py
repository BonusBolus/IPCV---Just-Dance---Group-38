import cv2
import mediapipe as mp
import numpy as np


# MediaPipe Face Landmarker setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=10
)

landmarker = FaceLandmarker.create_from_options(options)

# State used by ``smooth_face_properties``. Keeping it here makes the filter
# persist across frames while leaving face detection itself stateless.
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

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect faces
    result = landmarker.detect(mp_image)

    faces = []

    for landmarks in result.face_landmarks:

        # -----------------------------
        # Location and size
        # -----------------------------

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        x1 = int(min_x * width)
        y1 = int(min_y * height)
        x2 = int(max_x * width)
        y2 = int(max_y * height)

        face_width = x2 - x1
        face_height = y2 - y1

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # -----------------------------
        # Orientation
        # -----------------------------

        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]
        chin = landmarks[152]

        left_eye = np.array([
            left_eye.x * width,
            left_eye.y * height
        ])

        right_eye = np.array([
            right_eye.x * width,
            right_eye.y * height
        ])

        nose = np.array([
            nose.x * width,
            nose.y * height
        ])

        chin = np.array([
            chin.x * width,
            chin.y * height
        ])

        # Roll
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        roll = np.degrees(np.arctan2(dy, dx))

        # Yaw
        eye_center = (left_eye + right_eye) / 2

        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        yaw = (
            (nose[0] - eye_center[0])
            / eye_distance
        ) * 90

        # Pitch
        face_height_pixels = np.linalg.norm(
            chin - eye_center
        )

        nose_vertical = nose[1] - eye_center[1]

        pitch = (
            (nose_vertical / face_height_pixels) - 0.5
        ) * 90

        # -----------------------------
        # Store result
        # -----------------------------

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
