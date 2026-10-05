"""Draw a pose as a stick figure. Used by the move cards."""
import cv2

# Lines of the stick figure
BONES = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "neck"), ("right_shoulder", "neck"), ("nose", "neck")
]


def draw_pose(frame, pose, center_x, nose_y, scale, color, thickness=1):
    """Stick figure of a pose from poses.json: {"nose": (0, 0), "left_wrist": (x, y), ...}.

    center_x, nose_y: where the nose goes on the screen (pixels)
    scale:            pixels per pose unit
    """
    pose_points = {**pose, "neck": (0, 0.3)}

    def to_pixels(point):
        return int(center_x + point[0] * scale), int(nose_y + point[1] * scale)

    for point in pose_points.values():
        if point is not None:
            cv2.circle(frame, to_pixels(point), 2, color, -1)
    if pose_points.get("nose") is not None:
        cv2.circle(frame, to_pixels(pose_points["nose"]), int(scale * 0.2), color, -1)

    for start, end in BONES:
        if pose_points.get(start) is not None and pose_points.get(end) is not None:
            cv2.line(frame, to_pixels(pose_points[start]), to_pixels(pose_points[end]), color, thickness)
    return frame
