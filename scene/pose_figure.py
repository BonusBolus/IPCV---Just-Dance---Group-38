import cv2

# Lines of the stick figure
BONES = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "neck"), ("right_shoulder", "neck"), ("nose", "neck")
]


def draw_pose(frame, pose, center_x, nose_y, scale, color, thickness=1):
    """Stick figure of a pose from poses.json: {"nose": (0, 0), "left_wrist": (x, y), ...}.

    center_x, nose_y: where the nose goes on the screen (pixels)
    scale:            pixels per pose unit"""
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


def blend_poses(pose_a, pose_b, amount):
    """A pose between pose_a (amount 0) and pose_b (amount 1). A point missing in one pose is taken from
    the pose it is closest to."""
    blended = {}
    for name in set(pose_a) | set(pose_b):
        a, b = pose_a.get(name), pose_b.get(name)
        if a is not None and b is not None:
            blended[name] = (a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount)
        else:
            blended[name] = a if amount < 0.5 else b
    return blended


def animated_pose(pose, next_pose, progress):
    """The figure of a step of an animated move: it holds the pose for the first half of the step, then
    moves smoothly to the pose of the next step (progress: 0 at the start of the step, 1 at the end).
    The players see where to go a moment before they have to be there."""
    if next_pose is None or progress <= 0.5:
        return pose
    amount = min(1.0, (progress - 0.5) * 2)
    return blend_poses(pose, next_pose, amount * amount * (3 - 2 * amount))     # smooth start and stop
