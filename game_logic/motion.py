"""Motion per player, and the direction check for animated moves.

The motion is how far each body point moved since the last frame, in shoulder widths (so it does not
depend on the distance to the camera), smoothed over time like the motion of Task 2. It is made per
PLAYER, from the keypoints of the identity tracker: the PoseEstimator's own motion is per detected
person in YOLO's order, and that order can change between frames when two people are in the image.

Direction check (flow): between two steps of an animated move, the hands, elbows, knees and feet must move
the same way as the figure on the card. flow() compares the player's movement with the movement between
the two poses: 1 = exactly the same direction, 0 = at right angles, -1 = the opposite way.
"""
import math
from collections import deque

import numpy as np

import config
from identity_tracking.identity import KEYPOINT_NAMES

MOVING_JOINTS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"]
JOINT_INDEX = [KEYPOINT_NAMES.index(name) for name in MOVING_JOINTS]
LEFT_SHOULDER, RIGHT_SHOULDER = KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("right_shoulder")
MIN_CONFIDENCE = 0.5     # a keypoint must be this sure in both frames to count as motion
HISTORY = 2.0            # seconds of motion that are kept per player
MIN_JOINT_MOVE = 0.3     # shoulder widths: a joint that moves less between two poses is not checked
MIN_MOVED = 0.3          # the player must move at least this part of the figure's movement


class PlayerMotion:
    def __init__(self, smoothing=None):
        self.smoothing = config.MOTION_SMOOTHING if smoothing is None else smoothing
        self.previous = {}       # player id -> (17, 3) keypoints of the last frame
        self.smoothed = {}       # player id -> (17, 2) smoothed motion
        self.history = {}        # player id -> deque of (song time, (17, 2) motion)

    def update(self, player_id, keypoints, t):
        """keypoints: (17, 3) x, y, confidence in pixels, with the names of the poses (after MIRROR_SWAP);
        None when the player is not seen (then the next frame starts again, no jump)."""
        if keypoints is None:
            self.previous.pop(player_id, None)
            return
        previous = self.previous.get(player_id)
        self.previous[player_id] = keypoints
        width = float(np.linalg.norm(keypoints[RIGHT_SHOULDER, :2] - keypoints[LEFT_SHOULDER, :2]))
        if previous is None or width < 1.0:
            return
        sure = (keypoints[:, 2] >= MIN_CONFIDENCE) & (previous[:, 2] >= MIN_CONFIDENCE)
        motion = np.where(sure[:, None], (keypoints[:, :2] - previous[:, :2]) / width, 0.0)
        smoothed = self.smoothed.get(player_id)
        smoothed = motion if smoothed is None else self.smoothing * motion + (1 - self.smoothing) * smoothed
        self.smoothed[player_id] = smoothed
        history = self.history.setdefault(player_id, deque())
        history.append((t, smoothed))
        while history and history[0][0] < t - HISTORY:
            history.popleft()

    def displacement(self, player_id, t0, t1):
        """How far every body point moved between song times t0 and t1 (17, 2), in shoulder widths;
        None when there is no motion of this player in that time."""
        samples = [motion for t, motion in self.history.get(player_id, ()) if t0 <= t < t1]
        return np.sum(samples, axis=0) if samples else None


def flow(displacement, pose_from, pose_to):
    """How well the player's movement matches the movement from pose_from to pose_to (-1..1).
    None when nothing in the two poses moves much (then there is no direction to check)."""
    figure, player = [], []
    for name, index in zip(MOVING_JOINTS, JOINT_INDEX):
        a, b = pose_from.get(name), pose_to.get(name)
        if a is None or b is None:
            continue
        move = (b[0] - a[0], b[1] - a[1])
        if math.hypot(*move) < MIN_JOINT_MOVE:
            continue
        figure.append(move)
        player.append(displacement[index] if displacement is not None else (0.0, 0.0))
    if not figure:
        return None
    figure, player = np.ravel(figure), np.ravel(player)
    figure_length, player_length = np.linalg.norm(figure), np.linalg.norm(player)
    if player_length < MIN_MOVED * figure_length:
        return 0.0                                     # (almost) did not move
    return float(figure @ player / (figure_length * player_length))
