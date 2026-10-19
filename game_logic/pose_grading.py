import json

import numpy as np
import config


class PoseGrader:

    KEYPOINTS = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(self):
        # config.POSES_PATH is built from the project folder, so this also works
        # when the game is started from another folder
        with open(config.POSES_PATH, "r") as f:
            self.poses = json.load(f)

        # Convert reference keypoints to reference angles once
        self.processed_poses = {
            name: {
                "name": pose["name"],
                "angles": [
                    self._get_reference_angles(pose[key])
                    for key in ("keypoints", "keypoints_2")
                    if key in pose
                ],
            }
            for name, pose in self.poses.items()
        }

    def _angle(self, vector):
        """
        Directed angle:
            0°    = right
            90°   = down
            180°  = left
            -90°  = up
        """
        return np.degrees(
            np.arctan2(vector[1], vector[0])
        )

    def _angle_difference(self, angle_a, angle_b):
        """Smallest difference between two angles."""
        return abs(
            (angle_a - angle_b + 180.0) % 360.0 - 180.0
        )

    def _get_reference_angles(self, keypoints):
        """
        Convert reference keypoint coordinates from poses.json
        to directed limb angles.
        """

        def point(name):
            return np.asarray(keypoints[name], dtype=np.float32)

        angles = {}

        for side in ("left", "right"):

            shoulder = point(f"{side}_shoulder")
            elbow = point(f"{side}_elbow")
            wrist = point(f"{side}_wrist")

            hip = point(f"{side}_hip")
            knee = point(f"{side}_knee")
            ankle = point(f"{side}_ankle")
            
            limbs = {
            f"{side}_shoulder": (elbow - shoulder, [shoulder, elbow]),
            f"{side}_elbow": (wrist - elbow, [elbow, wrist]),
            f"{side}_hip": (knee - hip, [hip, knee]),
            f"{side}_knee": (ankle - knee, [knee, ankle]),
            }

            #some poses might not have knee or leg references (otherwise you would stand very far away)
            #delete these agnles
            for joint, (vector, points) in limbs.items():
                if all(np.all(np.isfinite(p)) for p in points):
                    angles[joint] = self._angle(vector)

        return angles


    def _body_coordinate_system(self, player):
        """
        Local body coordinate system.

        x-axis: left shoulder -> right shoulder
        y-axis: shoulder center -> hip center
        """

        left_shoulder = player[
            self.KEYPOINTS["left_shoulder"], :2
        ]

        right_shoulder = player[
            self.KEYPOINTS["right_shoulder"], :2
        ]

        left_hip = player[
            self.KEYPOINTS["left_hip"], :2
        ]

        right_hip = player[
            self.KEYPOINTS["right_hip"], :2
        ]

        shoulder_center = (
            left_shoulder + right_shoulder
        ) / 2.0

        hip_center = (
            left_hip + right_hip
        ) / 2.0

        x_axis = right_shoulder - left_shoulder
        x_axis /= np.linalg.norm(x_axis)

        y_axis = hip_center - shoulder_center
        y_axis /= np.linalg.norm(y_axis)

        return x_axis, y_axis

    def _vector_to_body_coordinates(
        self,
        vector,
        x_axis,
        y_axis,
    ):
        return np.array([
            np.dot(vector, x_axis),
            np.dot(vector, y_axis),
        ])

    def _get_player_angles(self, player_keypoints):

        player = np.asarray(
            player_keypoints,
            dtype=np.float32,
        )

        x_axis, y_axis = self._body_coordinate_system(player)

        angles = {}

        for side in ("left", "right"):

            shoulder = player[
                self.KEYPOINTS[f"{side}_shoulder"], :2
            ]

            elbow = player[
                self.KEYPOINTS[f"{side}_elbow"], :2
            ]

            wrist = player[
                self.KEYPOINTS[f"{side}_wrist"], :2
            ]

            hip = player[
                self.KEYPOINTS[f"{side}_hip"], :2
            ]

            knee = player[
                self.KEYPOINTS[f"{side}_knee"], :2
            ]

            ankle = player[
                self.KEYPOINTS[f"{side}_ankle"], :2
            ]

            upper_arm = self._vector_to_body_coordinates(
                elbow - shoulder,
                x_axis,
                y_axis,
            )

            forearm = self._vector_to_body_coordinates(
                wrist - elbow,
                x_axis,
                y_axis,
            )

            thigh = self._vector_to_body_coordinates(
                knee - hip,
                x_axis,
                y_axis,
            )

            lower_leg = self._vector_to_body_coordinates(
                ankle - knee,
                x_axis,
                y_axis,
            )

            angles[f"{side}_shoulder"] = self._angle(upper_arm)
            angles[f"{side}_elbow"] = self._angle(forearm)
            angles[f"{side}_hip"] = self._angle(thigh)
            angles[f"{side}_knee"] = self._angle(lower_leg)

        return angles

    def grade_pose(self, player_keypoints, reference_pose, player = 0):

        player_angles = self._get_player_angles(
            player_keypoints
        )

        reference_angles = self.processed_poses[
            reference_pose
        ]["angles"][player]

        errors = []

        for joint, target_angle in reference_angles.items():
                
            error = self._angle_difference(
                player_angles[joint],
                target_angle,
            )

            #make legs 4 times less important
            if "hip" in joint:
                error = error / 4
                
                
            errors.append(error)

        mean_error = np.mean(errors)

        score = 100.0 * np.exp(
            -(mean_error / 30.0)**config.SCORE_MULTIPLIER
        )
        return score
    
    def interaction_met(self, players, reference_pose):
        """players: two keypoint arrays, ordered left to right in the image."""
        interaction = self.poses[reference_pose]["interaction"]

        # Distances are in shoulder widths, so they don't depend on camera distance
        width = np.mean([
            np.linalg.norm(
                p[self.KEYPOINTS["right_shoulder"], :2]
                - p[self.KEYPOINTS["left_shoulder"], :2]
            )
            for p in players
        ])

        for c in interaction:
            a = players[0][self.KEYPOINTS[c["a"]], :2]
            b = players[1][self.KEYPOINTS[c["b"]], :2]
            if np.linalg.norm(a - b) / width > c["max_dist"]:
                return False

        return True