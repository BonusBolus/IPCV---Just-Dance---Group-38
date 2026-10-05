import numpy as np

class PoseGrader:

    # Reference poses, measured on a real player (made symmetric), in the units of _normalize_keypoints():
    # x in shoulder widths (shoulders at -0.5 and 0.5), y so that nose -> middle of the hips = 2.
    # left_* is on the left of the screen (the game swaps YOLO's left/right before grading).
    # Knees and ankles are None: these poses do not use the legs, and a webcam often cannot see them.
    # The grader skips None points (see _preprocess_pose). A pose that uses the legs can add them back.
    cactus = {
    "name": "Cactus",
    "keypoints": {
        "nose": (0,0),
        "left_eye": None,
        "right_eye": None,
        "left_ear": None,
        "right_ear": None,
        "left_shoulder": (-0.5,0.5),
        "right_shoulder": (0.5,0.5),
        "left_elbow": (-1.2,0.7),
        "right_elbow": (1.2,0.7),
        "left_wrist": (-1.25,-0.1),
        "right_wrist": (1.25,1.35),
        "left_hip": (-0.35,2),
        "right_hip": (0.35,2),
        "left_knee": None,
        "right_knee": None,
        "left_ankle": None,
        "right_ankle": None,
        }
    }

    t_pose = {
        "name": "T-Pose",
        "keypoints": {
            "nose": (0,0),
            "left_eye": None,
            "right_eye": None,
            "left_ear": None,
            "right_ear": None,
            "left_shoulder": (-0.5,0.5),
            "right_shoulder": (0.5,0.5),
            "left_elbow": (-1.4,0.6),
            "right_elbow": (1.4,0.6),
            "left_wrist": (-2.2,0.55),
            "right_wrist": (2.2,0.55),
            "left_hip": (-0.35,2),
            "right_hip": (0.35,2),
            "left_knee": None,
            "right_knee": None,
            "left_ankle": None,
            "right_ankle": None
            }
    }

    pencil = {
        "name": "Pencil",
        "keypoints": {
            "nose": (0,0),
            "left_eye": None,
            "right_eye": None,
            "left_ear": None,
            "right_ear": None,
            "left_shoulder": (-0.5,0.55),
            "right_shoulder": (0.5,0.55),
            "left_elbow": (-0.7,1.35),
            "right_elbow": (0.7,1.35),
            "left_wrist": (-0.95,2.1),
            "right_wrist": (0.95,2.1),
            "left_hip": (-0.35,2),
            "right_hip": (0.35,2),
            "left_knee": None,
            "right_knee": None,
            "left_ankle": None,
            "right_ankle": None
            }
    }

    poses = {
        "cactus": cactus,
        "t_pose": t_pose,
        "pencil": pencil,
    }

        
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
        
        #Convert all poses to numpy arrays for easier processing
        self.processed_poses = {
            name: self._preprocess_pose(pose)
            for name, pose in self.poses.items()
        }
        
    def _preprocess_pose(self, pose):
        
        # Remove keypoints that are None in reference pose
        valid = [
            (name, position)
            for name, position in pose["keypoints"].items()
            if position is not None
        ]
        
        # Convert the readable names to YOLO keypoint indices, these are still used by all keypoint arrays, so we need to keep track of them
        indices = np.array(
            [self.KEYPOINTS[name] for name, _ in valid],
            dtype=np.int32,
        )
        
        # Get corresponiding reference positions per keypoint. 
        reference = np.array(
            [position for _, position in valid],
            dtype=np.float32,
        )
        
        #return a dictionary with the name of the pose, the indices of the keypoints, and the reference positions
        return {
            "name": pose["name"],
            "indices": indices,
            "reference": reference,
        }
        
    def _normalize_keypoints(self, player_keypoints):
        if player_keypoints is None:
            raise ValueError("player_keypoints is None")

        player = np.asarray(player_keypoints, dtype=np.float32)
        if player.ndim != 2 or player.shape[1] not in (2, 3):
            raise ValueError(f"Expected keypoints with shape (N, 2 or 3), got {player.shape}")

        player = player.copy()

        # Center on nose
        nose = player[self.KEYPOINTS["nose"], :2].copy()
        player[:, :2] -= nose

        # Horizontal scale: shoulder width
        left_shoulder = player[self.KEYPOINTS["left_shoulder"], :2]
        right_shoulder = player[self.KEYPOINTS["right_shoulder"], :2]

        shoulder_width = np.linalg.norm(
            right_shoulder - left_shoulder
        )

        # Vertical scale: nose -> hip center
        left_hip = player[self.KEYPOINTS["left_hip"], :2]
        right_hip = player[self.KEYPOINTS["right_hip"], :2]

        hip_center = (left_hip + right_hip) / 2

        torso_height = abs(hip_center[1])

        player[:, 0] /= shoulder_width
        player[:, 1] /= torso_height / 2.0

        return player

    def grade_pose(self, player_keypoints, reference_pose):
        
        player = self._normalize_keypoints(player_keypoints)
        
        pose = self.processed_poses[reference_pose]
        
        # Only select keypoints that are defined in the reference pose
        player_positions = player[
            pose["indices"], :2
        ]
        
        # Euclidean error for every keypoint
        errors = np.linalg.norm(
            player_positions - pose["reference"],
            axis=1,
        )
        
        mean_error = errors.mean()
        
        # Convert error to score between approximately 0 and 100
        score = 100.0 * np.exp(-2.0 * mean_error)
        return score