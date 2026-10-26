from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment
import config

MATCH_DISTANCE = 0.5     # the same person as in the last frame when the keypoints are on average closer than
                         # this part of the person's height (moving between two frames is much less)
MAX_MISSED_FRAMES = 3    # nobody detected: keep the people of the last frame this many frames, then forget them


def use_one_openvino_core():
    """Make Ultralytics use one OpenVINO Core for all models (pose and segmentation).

    Ultralytics makes a new Core for every model and asks it for the available devices. On some laptops
    (seen on an AMD Ryzen AI laptop, OpenVINO 2026.4) the second Core crashes the whole program while it
    looks for Intel GPU/NPU devices: "segfault in libze_loader.so". With one Core the devices are searched
    only once. Call this before a model is loaded; calling it again does nothing."""
    import openvino
    if getattr(openvino.Core, "shared", False):
        return
    core = openvino.Core()

    def shared_core(*args, **kwargs):
        return core
    shared_core.shared = True
    openvino.Core = shared_core


def load_pose_model():
    """The YOLO pose model. With USE_OPENVINO it runs about 2x faster on the CPU.

    It is the same model, converted once to OpenVINO (takes about a second, the folder is made
    next to the .pt file). The keypoints are at most about 1 px different from PyTorch.
    `dynamic=True` makes it work for every camera size, like PyTorch."""
    if not config.USE_OPENVINO:
        return YOLO(str(config.YOLO_MODEL_PATH))
    use_one_openvino_core()
    folder = config.YOLO_MODEL_PATH.with_name(config.YOLO_MODEL_PATH.stem + "_openvino_model")
    if not folder.exists():
        print("Converting the YOLO model to OpenVINO (only the first time)...")
        YOLO(str(config.YOLO_MODEL_PATH)).export(format="openvino", dynamic=True)
    return YOLO(str(folder), task="pose")   # the folder name must end with _openvino_model


class PoseEstimator:
    """A class that estimates the pose of a person in a video frame using a YOLO model.
    First identifies the keypoints of the person in the frame, then estimates the motion based on the keypoints.
    Applies different smoothing techniques to the keypoints and motion to reduce noise and improve accuracy."""
    
    ### YOLO DEFINITIONS ###
    NUM_KEYPOINTS = 17
    KEYPOINT_SIZE = 3  # x, y, confidence

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
    
    def __init__(self, 
                 CONFIDENCE_THRESHOLD=0.1,
                 KEYPOINTS_SMOOTHING=0.7, 
                 MOTIONS_SMOOTHING=0.5):
        
        # Initialize the PoseEstimator with previous keypoints and motion set to None.
        self.previous_keypoints = None
        self.previous_motion = None
        self.missed_frames = 0      # frames in a row in which nobody was detected
        
        # Variables for smoothing and confidence threshold
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.KEYPOINTS_SMOOTHING = KEYPOINTS_SMOOTHING  # Smoothing factor for keypoint estimation
        self.MOTIONS_SMOOTHING = MOTIONS_SMOOTHING  # Smoothing factor for motion estimation

        #lightweight pose estimation model
        self.model = load_pose_model()
    
    def _empty_keypoints(self):
        return np.empty(
            (0, self.NUM_KEYPOINTS, self.KEYPOINT_SIZE),
            dtype=np.float32,
        )
        
    def _empty_motion(self):
        return np.empty(
            (0, self.NUM_KEYPOINTS, 2),
            dtype=np.float32,
        ) 
    
    def _estimate_keypoints(self, frame):
        
        # Use the YOLO model to predict keypoints from the frame.
        result = self.model.predict(frame, verbose=False)[0]
        
        #Check if results contain something
        if not result or result.keypoints is None or result.keypoints.data is None:
            return self._empty_keypoints()
        
        # Shape: [number_of_people, number_of_keypoints, 2] - convert to cpu, yolo runs on gpu by default
        positions = result.keypoints.xy.cpu().numpy()

        # Shape: [number_of_people, number_of_keypoints, 3]
        # Last dimension: [x, y, confidence]
        keypoints = result.keypoints.data.cpu().numpy()

        # Dimension check
        if keypoints.shape[-2:] != (
            self.NUM_KEYPOINTS,
            self.KEYPOINT_SIZE,
        ):
            raise ValueError(
                f"Unexpected keypoint shape: {keypoints.shape}"
            )

        return keypoints
    
    def _match_previous(self, keypoints):
        """For every person in this frame: the row of the same person in the previous frame, or -1 (new).

        YOLO does not keep the order of the people between frames (it sorts them by confidence), so
        with two people row 0 can be the other person in the next frame. The people are matched by
        position instead: the Hungarian algorithm on the mean distance of the keypoints that are sure
        in both frames. A pair further apart than MATCH_DISTANCE x the person's height is no match."""
        match = np.full(len(keypoints), -1)
        if self.previous_keypoints is None or len(keypoints) == 0 or len(self.previous_keypoints) == 0:
            return match
        previous = self.previous_keypoints
        cost = np.full((len(keypoints), len(previous)), np.inf)
        for i, person in enumerate(keypoints):
            for j, before in enumerate(previous):
                sure = (person[:, 2] >= self.CONFIDENCE_THRESHOLD) & (before[:, 2] >= self.CONFIDENCE_THRESHOLD)
                if sure.sum() >= 3:
                    cost[i, j] = np.linalg.norm(person[sure, :2] - before[sure, :2], axis=1).mean()
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))
        for i, j in zip(rows, cols):
            sure = keypoints[i, :, 2] >= self.CONFIDENCE_THRESHOLD
            height = np.ptp(keypoints[i, sure, 1]) if sure.sum() >= 2 else 0.0
            if cost[i, j] <= MATCH_DISTANCE * max(height, 50.0):
                match[i] = j
        return match

    def _handle_missing_keypoints(self, keypoints, match):
        
        # If the current keypoints are empty, keep the previous keypoints for a few frames (a short miss).
        if keypoints.size == 0:
            if self.previous_keypoints is not None and self.missed_frames < MAX_MISSED_FRAMES:
                self.missed_frames += 1
                return self.previous_keypoints
            return keypoints
        self.missed_frames = 0
        
        keypoints = keypoints.copy()  # Create a copy to avoid modifying the original array
        
        # For each person that was also in the previous frame, replace missing keypoints with that
        # person's keypoints of the previous frame.
        for i, j in enumerate(match):
            if j < 0:
                continue                                  # a new person: nothing to fill in from
            for k in range(self.NUM_KEYPOINTS):
                if keypoints[i, k, 2] < self.CONFIDENCE_THRESHOLD:  # Confidence threshold
                    # Reuse position, but keep current confidence.
                    keypoints[i, k, :2] = self.previous_keypoints[j, k, :2]
        
        return keypoints
    
    def _smooth_keypoints(self, keypoints, match):
        
        if keypoints.size == 0:
            return keypoints

        smoothed = keypoints.copy()

        # Apply exponential smoothing to the keypoints, each person with the same person of the last frame.
        for i, j in enumerate(match):
            if j >= 0:
                smoothed[i, :, :2] = (
                    self.KEYPOINTS_SMOOTHING * keypoints[i, :, :2]
                    + (1 - self.KEYPOINTS_SMOOTHING)
                    * self.previous_keypoints[j, :, :2]
                )

        return smoothed
    
    
    def _estimate_motion(self, keypoints, match):
        # No people: no motion.
        if keypoints.size == 0:
            return self._empty_motion()
        
        # Calculate the motion as the difference between the current and previous keypoints of the same
        # person. A new person has no motion yet.
        motion = np.zeros((len(keypoints), self.NUM_KEYPOINTS, 2), dtype=np.float32)
        for i, j in enumerate(match):
            if j >= 0:
                motion[i] = keypoints[i, :, :2] - self.previous_keypoints[j, :, :2]
        
        return motion
    
    def _smooth_motion(self, motion, match):
        # Apply exponential smoothing to the motion, each person with the same person of the last frame.
        if self.previous_motion is None or motion.size == 0:
            return motion
        smoothed_motion = motion.copy()
        for i, j in enumerate(match):
            if 0 <= j < len(self.previous_motion):
                smoothed_motion[i] = (self.MOTIONS_SMOOTHING * motion[i]
                                      + (1 - self.MOTIONS_SMOOTHING) * self.previous_motion[j])
        
        return smoothed_motion
        
        
    def process(self, frame):
            
            # Check frame validation
            if frame is None or frame.size == 0:
                return self._empty_keypoints(), self._empty_motion()
            
            # Estimate keypoints from the frame using the YOLO model, find each person of the previous
            # frame again (YOLO's order can change), handle any missing keypoints.
            keypoints = self._estimate_keypoints(frame)
            match = self._match_previous(keypoints)
            if keypoints.size == 0 and self.previous_keypoints is not None:
                match = np.arange(len(self.previous_keypoints))     # kept for a short miss: the same people
            keypoints = self._handle_missing_keypoints(keypoints, match)
            if len(match) != len(keypoints):
                match = np.full(len(keypoints), -1)                 # forgotten after the miss: no one to match
            keypoints = self._smooth_keypoints(keypoints, match)
            
            # Estimate motion based on the keypoints, apply smoothing to the motion.
            motion = self._estimate_motion(keypoints, match)
            motion = self._smooth_motion(motion, match)
    
            # Update the previous keypoints and motion for the next frame.
            self.previous_keypoints = keypoints
            self.previous_motion = motion
    
            return keypoints, motion
