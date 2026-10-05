from ultralytics import YOLO
import numpy as np
import cv2
import config


class PoseEstimator:
    
    """
    A class that estimates the pose of a person in a video frame using a YOLO model.
    First identifies the keypoints of the person in the frame, then estimates the motion based on the keypoints.
    Applies different smoothing techniques to the keypoints and motion to reduce noise and improve accuracy.
    """
    
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
        
        # Variables for smoothing and confidence threshold
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.KEYPOINTS_SMOOTHING = KEYPOINTS_SMOOTHING  # Smoothing factor for keypoint estimation
        self.MOTIONS_SMOOTHING = MOTIONS_SMOOTHING  # Smoothing factor for motion estimation

        #lightweight pose estimation model
        self.model = YOLO(str(config.YOLO_MODEL_PATH))
    
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
    
    def _handle_missing_keypoints(self, keypoints):
        
        # If there are no previous keypoints, return the current keypoints as is.
        if self.previous_keypoints is None:
            return keypoints
        
        # If the current keypoints are empty, return the previous keypoints.
        if keypoints.size == 0:
            return self.previous_keypoints
        
        # If the number of people detected has changed, return the current keypoints as is.
        if keypoints.shape[0] != self.previous_keypoints.shape[0]:
            return keypoints
        
        keypoints = keypoints.copy()  # Create a copy to avoid modifying the original array
        
        # For each person, check for missing keypoints and replace them with the previous frame's keypoints.
        for i in range(keypoints.shape[0]):
            for j in range(self.NUM_KEYPOINTS):
                if keypoints[i, j, 2] < self.CONFIDENCE_THRESHOLD:  # Confidence threshold
                    # Reuse position, but keep current confidence.
                    keypoints[i, j] = self.previous_keypoints[i, j]
        
        return keypoints
    
    def _smooth_keypoints(self, keypoints):
        
        if self.previous_keypoints is None:
            return keypoints

        if keypoints.size == 0:
            return keypoints

        if keypoints.shape[0] != self.previous_keypoints.shape[0]:
            return keypoints

        smoothed = keypoints.copy()

        # Apply exponential smoothing to the keypoints.
        smoothed[:, :, :2] = (
            self.KEYPOINTS_SMOOTHING * keypoints[:, :, :2]
            + (1 - self.KEYPOINTS_SMOOTHING)
            * self.previous_keypoints[:, :, :2]
        )

        return smoothed
    
    
    def _estimate_motion(self, keypoints):
        # If there are no previous keypoints, return an empty motion array.
        if self.previous_keypoints is None:
            return self._empty_motion()
        
        # If the current keypoints are empty, return an empty motion array.
        if keypoints.size == 0:
            return self._empty_motion()
        
        # If the number of people detected has changed, return an empty motion array.
        if keypoints.shape[0] != self.previous_keypoints.shape[0]:
            return self._empty_motion()
        
        # Calculate the motion as the difference between the current and previous keypoints.
        motion = keypoints[:, :, :2] - self.previous_keypoints[:, :, :2]
        
        return motion
    
    def _smooth_motion(self, motion, MOTIONS_SMOOTHING=0.5):
        # If there are no previous motion values, return the current motion as is.
        if self.previous_motion is None:
            return motion
        
        # If the current motion is empty, return the previous motion.
        if motion.size == 0:
            return self.previous_motion
        
        # If the number of people detected has changed, return the current motion as is.
        if motion.shape[0] != self.previous_motion.shape[0]:
            return motion
        
        # Apply exponential smoothing to the motion.
        smoothed_motion = self.MOTIONS_SMOOTHING * motion + (1 - self.MOTIONS_SMOOTHING) * self.previous_motion
        
        return smoothed_motion
        
        
    def process(self, frame):
            
            # Check frame validation
            if frame is None or frame.size == 0:
                return self._empty_keypoints(), self._empty_motion()
            
            # Estimate keypoints from the frame using the YOLO model, handle any missing keypoints.
            keypoints = self._estimate_keypoints(frame)
            keypoints = self._handle_missing_keypoints(keypoints)
            keypoints = self._smooth_keypoints(keypoints)
            
            # Estimate motion based on the keypoints, apply smoothing to the motion.
            motion = self._estimate_motion(keypoints)
            motion = self._smooth_motion(motion)
    
            # Update the previous keypoints and motion for the next frame.
            self.previous_keypoints = keypoints
            self.previous_motion = motion
    
            return keypoints, motion


# def draw_keypoints(image, keypoints, confidence_threshold=0.1):
#     """
#     Draw detected pose keypoints on the image.
#     keypoints shape:
#         (num_people, num_keypoints, 3)

#     Last dimension:
#         [x, y, confidence]
#     """

#     for person in keypoints:
#         for x, y, confidence in person:
#             if confidence < confidence_threshold:
#                 continue
#             cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1,)
#     return image