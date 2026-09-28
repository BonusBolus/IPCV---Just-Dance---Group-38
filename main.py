"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
Press q or ESC to quit.
"""
import argparse

import cv2

from camera import Camera, FPSCounter

from scene.scene import createScene
from poses.poses import poses

def process_frame(frame):
    """Everything that happens with one camera frame. The tasks are added here:

    1. body pose estimation        (task 2)
    2. face tracking               (task 1)
    3. assign Player 1 / Player 2  (task 3)
    4. game logic and scoring      (task 4)
    5. draw the game scene         (task 5)

    Returns the image that is shown on screen.
    """
    output = frame.image.copy()

    current_pose = poses["t_pose"]  # Example current pose
    example_player_1 = {
        "ID": 0,
        "color": (0, 0, 255),  # Example color for Player 1 (RGB)
        "score": 100,
        "keypoints": {
            "nose":           (340, 120),
            "left_eye":       (320, 104),
            "right_eye":      (360, 104),
            "left_ear":       (298, 110),
            "right_ear":      (382, 110),
            "left_shoulder":  (258, 203),
            "right_shoulder": (422, 203),
            "left_elbow":     (126, 203),
            "right_elbow":    (555, 203),
            "left_wrist":     (126, 71),
            "right_wrist":    (555, 335),
            "left_hip":       (282, 368),
            "right_hip":      (398, 368),
            "left_knee":      (258, 491),
            "right_knee":     (422, 491),
            "left_ankle":     (266, 615),
            "right_ankle":    (414, 615),
        },
    }

    example_player_2 = {
        "ID": 1,
        "color": (0, 255, 0),  # Example color for Player 2 (RGB)
        "score": 200,
        "keypoints": {
            "nose":           (940, 120),
            "left_eye":       (920, 104),
            "right_eye":      (960, 104),
            "left_ear":       (898, 110),
            "right_ear":      (982, 110),
            "left_shoulder":  (858, 203),
            "right_shoulder": (1022, 203),
            "left_elbow":     (726, 203),
            "right_elbow":    (1155, 203),
            "left_wrist":     (726, 71),
            "right_wrist":    (1155, 335),
            "left_hip":       (882, 368),
            "right_hip":      (998, 368),
            "left_knee":      (858, 491),
            "right_knee":     (1022, 491),
            "left_ankle":     (866, 615),
            "right_ankle":    (1014, 615),
        },
    }
    players = [example_player_1, example_player_2]  # Example list of players
    output = createScene(output, players, current_pose)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="use a video file instead of the webcam")
    args = parser.parse_args()

    camera = Camera(args.video if args.video else args.camera)
    fps = FPSCounter()

    while True:
        frame = camera.read()
        if frame is None:
            break

        output = process_frame(frame)

        fps.update()
        cv2.putText(output, f"FPS: {fps.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Just Dance", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
