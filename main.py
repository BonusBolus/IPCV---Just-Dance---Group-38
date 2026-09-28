"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
Press q or ESC to quit.
"""
import argparse
import cv2

from camera import Camera, FPSCounter

from pose_tracking.pose_processing import PoseEstimator
pose_estimator = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)
from scene.scene import createScene
from poses.poses import poses
from game_logic.pose_grading import PoseGrader
pose_grader = PoseGrader()

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
    
    keypoints, motion = pose_estimator.process(output)
    score = pose_grader.grade_pose(keypoints[0], "cactus")
    
    if len(keypoints) > 0:
        
        score = pose_grader.grade_pose(
                keypoints[0],
                "cactus",
            )

        draw_pose_comparison(
                output,
                keypoints[0],
                pose_grader,
                "cactus",
            )

        draw_keypoints(
            output,
            keypoints,
            confidence_threshold=0.1,
        )

        return output
    
    


    # score = [100, 200]  # Example scores for Player 1 and Player 2
    # colors = [(255, 0, 0), (0, 0, 255)]  # Colors for Player 1 and Player 2
    # current_pose = poses.cactus  # Example current pose
    # output = createScene(output, score, colors, current_pose)
    return output

def draw_keypoints(image, keypoints, confidence_threshold=0.1):
    """
    Draw detected pose keypoints on the image.

    keypoints shape:
        (num_people, num_keypoints, 3)

    Last dimension:
        [x, y, confidence]
    """
    for person in keypoints:
        for x, y, confidence in person:
            if confidence < confidence_threshold:
                continue

            cv2.circle(
                image,
                (int(x), int(y)),
                4,
                (0, 255, 0),
                -1,
            )

    return image

def draw_pose_comparison(
    image,
    player_keypoints,
    pose_grader,
    pose_name,
    origin=(150, 80),
    scale=60,
):
    """
    Draw normalized player pose and reference pose in the same coordinate system.

    Player:    green
    Reference: red
    """

    player = pose_grader._normalize_keypoints(player_keypoints)
    pose = pose_grader.processed_poses[pose_name]

    indices = pose["indices"]
    reference = pose["reference"]

    ox, oy = origin

    # Draw reference pose
    for ref_pos in reference:
        x, y = ref_pos

        point = (
            int(ox + x * scale),
            int(oy + y * scale),
        )

        cv2.circle(
            image,
            point,
            5,
            (0, 0, 255),
            -1,
        )

    # Draw normalized player pose
    for index in indices:
        x, y = player[index, :2]

        point = (
            int(ox + x * scale),
            int(oy + y * scale),
        )

        cv2.circle(
            image,
            point,
            5,
            (0, 255, 0),
            -1,
        )

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
