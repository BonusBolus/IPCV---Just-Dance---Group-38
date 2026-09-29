"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
Press q or ESC to quit, r to register the players again.
"""


import argparse
import cv2
import time

from functions import put_text_right
from camera import Camera, FPSCounter
from identity_tracking.identity import IdentityTracker, draw_labels, people_from_pose
from pose_tracking.pose_processing import PoseEstimator


from pose_tracking.pose_processing import PoseEstimator
pose_estimator = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)

from scene.scene import createScene
from poses.poses import poses
from game_logic.pose_grading import PoseGrader
pose_grader = PoseGrader()
from scene.scene import Scene

from game_loop.game_loop import get_current_pose, load_song

def process_frame(frame, tracker, song, start_time, scene):

    """Everything that happens with one camera frame. The tasks are added here:

    1. body pose estimation        (task 2)
    2. face tracking               (task 1)
    3. assign Player 1 / Player 2  (task 3)
    4. game logic and scoring      (task 4)
    5. draw the game scene         (task 5)

    Returns the image that is shown on screen.
    """
    
    output = frame.image.copy()
    
    
    keypoints, motion = pose_estimator.process(frame.image)   # task 2: body keypoints, no identity yet
    people = people_from_pose(keypoints, min_confidence=0.1)  # box around each set of keypoints
    tracked = tracker.update(frame.image, people)             # task 3: who is who  

    for player_id, keypoints in tracked.items():
        
        if player_id <= 2: # Only track Player 1 and Player 2
            score = pose_grader.grade_pose(keypoints, "cactus")

            player_name = f"Player {player_id}"
            
            # Needs to be initialed first time, otherwise KeyError
            players[player_name]["score"] += score
            players[player_name]["keypoints"] = keypoints
    


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
    # draw_keypoints(output, keypoints, confidence_threshold=0.1,)    # keypoints from body pose tracker, for debugging
    # draw_labels(output, tracked)                                    # labels from identity tracker, for debugging


    current_time = time.time()
    song_start_time = 5
    song_time = current_time - start_time - song_start_time
    if current_time - start_time < song_start_time:
        current_pose = None
    else:
        current_pose = get_current_pose(song, song_time)
    put_text_right(output, f"Song Time: {song_time:.1f}", (output.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output = scene.render(output, players, current_pose)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="use a video file instead of the webcam")
    args = parser.parse_args()

    camera = Camera(args.video if args.video else args.camera)
    tracker = IdentityTracker(max_players=2)
    fps = FPSCounter()

    scene = Scene()

    song = load_song("songs/song_1.json")

    start_time = time.time()

    while True:
        frame = camera.read()
        if frame is None:
            break

        output = process_frame(frame, tracker, song, start_time, scene)

        fps.update()
        cv2.putText(output, f"FPS: {fps.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Just Dance", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            tracker.reset()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()