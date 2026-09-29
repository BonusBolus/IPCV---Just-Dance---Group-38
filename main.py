"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
Press q or ESC to quit.
"""
import argparse

import cv2
import time

from functions import put_text_right

from camera import Camera, FPSCounter
from scene.scene import Scene

from game_loop.game_loop import get_current_pose, load_song

def process_frame(frame, song, start_time, scene):
    """Everything that happens with one camera frame. The tasks are added here:

    1. body pose estimation        (task 2)
    2. face tracking               (task 1)
    3. assign Player 1 / Player 2  (task 3)
    4. game logic and scoring      (task 4)
    5. draw the game scene         (task 5)

    Returns the image that is shown on screen.
    """
    output = frame.image.copy()

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
    fps = FPSCounter()

    scene = Scene()

    song = load_song("songs/song_1.json")

    start_time = time.time()

    while True:
        frame = camera.read()
        if frame is None:
            break

        output = process_frame(frame, song, start_time, scene)

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