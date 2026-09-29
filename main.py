"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
Press q or ESC to quit.
"""
import argparse

import cv2

from camera import Camera, FPSCounter

from face_tracking import get_face_properties, smooth_face_properties
from face_overlay import draw_face_outline
from head_filter import enlarge_heads

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
    
    faces = smooth_face_properties(get_face_properties(output)) # Get properties of every detected face
    HEAD_ENLARGEMENT = 1.35
    enlarge_heads(output, faces, HEAD_ENLARGEMENT) # Enlarge detected heads
    # draw_face_outline(output, faces) # Draw oval shaped outline of heads

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
