"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
      python main.py --debug         (debug drawings, n = next state)

Play: raise both hands above your head to start. q or ESC closes the game.
"""
import argparse

from camera import Camera
from game_logic.game import Game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="use a video file instead of the webcam")
    parser.add_argument("--debug", action="store_true", help="show debug drawings, n jumps to the next state")
    args = parser.parse_args()

    camera = Camera(args.video if args.video else args.camera)
    Game(camera, debug=args.debug).run()


if __name__ == "__main__":
    main()
