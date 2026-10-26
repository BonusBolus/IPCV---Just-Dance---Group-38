"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
      python main.py --debug         (debug drawings, n = next state)
      python main.py --windowed      (a window instead of full screen)
      python main.py --pose-timing pose_timing.csv  (save pose latency samples and summary)

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
    parser.add_argument("--windowed", action="store_true", help="show the game in a window instead of full screen")
    parser.add_argument("--pose-timing", metavar="PATH",
                        help="save per-frame pose latency data and a mean/min/max summary to CSV")
    args = parser.parse_args()

    camera = Camera(args.video if args.video else args.camera)
    Game(camera, debug=args.debug, fullscreen=not args.windowed, timing_path=args.pose_timing).run()


if __name__ == "__main__":
    main()
