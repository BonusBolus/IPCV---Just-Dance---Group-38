"""Just Dance, IPCV Group 38.

Run:  python main.py                 (webcam 0)
      python main.py --camera 1      (other webcam)
      python main.py --video file.mp4
      python main.py --debug         (debug drawings, n = next state)
      python main.py --windowed      (a window instead of full screen)
      python main.py --measure       (time every stage, see game_logic/measurements.py)
      python main.py --measure --duration 60   (stop after 60 seconds)

Play: raise both hands above your head to start. q or ESC closes the game.
"""
import time

PROGRAM_START = time.perf_counter()     # for the start-up time in --measure (includes loading the libraries)

import argparse

from camera import Camera
from game_logic.game import Game
from game_logic.measurements import Measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="use a video file instead of the webcam")
    parser.add_argument("--debug", action="store_true", help="show debug drawings, n jumps to the next state")
    parser.add_argument("--windowed", action="store_true", help="show the game in a window instead of full screen")
    parser.add_argument("--measure", action="store_true",
                        help="time every stage; results in measurements/<date_time>/ (see game_logic/measurements.py)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds (for measurements)")
    args = parser.parse_args()

    measure = Measurements(program_start=PROGRAM_START) if args.measure else None
    camera = Camera(args.video if args.video else args.camera)
    Game(camera, debug=args.debug, fullscreen=not args.windowed, measure=measure, duration=args.duration).run()


if __name__ == "__main__":
    main()
