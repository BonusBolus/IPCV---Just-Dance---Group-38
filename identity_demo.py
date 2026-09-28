"""Try out the identity tracking on its own (task 3).

Run:  python identity_demo.py              (webcam)
      python identity_demo.py --video f.mp4
Keys: r = forget the players and register again, q / ESC = quit
"""
import argparse
import cv2
from camera import Camera, FPSCounter
from detection.people_detector import PeopleDetector, draw_people
from identity import IdentityTracker, draw_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video")
    args = parser.parse_args()

    camera = Camera(args.video if args.video else args.camera)
    detector = PeopleDetector()
    tracker = IdentityTracker(max_players=2)
    fps = FPSCounter()

    while True:
        frame = camera.read()
        if frame is None:
            break

        people = detector.detect(frame.image)   # box + 17 keypoints per person
        tracked = tracker.update(frame.image, people)

        output = frame.image.copy()
        draw_people(output, people)
        draw_labels(output, tracked)
        fps.update()
        cv2.putText(output, f"FPS: {fps.fps:.1f}  people: {len(people)}  players: {len(tracker.players)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Identity tracking", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            tracker.reset()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
