"""Turn a video of the model dancer into the choreography timeline (assets/choreography.json).

The reference video is processed UN-mirrored, while the live camera IS mirrored. A player
copying the on-screen dancer then produces the same image-space pose as the dancer, so no
left/right swapping is needed when scoring.

Move segments are not detected automatically: annotate "moves" by hand in the JSON afterwards
(see gameplay/choreography.py for the format).

Usage:
    python -m tools.extract_reference --video assets/model_dance.mp4 [--out assets/choreography.json]
Requires a working PoseEstimator (Task 2).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from core.config import ASSETS, Config
from core.types import FrameData
from gameplay.choreography import Choreography
from pose.pose_estimator import PoseEstimator

log = logging.getLogger("extract_reference")


def _area(pose) -> float:
    box = pose.bbox()
    return box[2] * box[3] if box else 0.0


def main() -> None:
    logging.basicConfig(level="INFO")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=str(ASSETS / "choreography.json"))
    p.add_argument("--song", default="song.mp3", help="song file name stored in the JSON")
    args = p.parse_args()

    estimator = PoseEstimator(Config())
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    times, kps, confs = [], [], []
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = idx / fps
        poses = estimator.process(FrameData(image=img, timestamp=t, index=idx))
        idx += 1
        if poses:
            dancer = max(poses, key=_area)  # the model dancer = the largest person in view
            times.append(t)
            kps.append(dancer.keypoints)
            confs.append(dancer.confidence)
        if idx % 100 == 0:
            log.info("%d frames processed", idx)
    cap.release()

    if not times:
        log.warning("No poses found. Is PoseEstimator implemented yet?")
    choreo = Choreography(times, kps, confs, moves=[], duration=idx / fps, song=args.song)
    choreo.save(args.out)
    log.info("Saved %d reference frames (%.1f s) to %s. Now annotate the moves.", len(times), idx / fps, Path(args.out))


if __name__ == "__main__":
    main()
