"""Task 1 evaluation: face detection rate per player, face size, jitter, inference time.

    python -m eval.eval_face [--video f.mp4] [--seconds 20]

Metrics
  detection rate  frames in which an active player has a face assigned / frames the player is active
  face size       face box height in px (small faces = far away = the hard case)
  jitter          mean |second difference| of the face centre, px/frame, raw vs. smoothed
  yaw range       head turn angles that were still tracked
  time            FaceTracker.process per frame (all players together)
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from core.config import Config
from eval.common import add_source_args, frames, jitter, stats
from face.face_filter import FaceSmoother
from face.face_tracker import FaceTracker
from identity.player_tracker import PlayerTracker
from pose.pose_estimator import PoseEstimator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    args = p.parse_args()
    cfg = Config()
    est, faces_model, tracker, smoother = PoseEstimator(cfg), FaceTracker(cfg), PlayerTracker(cfg), FaceSmoother(cfg)

    active = defaultdict(int)
    found = defaultdict(int)
    raw_c, smooth_c = defaultdict(list), defaultdict(list)
    sizes, yaws, times = [], [], []
    for frame in frames(args):
        h, w = frame.image.shape[:2]
        poses = est.process(frame)
        t0 = time.perf_counter()
        faces = faces_model.process(frame, poses)
        times.append((time.perf_counter() - t0) * 1000)
        players = tracker.update(frame, poses, faces, frame.timestamp)
        for pid, pl in players.items():
            if not pl.is_active:
                continue
            active[pid] += 1
            if pl.face is None:
                raw_c[pid].append((np.nan, np.nan))
                smooth_c[pid].append((np.nan, np.nan))
                continue
            found[pid] += 1
            sizes.append(pl.face.bbox[3] * h)
            yaws.append(pl.face.yaw)
            raw_c[pid].append(np.array(pl.face.center) * (w, h))
            sm = smoother.update(pid, pl.face, frame.timestamp)
            smooth_c[pid].append(np.array(sm.center) * (w, h))
    est.close()
    faces_model.close()

    print(f"face tracker ms/frame: {stats(times)}")
    print(f"face height px:        {stats(sizes)}")
    if yaws:
        print(f"yaw range tracked:     {min(yaws):.0f} .. {max(yaws):.0f} deg")
    for pid in sorted(active):
        rate = found[pid] / active[pid]
        print(f"P{pid}: detection rate {rate:.1%} ({found[pid]}/{active[pid]} frames), jitter raw "
              f"{jitter(np.array(raw_c[pid])):.2f} px -> smoothed {jitter(np.array(smooth_c[pid])):.2f} px")


if __name__ == "__main__":
    main()
