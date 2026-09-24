"""Task 3 evaluation: identity switches, recovery after leaving, spatial accuracy, timing.

    python -m eval.eval_identity --mock              # synthetic crossings with ground truth
    python -m eval.eval_identity [--video f.mp4]     # real footage: track statistics
    python -m eval.eval_identity --distance 2.5      # stand at a tape-measured 2.5 m: depth error

Mock experiments (ground truth known; detections shuffled every frame like a real detector):
  crossing      players swap places every 10 s          -> identity switches
  dropout       + 15% random missed detections          -> identity switches
  leave/re-enter player 2 disappears for 2 s every 12 s -> correct pid after re-entry
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from core.config import Config
from core.mock_source import MockPoseSource
from core.types import FrameData
from eval.common import add_source_args, frames, stats
from face.face_tracker import FaceTracker
from identity.player_tracker import PlayerTracker
from identity.spatial import SpatialEstimator
from pose.pose_estimator import PoseEstimator

W, H = 1280, 720


def mock_run(seconds: float, dropout: float = 0.0, leave: bool = False, seed: int = 0) -> dict:
    cfg = Config()
    tracker = PlayerTracker(cfg)
    mock = MockPoseSource(scenario="cross", dropout=dropout, seed=seed)
    frame = FrameData(np.zeros((H, W, 3), np.uint8), 0.0, 0)
    mapping: dict[int, int] = {}   # true id -> assigned pid
    switches, reentries_ok, reentries = 0, 0, 0
    was_gone = False
    times = []
    for i in range(int(seconds * 30)):
        t = i / 30
        frame.index, frame.timestamp = i, t
        poses, faces = mock.generate(t, (H, W))
        ids = list(mock.true_ids)
        gone = leave and (t % 12) > 10
        if gone:  # player 2 leaves the frame
            keep = [k for k, tid in enumerate(ids) if tid != 2]
            poses, faces, ids = [poses[k] for k in keep], [faces[k] for k in keep], [ids[k] for k in keep]
        t0 = time.perf_counter()
        players = tracker.update(frame, poses, faces, t)
        times.append((time.perf_counter() - t0) * 1000)
        for tid, pose in zip(ids, poses):
            pid = next((p.pid for p in players.values() if p.pose is pose), None)
            if pid is None:
                continue
            if tid in mapping and mapping[tid] != pid:
                switches += 1
            if tid == 2 and was_gone and not gone:
                reentries += 1
                reentries_ok += int(mapping.get(2, pid) == pid)
            mapping[tid] = pid
        was_gone = gone
    return {"switches": switches, "reentries": reentries, "reentries_ok": reentries_ok,
            "crossings": int(seconds / 5), "ms": times}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--distance", type=float, help="true distance (m) of the player to the camera")
    args = p.parse_args()

    if args.mock:
        seconds = max(args.seconds, 60)
        for name, kw in (("crossing", {}), ("crossing + 15% dropout", {"dropout": 0.15}),
                         ("crossing + leave/re-enter", {"leave": True})):
            r = mock_run(seconds, **kw)
            print(f"{name:<28} switches {r['switches']:3d} over {r['crossings']} crossings"
                  + (f", re-entries correct {r['reentries_ok']}/{r['reentries']}" if r["reentries"] else "")
                  + f", update {np.mean(r['ms']):.2f} ms")
        return

    cfg = Config()
    est, face, tracker, spatial = PoseEstimator(cfg), FaceTracker(cfg), PlayerTracker(cfg), SpatialEstimator(cfg)
    times, depth = [], {}
    active_frames = Counter()
    for frame in frames(args):
        poses = est.process(frame)
        faces = face.process(frame, poses)
        t0 = time.perf_counter()
        players = tracker.update(frame, poses, faces, frame.timestamp)
        times.append((time.perf_counter() - t0) * 1000)
        for pid, pl in players.items():
            if pl.is_active:
                active_frames[pid] += 1
                pos = spatial.estimate(pl, frame.image.shape)
                if pos is not None:
                    depth.setdefault(pid, []).append(pos[1])
    est.close()
    face.close()
    print(f"tracker update ms: {stats(times)}")
    print(f"tracker counters:  {dict(tracker.stats)}")
    for pid in sorted(active_frames):
        z = np.array(depth.get(pid, [np.nan]))
        line = f"P{pid}: active {active_frames[pid]} frames, depth {stats(z)} m"
        if args.distance:
            line += f", error {np.nanmean(z) - args.distance:+.2f} m ({(np.nanmean(z) / args.distance - 1):+.0%})"
        print(line)


if __name__ == "__main__":
    main()
