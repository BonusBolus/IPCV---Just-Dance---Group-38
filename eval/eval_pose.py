"""Task 2 evaluation: keypoint stability, filter lag, missing keypoints, inference time.

    python -m eval.eval_pose --mock                 # synthetic dancer: error vs. ground truth
    python -m eval.eval_pose [--video f.mp4]        # real footage (default: webcam, 20 s)

Metrics
  jitter   mean |second difference| of keypoint positions, px/frame (lower = smoother)
  lag      delay (frames) of the filtered wrist vs. the ground truth (mock) or vs. a zero-phase
           smoothed copy of the raw signal (real footage)
  error    (mock only) mean distance to the noise-free ground truth, px: all / torso / wrist
  missing  fraction of frames in which a keypoint is below the confidence threshold
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from core.config import Config
from core.filters import ExponentialSmoother
from core.mock_source import MockPoseSource
from core.types import KP, KP_NAMES
from eval.common import add_source_args, frames, jitter, lag_frames, stats, zero_phase_smooth
from pose.keypoint_filter import PoseSmoother
from pose.pose_estimator import PoseEstimator

W, H = 1280, 720


def run_mock(cfg: Config, seconds: float) -> dict:
    mock = MockPoseSource(num_players=1, noise=0.004, shuffle=False)
    truth = MockPoseSource(num_players=1, noise=0.0, shuffle=False)
    smoother, ema = PoseSmoother(cfg), ExponentialSmoother(0.4)
    out = {"raw": [], "ema": [], "one_euro": [], "truth": []}
    for i in range(int(seconds * 30)):
        t = i / 30
        pose = mock.generate(t, (H, W))[0][0]
        out["truth"].append(truth.generate(t, (H, W))[0][0].keypoints)
        out["raw"].append(pose.keypoints)
        out["ema"].append(ema(pose.keypoints))
        out["one_euro"].append(smoother.update(1, pose, t).keypoints)
    return {k: np.array(v) * [W, H] for k, v in out.items()}


def run_real(cfg: Config, args) -> tuple[dict, list[float], np.ndarray]:
    est, smoother, ema = PoseEstimator(cfg), PoseSmoother(cfg), ExponentialSmoother(0.4)
    out = {"raw": [], "ema": [], "one_euro": []}
    times, conf = [], []
    for frame in frames(args):
        t0 = time.perf_counter()
        poses = est.process(frame)
        times.append((time.perf_counter() - t0) * 1000)
        if not poses:
            continue
        pose = max(poses, key=lambda p: (p.bbox() or (0, 0, 0, 0))[3])  # the nearest person
        h, w = frame.image.shape[:2]
        scale = np.array([w, h], np.float32)
        bad = pose.confidence < cfg.pose.min_keypoint_conf
        raw = np.where(bad[:, None], np.nan, pose.keypoints)
        out["raw"].append(raw * scale)
        out["ema"].append(ema(pose.keypoints) * scale)
        out["one_euro"].append(smoother.update(1, pose, frame.timestamp).keypoints * scale)
        conf.append(pose.confidence)
    est.close()
    return {k: np.array(v) for k, v in out.items()}, times, np.array(conf)


def report(tracks: dict) -> None:
    wrist = KP["right_wrist"]
    still = [KP["nose"], KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    reference = tracks.get("truth")
    if reference is None:
        reference = zero_phase_smooth(tracks["raw"])
    print(f"{'filter':<10} {'jitter px/frame':>16} {'wrist lag (frames)':>19} "
          + ("{:>13} {:>13} {:>13}".format("err all px", "err torso px", "err wrist px") if "truth" in tracks else ""))
    for name in ("raw", "ema", "one_euro"):
        tr = tracks[name]
        line = f"{name:<10} {jitter(tr):16.3f} {lag_frames(reference[:, wrist], tr[:, wrist]):19d} "
        if "truth" in tracks:
            e = np.linalg.norm(tr - tracks["truth"], axis=-1)
            line += f"{np.nanmean(e):13.2f} {np.nanmean(e[:, still]):13.2f} {np.nanmean(e[:, wrist]):13.2f}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    cfg = Config()
    if args.mock:
        report(run_mock(cfg, args.seconds))
        return
    tracks, times, conf = run_real(cfg, args)
    print(f"pose inference ms: {stats(times)}")
    if len(conf) == 0:
        print("no person detected")
        return
    report(tracks)
    missing = (conf < cfg.pose.min_keypoint_conf).mean(axis=0)
    print("missing (below threshold) per keypoint:")
    print("  " + "  ".join(f"{n}={m:.0%}" for n, m in zip(KP_NAMES, missing)))


if __name__ == "__main__":
    main()
