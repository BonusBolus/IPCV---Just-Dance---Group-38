"""Shared helpers for the evaluation scripts."""
from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

import cv2
import numpy as np

from core.camera import Camera
from core.config import CameraConfig
from core.types import FrameData


def add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--video", help="recorded video (default: live webcam)")
    p.add_argument("--camera", default="0")
    p.add_argument("--seconds", type=float, default=20.0, help="how long to evaluate")


def frames(args) -> Iterator[FrameData]:
    """Frames from a video file (as fast as possible, timestamps from the file's fps) or the webcam."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while i / fps < args.seconds:
            ok, img = cap.read()
            if not ok:
                break
            yield FrameData(cv2.flip(img, 1), i / fps, i)
            i += 1
        cap.release()
        return
    with Camera(CameraConfig(source=args.camera)) as cam:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            f = cam.read(2.0)
            if f is not None:
                yield f


def stats(values) -> str:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if len(v) == 0:
        return "n/a"
    return f"mean {v.mean():7.2f}  p50 {np.percentile(v, 50):7.2f}  p95 {np.percentile(v, 95):7.2f}  (n={len(v)})"


def jitter(track: np.ndarray) -> float:
    """Mean magnitude of the second difference (T, ..., 2) -> 'acceleration noise' per frame.
    Smooth motion has a small second difference; frame-to-frame jitter makes it large."""
    if len(track) < 3:
        return float("nan")
    d2 = track[2:] - 2 * track[1:-1] + track[:-2]
    n = np.linalg.norm(d2, axis=-1)
    return float(np.nanmean(n))


def zero_phase_smooth(track: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred (non-causal) moving average: removes noise without adding delay. Used as the
    reference trajectory for lag measurements on real footage, where there is no ground truth."""
    kernel = np.ones(window) / window
    flat = track.reshape(len(track), -1)
    out = np.stack([np.convolve(np.nan_to_num(flat[:, j]), kernel, mode="same") for j in range(flat.shape[1])], 1)
    return out.reshape(track.shape)


def lag_frames(reference: np.ndarray, filtered: np.ndarray, max_lag: int = 10) -> int:
    """Delay k (frames) that minimises |filtered[t] - reference[t - k]|."""
    errs = []
    for k in range(max_lag + 1):
        d = filtered[k:] - reference[: len(reference) - k]
        errs.append(np.nanmean(np.abs(d)) if len(d) else np.inf)
    return int(np.nanargmin(errs))
