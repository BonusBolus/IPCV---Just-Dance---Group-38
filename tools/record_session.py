"""Record raw webcam footage for reproducible testing and evaluation.

Replay it in the game with:  python main.py --video recordings/crossing_01.mp4
Frames are stored UN-mirrored; the game mirrors them on replay exactly like live input.

Usage:
    python -m tools.record_session --out recordings/crossing_01.mp4 [--camera 0] [--seconds 30]
Press Q or ESC to stop.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from core.camera import Camera
from core.config import CameraConfig
from core.image_utils import put_text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=0, help="stop automatically after N seconds (0 = manual)")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = CameraConfig(source=args.camera, width=args.width, height=args.height, fps=args.fps, mirror=False)
    writer = None
    n = 0
    with Camera(cfg) as cam:
        t0 = time.perf_counter()
        while True:
            frame = cam.read(timeout=1.0)
            if frame is None:
                if cam.error:
                    break
                continue
            if writer is None:
                h, w = frame.image.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write {out}")
            writer.write(frame.image)
            n += 1
            elapsed = time.perf_counter() - t0

            preview = cv2.flip(frame.image, 1)
            put_text(preview, f"REC {elapsed:5.1f}s  ({cam.fps:4.1f} fps)  Q to stop", (20, 40), 0.8, (60, 60, 255))
            cv2.imshow("record", preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")) or (args.seconds and elapsed >= args.seconds):
                break
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {n} frames to {out}. Measured camera fps: {cam.fps:.1f} "
          f"(file is tagged {args.fps} fps; pass --fps to match if they differ)")


if __name__ == "__main__":
    main()
