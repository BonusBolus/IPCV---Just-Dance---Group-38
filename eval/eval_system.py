"""Task 5 evaluation: end-to-end frame rate and per-stage cost.

    python -m eval.eval_system --csv logs/run.csv     # summarise a game run
                                                      # (python main.py --profile-csv logs/run.csv)
    python -m eval.eval_system [--video f.mp4]        # run the full pipeline headless and measure
    python -m eval.eval_system --mock                 # rendering/gameplay cost without CV models

Reports per section: mean / median / 95th percentile in ms, the share of the frame budget,
and the resulting processing frame rate. The live frame rate is min(camera fps, processing fps).
"""
from __future__ import annotations

import argparse
import csv
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.config import Config
from core.types import FrameData
from eval.common import add_source_args, frames
from scene.state_machine import Phase


def summarise(rows: list[tuple[int, str, float]]) -> None:
    per = defaultdict(list)
    per_frame = defaultdict(float)
    for frame, section, ms in rows:
        per[section].append(ms)
        per_frame[frame] += ms
    total = np.array(list(per_frame.values()))
    print(f"{'section':<12}{'mean':>8}{'p50':>8}{'p95':>8}   share")
    for section, v in per.items():
        v = np.array(v)
        print(f"{section:<12}{v.mean():8.2f}{np.median(v):8.2f}{np.percentile(v, 95):8.2f}   "
              f"{v.sum() / total.sum():5.0%}")
    print(f"{'frame total':<12}{total.mean():8.2f}{np.median(total):8.2f}{np.percentile(total, 95):8.2f}")
    print(f"processing capacity: {1000 / total.mean():.1f} fps (mean), {1000 / np.percentile(total, 95):.1f} fps (p95)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(p)
    p.add_argument("--csv", help="profile CSV written by main.py --profile-csv")
    p.add_argument("--mock", action="store_true", help="mock players instead of the CV models")
    args = p.parse_args()

    if args.csv:
        with open(args.csv, newline="") as f:
            rows = [(int(r["frame"]), r["section"], float(r["ms"])) for r in csv.DictReader(f)]
        summarise(rows)
        return

    from main import App

    cfg = Config()
    cfg.game.enable_audio = False
    app = App(cfg, use_mock=args.mock, load_models=not args.mock)
    app.phases.go(Phase.PLAYING, time.perf_counter())
    app.song.play()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.csv"
        app.profiler.open_csv(path)
        source = frames(args) if not args.mock else (
            FrameData(np.full((720, 1280, 3), 90, np.uint8), i / 30, i) for i in range(int(args.seconds * 30)))
        for frame in source:
            app.process_frame(frame)
        app.close()
        with path.open(newline="") as f:
            rows = [(int(r["frame"]), r["section"], float(r["ms"])) for r in csv.DictReader(f)]
    summarise(rows)


if __name__ == "__main__":
    main()
