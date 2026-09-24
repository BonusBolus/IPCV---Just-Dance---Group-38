"""Per-stage timing and frame-rate measurement (needed for every task's evaluation).

Usage:
    prof = Profiler()
    prof.tick()                     # once per frame
    with prof.section("pose"):
        ...
    prof.mean_ms("pose"), prof.fps

`open_csv(path)` logs every section time (long format: frame, section, ms) for the report.
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path


class Profiler:
    def __init__(self, window: int = 60):
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._frame_times: deque[float] = deque(maxlen=window)
        self._frame = 0
        self._csv_file = None
        self._csv = None

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self._samples[name].append(ms)
            if self._csv is not None:
                self._csv.writerow((self._frame, name, f"{ms:.3f}"))

    def tick(self) -> None:
        """Mark the start of a new frame."""
        self._frame += 1
        self._frame_times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def mean_ms(self, name: str) -> float:
        s = self._samples.get(name)
        return sum(s) / len(s) if s else 0.0

    def summary(self) -> list[str]:
        lines = [f"FPS {self.fps:5.1f}"]
        lines += [f"{name:<10} {self.mean_ms(name):6.1f} ms" for name in self._samples]
        return lines

    def open_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = path.open("w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(("frame", "section", "ms"))

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = self._csv = None
