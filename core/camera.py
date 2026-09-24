"""Threaded webcam / video-file capture.

The capture thread keeps only the newest frame, so slow processing never builds up latency:
the game always works on the most recent image. A video file can replace the webcam
("replay mode"), which makes testing and evaluation reproducible.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from core.config import CameraConfig
from core.types import FrameData

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.is_file = False
        self.error: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._cond = threading.Condition()
        self._latest: FrameData | None = None
        self._index = 0
        self._last_read_index = -1
        self._frame_times: deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> Camera:
        self._cap = self._open()
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera opened: %s (%dx%d)", self.cfg.source, w, h)
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> Camera:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ public API
    def read(self, timeout: float = 1.0) -> FrameData | None:
        """Block until a frame newer than the previous read arrives. None on timeout/stop."""
        with self._cond:
            self._cond.wait_for(
                lambda: not self._running
                or (self._latest is not None and self._latest.index != self._last_read_index),
                timeout=timeout,
            )
            frame = self._latest
            if frame is None or frame.index == self._last_read_index:
                return None
            self._last_read_index = frame.index
            return frame

    @property
    def fps(self) -> float:
        """Measured capture frame rate."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------------ internals
    def _open(self) -> cv2.VideoCapture:
        src = self.cfg.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, int):
            cap = None
            if sys.platform.startswith("win"):
                # DirectShow opens much faster than the default MSMF backend on Windows.
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            if cap is None:
                cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open webcam {src}. Is it in use? Try --camera 1.")
            # MJPG allows 720p at 30 fps on most USB/laptop cameras (raw YUY2 often caps at ~10).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        path = Path(src)
        if not path.exists():
            raise RuntimeError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self.is_file = True
        return cap

    def _run(self) -> None:
        file_dt = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.is_file else 0.0
        next_t = time.perf_counter()
        failures = 0
        while self._running:
            ok, img = self._cap.read()
            if not ok:
                if self.is_file and self.cfg.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_file:
                    self.error = "End of video file"
                    break
                failures += 1
                if failures > 100:
                    self.error = "Camera stopped delivering frames"
                    break
                time.sleep(0.01)
                continue
            failures = 0

            if self.is_file:  # replay at the recorded speed instead of as fast as possible
                next_t += file_dt
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

            if self.cfg.mirror:
                img = cv2.flip(img, 1)
            now = time.perf_counter()
            with self._cond:
                self._latest = FrameData(image=img, timestamp=now, index=self._index)
                self._index += 1
                self._frame_times.append(now)
                self._cond.notify_all()

        if self.error:
            log.error(self.error)
        self._running = False
        with self._cond:
            self._cond.notify_all()
