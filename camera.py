"""Webcam input for the game."""
import sys
import threading
import time
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR image as read by OpenCV
    time: float        # capture time in seconds
    index: int         # frame number


class Camera:
    """Reads frames from a webcam (or a video file for testing).

    The image is mirrored, so the players see themselves like in a mirror.

    A webcam is read in a background thread that keeps only the newest image. Without it, the
    images wait in the webcam's buffer when the game is slower than the camera, and the game
    shows an image of about 160 ms ago (the players see themselves late). A video file is read
    normally, so no frame is skipped.
    """

    def __init__(self, source=0, width=1280, height=720, mirror=True):
        if isinstance(source, int) and sys.platform == "win32":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # opens a lot faster on Windows
        else:
            self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mirror = mirror
        self.index = 0

        self.live = isinstance(source, int)
        if self.live:
            self.latest = None              # (ok, image, capture time) of the newest image
            self.new_image = threading.Condition()
            self.running = True
            self.thread = threading.Thread(target=self._read_newest, daemon=True)
            self.thread.start()

    def _read_newest(self):
        """Background thread: read the webcam all the time, keep only the newest image."""
        while self.running:
            ok, image = self.cap.read()
            with self.new_image:
                self.latest = (ok, image, time.perf_counter())
                self.new_image.notify()
            if not ok:
                break

    def read(self):
        if self.live:
            with self.new_image:
                while self.latest is None:      # wait for an image that the game did not get yet
                    if not self.thread.is_alive():
                        return None             # the webcam stopped
                    self.new_image.wait(timeout=0.5)
                ok, image, capture_time = self.latest
                self.latest = None
        else:
            ok, image = self.cap.read()
            capture_time = time.perf_counter()
        if not ok:
            return None
        if self.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image, capture_time, self.index)
        self.index += 1
        return frame

    def release(self):
        if self.live:
            self.running = False
            self.thread.join(timeout=1.0)
        self.cap.release()


class FPSCounter:
    """Frame rate averaged over the last `n` frames."""

    def __init__(self, n=30):
        self.n = n
        self.times = []

    def update(self):
        self.times.append(time.perf_counter())
        self.times = self.times[-self.n:]

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])
