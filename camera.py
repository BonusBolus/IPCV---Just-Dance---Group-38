"""Webcam input for the game."""
import sys
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

    def read(self):
        ok, image = self.cap.read()
        if not ok:
            return None
        if self.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image, time.perf_counter(), self.index)
        self.index += 1
        return frame

    def release(self):
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
