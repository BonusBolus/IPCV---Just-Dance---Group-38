"""Webcam input for the game."""
import sys
import threading
import time
from dataclasses import dataclass
import cv2
import numpy as np

import config


@dataclass
class Frame:
    image: np.ndarray  # BGR image the game works with (at most GAME_SIZE)
    time: float        # capture time in seconds
    index: int         # frame number
    full: np.ndarray | None = None   # the camera image at full size (CAMERA_SIZE), for small details such as
                                     # faces; None when it is the same as `image`
    capture_ms: float = float("nan")  # time to decode and prepare this image (for the measurements)

    @property
    def detail(self):
        """The sharpest image there is: `full`, or `image` when the camera is not bigger than the game."""
        return self.image if self.full is None else self.full


def game_copy(image, size=None):
    """A copy of the camera image at most `size` (GAME_SIZE): the image itself when it is small enough."""
    max_width, max_height = size or config.GAME_SIZE
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


class Camera:
    """Reads frames from a webcam (or a video file for testing).

    The image is mirrored, so the players see themselves like in a mirror.

    A webcam is read in a background thread that keeps only the newest image. Without it, the
    images wait in the webcam's buffer when the game is slower than the camera, and the game
    shows an image of about 160 ms ago (the players see themselves late). A video file is read
    normally, so no frame is skipped.

    The webcam is asked for CAMERA_SIZE (1920x1080). The game works on a copy of at most GAME_SIZE
    (1280x720), so it is not slower; the full image stays in Frame.full for the faces. Mirroring and
    the copy are made in the background thread too.
    """

    def __init__(self, source=0, width=None, height=None, mirror=True):
        width, height = width or config.CAMERA_SIZE[0], height or config.CAMERA_SIZE[1]
        if isinstance(source, int) and sys.platform == "win32":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # opens a lot faster on Windows
        else:
            self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            # Many webcams only give 1280x720 at 30 fps as MJPG (compressed); without asking for it,
            # OpenCV often gets 640x480. Ask before the size. A webcam without MJPG keeps its own format.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
            ok = self.cap.grab()                      # waits for the camera
            capture_time = time.perf_counter()
            ok, image = self.cap.retrieve() if ok else (False, None)    # decodes the image
            images = self._prepare(image) if ok else None
            capture_ms = (time.perf_counter() - capture_time) * 1000
            with self.new_image:
                self.latest = (ok, images, capture_time, capture_ms)
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
                ok, images, capture_time, capture_ms = self.latest
                self.latest = None
        else:
            ok = self.cap.grab()
            capture_time = time.perf_counter()
            ok, image = self.cap.retrieve() if ok else (False, None)
            images = self._prepare(image) if ok else None
            capture_ms = (time.perf_counter() - capture_time) * 1000
        if not ok:
            return None
        image, full = images
        frame = Frame(image, capture_time, self.index, full, capture_ms)
        self.index += 1
        return frame

    def _prepare(self, image):
        """(game image, full image or None): mirrored, and the game copy at most GAME_SIZE."""
        if self.mirror:
            image = cv2.flip(image, 1)
        small = game_copy(image)
        return small, (None if small is image else image)

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
