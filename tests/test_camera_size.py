"""The camera image is read big (CAMERA_SIZE) for the faces, and the game works on a copy of at most
GAME_SIZE, so the game is not slower with a 1080p webcam."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera import Camera, game_copy


def write_video(path, width, height, frames=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    for _ in range(frames):
        image = np.zeros((height, width, 3), np.uint8)
        image[:, : width // 4] = (0, 0, 255)                     # red on the left
        writer.write(image)
    writer.release()


class GameCopyTests(unittest.TestCase):
    def test_1080p_becomes_720p(self):
        self.assertEqual(game_copy(np.zeros((1080, 1920, 3), np.uint8), (1280, 720)).shape, (720, 1280, 3))

    def test_small_images_are_not_changed(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        self.assertIs(game_copy(image, (1280, 720)), image)

    def test_the_shape_stays_the_same(self):
        self.assertEqual(game_copy(np.zeros((1440, 1920, 3), np.uint8), (1280, 720)).shape, (720, 960, 3))


class CameraTests(unittest.TestCase):
    def read_one(self, width, height):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "video.avi"
            write_video(path, width, height)
            camera = Camera(str(path))
            frame = camera.read()
            camera.release()
        return frame

    def test_1080p_video_gives_a_720p_game_image_and_the_full_image(self):
        frame = self.read_one(1920, 1080)
        self.assertEqual(frame.image.shape[:2], (720, 1280))
        self.assertEqual(frame.full.shape[:2], (1080, 1920))
        self.assertIs(frame.detail, frame.full)
        # both are mirrored: the red part is on the right now
        self.assertGreater(frame.image[360, 1270, 2], 200)
        self.assertGreater(frame.full[540, 1910, 2], 200)
        self.assertLess(frame.full[540, 10, 2], 50)

    def test_720p_video_has_no_extra_full_image(self):
        frame = self.read_one(1280, 720)
        self.assertEqual(frame.image.shape[:2], (720, 1280))
        self.assertIsNone(frame.full)
        self.assertIs(frame.detail, frame.image)


if __name__ == "__main__":
    unittest.main()
