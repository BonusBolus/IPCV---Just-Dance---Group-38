"""The big heads are taken from the 1080p camera image (Frame.detail): the same place and size as before,
but sharper than a blown-up 720p head."""
import unittest

import cv2
import numpy as np

from camera import game_copy
from face_tracking.head_filter import enlarge_heads

FACE = {"location": (640, 360), "size": (120, 150)}


def camera_image():
    """1080p image with a fine stripe pattern (like hair or eyelashes) and a red dot in the middle of the face."""
    full = np.full((1080, 1920, 3), 90, np.uint8)
    full[:, ::2] = 200                                           # 1 px stripes: too fine for 720p
    cv2.circle(full, (960, 540), 6, (0, 0, 255), -1)
    return full


def sharpness(image):
    face = image[310:410, 590:690].astype(np.float32)
    return float(np.abs(np.diff(face, axis=1)).mean())


class SharpHeadTests(unittest.TestCase):
    def setUp(self):
        self.full = camera_image()
        self.small = game_copy(self.full, (1280, 720))

    def test_head_from_1080p_is_sharper(self):
        blown_up = enlarge_heads(self.small.copy(), [FACE], 1.35)
        sharp = enlarge_heads(self.small.copy(), [FACE], 1.35, detail=self.full)
        self.assertGreater(sharpness(sharp), 2 * sharpness(blown_up))

    def test_same_place_and_size(self):
        blown_up = enlarge_heads(self.small.copy(), [FACE], 1.35)
        sharp = enlarge_heads(self.small.copy(), [FACE], 1.35, detail=self.full)
        red = lambda image: np.argwhere((image[..., 2] > 150) & (image[..., 0] < 80))
        a, b = red(blown_up), red(sharp)
        np.testing.assert_allclose(a.mean(axis=0), b.mean(axis=0), atol=1.0)      # the dot stays in the middle
        self.assertLess(abs(len(a) - len(b)) / len(a), 0.25)                        # and keeps its size

    def test_outside_the_head_nothing_changes(self):
        sharp = enlarge_heads(self.small.copy(), [FACE], 1.35, detail=self.full)
        np.testing.assert_array_equal(sharp[:, :500], self.small[:, :500])

    def test_without_detail_the_same_as_before(self):
        image = self.small.copy()
        self.assertIs(enlarge_heads(image, [FACE], 1.35, detail=self.small), image)
        np.testing.assert_array_equal(enlarge_heads(self.small.copy(), [FACE], 1.35, detail=self.small),
                                      enlarge_heads(self.small.copy(), [FACE], 1.35))


if __name__ == "__main__":
    unittest.main()
