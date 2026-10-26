"""The room is replaced by the stage: only the players stay (bystanders disappear), the big head is not
cut off by the background, and the masks fit the image (YOLO's padding is removed)."""
import unittest

import cv2
import numpy as np

from background.segmentation import box_overlap, player_mask, remove_padding
from background.stage import Stage
from face_tracking.head_filter import enlarge_heads

H, W = 720, 1280


def person(x1, y1, x2, y2, scale=480 / 1280):
    """A person as find_people() gives it: the box in image pixels, and a small mask (480 x 270)."""
    mask = np.zeros((270, 480), np.float32)
    mask[int(y1 * scale):int(y2 * scale), int(x1 * scale):int(x2 * scale)] = 1.0
    return np.array([x1, y1, x2, y2], np.float32), mask


def people(*persons):
    return np.array([p[0] for p in persons]), np.array([p[1] for p in persons])


def player(x1, y1, x2, y2, visible=True):
    return {"visible": visible, "box": (x1, y1, x2 - x1, y2 - y1)}


class MaskTests(unittest.TestCase):
    def test_padding_is_removed(self):
        masks = np.zeros((1, 192, 320), np.float32)     # 720p at 320 px: 180 rows of image, 6 rows of padding above and below
        masks[0, 6:186] = 1.0
        out = remove_padding(masks, (720, 1280))
        self.assertEqual(out.shape, (1, 180, 320))
        self.assertTrue(np.all(out == 1.0))

    def test_box_overlap(self):
        self.assertEqual(box_overlap((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(box_overlap((0, 0, 10, 10), (5, 0, 15, 10)), 0.5)
        self.assertAlmostEqual(box_overlap((0, 0, 100, 100), (10, 10, 20, 20)), 1.0)   # small box inside

    def test_only_the_players_stay(self):
        found = people(person(100, 200, 300, 700), person(900, 200, 1100, 700), person(560, 100, 640, 300))
        mask = player_mask(found, [player(110, 210, 290, 690), player(910, 210, 1090, 690)], (H, W))
        self.assertGreater(mask[450, 200], 0.99)                # player 1
        self.assertGreater(mask[450, 1000], 0.99)               # player 2
        self.assertLess(mask[200, 600], 0.01)                   # the bystander in the back is gone
        self.assertLess(mask[100, 1200], 0.01)                  # the room

    def test_everyone_stays_when_no_player_is_seen(self):
        found = people(person(100, 200, 300, 700))
        mask = player_mask(found, [player(0, 0, 1, 1, visible=False)], (H, W))
        self.assertGreater(mask[450, 200], 0.99)                # not suddenly gone when the pose misses a frame

    def test_nobody_found(self):
        mask = player_mask((np.zeros((0, 4)), None), [], (H, W))
        self.assertEqual(mask.shape, (H, W))
        self.assertEqual(mask.max(), 0.0)


class StageTests(unittest.TestCase):
    def test_players_stay_and_the_room_becomes_the_stage(self):
        image = np.full((H, W, 3), 50, np.uint8)
        mask = np.zeros((H, W), np.float32)
        mask[200:700, 100:300] = 1.0
        stage = Stage()
        out = stage.place_players(image.copy(), mask, 2.0)
        np.testing.assert_array_equal(out[400, 200], (50, 50, 50))                # the player: unchanged
        np.testing.assert_array_equal(out[50, 900], stage.backdrop(W, H, 2.0)[50, 900])   # the room: stage

    def test_the_stage_moves_slowly(self):
        stage = Stage()
        a, b = stage.backdrop(W, H, 1.0).astype(int), stage.backdrop(W, H, 1.04).astype(int)
        self.assertGreater(np.abs(stage.backdrop(W, H, 10.0).astype(int) - a).max(), 0)   # it moves
        self.assertLessEqual(np.abs(b - a).max(), 12)                                    # but not jumpy


class BigHeadTests(unittest.TestCase):
    def test_the_mask_grows_with_the_head(self):
        image = np.full((H, W, 3), 80, np.uint8)
        mask = np.zeros((H, W), np.float32)
        cv2.ellipse(mask, (640, 300), (50, 65), 0, 0, 360, 1.0, -1)              # the head
        before = mask.copy()
        enlarge_heads(image, [{"location": (640, 300), "size": (100, 130)}], 1.35, mask=mask)
        self.assertGreater(mask.sum(), 1.3 * before.sum())                       # the head in the mask is bigger
        self.assertGreater(mask[300, 640 + 52], 0.95)                            # just outside the old head: player
        self.assertEqual(before[300, 640 + 52], 0.0)


if __name__ == "__main__":
    unittest.main()
