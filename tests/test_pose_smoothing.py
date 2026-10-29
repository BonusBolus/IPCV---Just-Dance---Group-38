"""The PoseEstimator smooths every person with the SAME person of the last frame, also when YOLO lists the
people in another order, and it does not keep showing people who are gone."""
import unittest
from unittest.mock import patch

import numpy as np

with patch("pose_tracking.pose_processing.load_pose_model", lambda: None):
    import pose_tracking.pose_processing as pose_processing
    from pose_tracking.pose_processing import PoseEstimator
MAX_MISSED_FRAMES = getattr(pose_processing, "MAX_MISSED_FRAMES", 3)


def person(x, y=100.0, height=300.0, confidence=0.9):
    """17 keypoints of a standing person around (x, y..y + height)."""
    kp = np.zeros((17, 3), np.float32)
    kp[:, 0] = x + np.linspace(-20, 20, 17)
    kp[:, 1] = y + np.linspace(0, height, 17)
    kp[:, 2] = confidence
    return kp


def estimator_with(frames):
    """A PoseEstimator whose YOLO returns the given people (list of (N, 17, 3) arrays), one per frame."""
    with patch("pose_tracking.pose_processing.load_pose_model", lambda: None):
        estimator = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)
    queue = list(frames)
    estimator._estimate_keypoints = lambda frame: queue.pop(0)
    return estimator


FRAME = np.zeros((10, 10, 3), np.uint8)


class PoseSmoothingTests(unittest.TestCase):
    def test_yolo_swaps_the_order_of_two_people(self):
        a, b = person(300), person(900)
        est = estimator_with([np.stack([a, b]), np.stack([b, a])])     # the second frame: order swapped
        est.process(FRAME)
        keypoints, motion = est.process(FRAME)
        # each row is still exactly its own person: no 30% of the other player blended in
        np.testing.assert_allclose(keypoints[0, :, 0], b[:, 0], atol=1e-4)
        np.testing.assert_allclose(keypoints[1, :, 0], a[:, 0], atol=1e-4)
        self.assertLess(np.abs(motion).max(), 1e-4)                     # nobody moved: no fake 600 px jump

    def test_smoothing_and_motion_follow_the_person(self):
        a, b = person(300), person(900)
        est = estimator_with([np.stack([a, b]), np.stack([person(910), person(300)])])   # b moved 10 px, swapped
        est.process(FRAME)
        keypoints, motion = est.process(FRAME)
        np.testing.assert_allclose(keypoints[0, :, 0], b[:, 0] + 7.0, atol=1e-4)        # 0.7 x 10 px
        np.testing.assert_allclose(motion[0, :, 0], 0.5 * 7.0, atol=1e-4)   # smoothed (0.5) with the first frame's 0
        np.testing.assert_allclose(motion[1, :, 0], 0.0, atol=1e-4)

    def test_missing_keypoints_come_from_the_same_person(self):
        a, b = person(300), person(900)
        b_unsure = b.copy()
        b_unsure[9, 2] = 0.0                                            # b's left wrist not seen
        est = estimator_with([np.stack([a, b]), np.stack([b_unsure, a])])
        est.process(FRAME)
        keypoints, _ = est.process(FRAME)
        self.assertAlmostEqual(float(keypoints[0, 9, 0]), float(b[9, 0]), places=3)   # not a's wrist

    def test_a_person_who_joins_does_not_stop_the_smoothing_of_the_others(self):
        a = person(300)
        est = estimator_with([np.stack([a]), np.stack([person(310), person(900)])])
        est.process(FRAME)
        keypoints, motion = est.process(FRAME)
        np.testing.assert_allclose(keypoints[0, :, 0], a[:, 0] + 7.0, atol=1e-4)  # still smoothed
        np.testing.assert_allclose(keypoints[1, :, 0], person(900)[:, 0], atol=1e-4)  # new: as it is
        self.assertLess(np.abs(motion[1]).max(), 1e-4)                   # new: no motion yet

    def test_people_who_are_gone_are_forgotten_after_a_short_miss(self):
        empty = np.empty((0, 17, 3), np.float32)
        est = estimator_with([np.stack([person(300)])] + [empty] * (MAX_MISSED_FRAMES + 2))
        est.process(FRAME)
        for _ in range(MAX_MISSED_FRAMES):
            self.assertEqual(len(est.process(FRAME)[0]), 1)              # a short miss: still there
        self.assertEqual(len(est.process(FRAME)[0]), 0)                  # then gone (no ghost)
        self.assertEqual(len(est.process(FRAME)[0]), 0)

    def test_someone_far_away_is_a_new_person(self):
        est = estimator_with([np.stack([person(300)]), np.stack([person(1100)])])   # 800 px: another person
        est.process(FRAME)
        keypoints, motion = est.process(FRAME)
        np.testing.assert_allclose(keypoints[0, :, 0], person(1100)[:, 0], atol=1e-4)
        self.assertLess(np.abs(motion).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
