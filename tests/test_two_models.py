"""The pose model and the segmentation model can run in one program.

With OpenVINO 2026.4 on an AMD Ryzen AI laptop, a second OpenVINO Core crashed the whole program
("segfault in libze_loader.so"), so the game crashed on its first frame. use_one_openvino_core() fixes that.
The test starts both models at the same time in two threads, like the first game frame, in a separate
process: a crash there fails the test instead of stopping all tests.
"""
import subprocess
import sys
import unittest
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = f"""
import sys, threading
sys.path.insert(0, {str(ROOT)!r})
import numpy as np
from pose_tracking.pose_processing import PoseEstimator
from background.segmentation import PersonSegmenter
pose = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)
seg = PersonSegmenter()
image = np.zeros((720, 1280, 3), np.uint8)
thread = threading.Thread(target=seg.find_people, args=(image,))
thread.start()
pose.process(image)
thread.join()
print("both models work")
"""


@unittest.skipUnless(config.USE_OPENVINO and config.YOLO_MODEL_PATH.exists() and config.SEGMENT_MODEL_PATH.exists(),
                     "needs both YOLO models and OpenVINO")
class TwoModelsTests(unittest.TestCase):
    def test_pose_and_segmentation_in_one_program(self):
        result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, f"crashed (exit code {result.returncode}):\n{result.stderr[-2000:]}")
        self.assertIn("both models work", result.stdout)


if __name__ == "__main__":
    unittest.main()
