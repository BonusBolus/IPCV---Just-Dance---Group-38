"""--record: the webcam and the game become MP4 videos (H.264) at real speed, the game ones with the music."""
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import config
from game_logic.recorder import Recorder

SONG = Path(config.CHOREOGRAPHY_DIR) / "song_1.ogg"


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and SONG.exists(), "needs ffmpeg and a song")
class RecorderTests(unittest.TestCase):
    def test_three_videos_at_real_speed_with_music(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(folder)
            webcam = np.full((720, 1280, 3), 60, np.uint8)
            game = np.full((720, 1280, 3), 160, np.uint8)
            start = time.perf_counter()
            k = 0
            while time.perf_counter() - start < 3.0:
                if k == 25:
                    recorder.song_started(SONG)              # the song starts after 1 s
                recorder.add(webcam, game)
                k += 1
                time.sleep(0.08 if k % 10 == 0 else 0.04)    # now and then a slow frame: it is repeated
            out = recorder.finish()
            files = {name: probe(Path(out) / f"{name}.mp4") for name in ("webcam", "game", "side_by_side")}
            self.assertFalse((Path(out) / "webcam.avi").exists())       # the raw files are removed

        for name, info in files.items():
            video = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual(video["codec_name"], "h264")
            self.assertAlmostEqual(float(info["format"]["duration"]), 3.0, delta=0.3)   # real speed
        side = next(s for s in files["side_by_side"]["streams"] if s["codec_type"] == "video")
        self.assertEqual((side["width"], side["height"]), (1920, 540))
        self.assertFalse(any(s["codec_type"] == "audio" for s in files["webcam"]["streams"]))
        self.assertTrue(any(s["codec_type"] == "audio" for s in files["game"]["streams"]))
        self.assertTrue(any(s["codec_type"] == "audio" for s in files["side_by_side"]["streams"]))

    def test_nothing_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(Recorder(folder).finish())


if __name__ == "__main__":
    unittest.main()
