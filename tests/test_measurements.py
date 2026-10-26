"""The measurements for the report: the summary and the table use the right frames and numbers."""
import tempfile
import time
import unittest
from types import SimpleNamespace

from game_logic.measurements import WARMUP_FRAMES, Measurements, NoMeasurements, timed


CLOCK = [100.0]          # the fake time.perf_counter(): only goes forward, also across calls


def run_fake_game(measurements, frames, frame_ms, state="playing", slow_start_ms=None):
    """Fake frames: every stage takes a known time; the shown time goes up frame_ms per frame."""
    clock = CLOCK
    original = time.perf_counter
    time.perf_counter = lambda: clock[0]
    try:
        for index in range(frames):
            frame = SimpleNamespace(index=index, time=clock[0], capture_ms=1.0)
            measurements.start_frame(frame, state)
            measurements.add("t2_pose", slow_start_ms if slow_start_ms and index == 0 else 9.0)
            measurements.add("t1_faces", 5.0)
            measurements.count("people", 2)
            clock[0] += frame_ms / 1000
            measurements.frame_shown(index, 0.5)
    finally:
        time.perf_counter = original


class MeasurementsTests(unittest.TestCase):
    def test_fps_latency_and_stage_times(self):
        with tempfile.TemporaryDirectory() as folder:
            m = Measurements(folder, program_start=CLOCK[0] - 10.0)
            run_fake_game(m, 200, 40.0)
            summary = m.summary()
        self.assertIn("FPS: mean 25.0", summary)
        self.assertIn("| t2_pose | 9.0 |", summary)
        self.assertIn("| latency | 40.0 |", summary)            # captured at the start, shown 40 ms later
        self.assertIn("people per frame: mean 2.00; frames with 2 or more: 100%", summary)

    def test_the_warm_up_is_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            m = Measurements(folder)
            run_fake_game(m, 200, 40.0, slow_start_ms=1000.0)  # the first frame: OpenVINO warm-up
            summary = m.summary()
        self.assertIn("| t2_pose | 9.0 | 9.0 | 9.0 | 9.0 |", summary)
        self.assertIn(f"the first {WARMUP_FRAMES} frames are left out", summary)

    def test_only_playing_frames_when_there_are_some(self):
        with tempfile.TemporaryDirectory() as folder:
            m = Measurements(folder)
            run_fake_game(m, 100, 40.0, state="song_select")
            run_fake_game(m, 100, 50.0, state="playing")
            self.assertTrue(all(row["state"] == "playing" for row in m.playing_rows()))
            self.assertIn("FPS: mean 20.0", m.summary())

    def test_the_files_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            m = Measurements(folder)
            run_fake_game(m, 100, 40.0)
            out = m.finish()
            self.assertTrue((out / "frames.csv").exists())
            table = (out / "table.tex").read_text()
        self.assertIn("Pose estimation (T2) & 9.0", table)
        self.assertIn("Face tracking (T1) & 5.0", table)

    def test_without_measure_nothing_happens(self):
        m = NoMeasurements()
        with m.stage("t2_pose"):
            pass
        m.add("t1_faces", 3.0)
        m.finish()

    def test_timed(self):
        result, ms = timed(lambda x: x * 2, 21)
        self.assertEqual(result, 42)
        self.assertGreaterEqual(ms, 0.0)


if __name__ == "__main__":
    unittest.main()
