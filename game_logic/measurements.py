"""Measurements for the report: frame rate, the time of every stage, latency, CPU, memory and start-up time.

Start the game with   python main.py --measure            (and optionally --duration 120 to stop after 120 s)
When the game closes, a folder measurements/<date_time>/ has:
    frames.csv      one row per shown frame: the time of every stage (ms), state, people, faces, ...
    resources.csv   once per second: CPU use, memory, CPU temperature
    summary.md      mean / median / 95th percentile per stage, FPS (mean and low), latency, start-up time
    table.tex       the rows for the timing table of the report

Stages (ms), as in the report:
    capture       camera thread: decode the camera image, mirror it, make the 1280x720 copy
    t1_faces      T1 face detection + smoothing (face thread, runs at the same time as T2)
    t1_effects    T1 big heads, crown and player labels
    t2_pose       T2 pose estimation (YOLO)
    t3_identity   T3 players from the keypoints (tracker) and faces given to the players
    t4_logic      T4 the state's update(): grading, gestures, star, state changes
    segmentation  person segmentation (segmentation thread, at the same time as T2)
    t5_render     T5 stage behind the players, edge glow and the state's screen
    display       showing the image in the window (cv2.imshow)
    wait_threads  how long step() waited for the face and segmentation threads
    step          all of step() for this frame (the parallel parts count once)
    frame_time    time between two shown frames (1000 / FPS)
    latency       from the moment the camera image arrived to the moment it was shown
Times are wall-clock times measured with time.perf_counter(). The parallel stages (t1_faces, segmentation)
run at the same time as t2_pose, so the stages do not add up to the frame time.
"""
import csv
import platform
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np

STAGES = ["capture", "t1_faces", "t1_effects", "t2_pose", "t3_identity", "t4_logic", "segmentation",
          "t5_render", "display", "wait_threads", "step", "frame_time", "latency"]
COUNTS = ["people", "faces", "players_visible", "heads"]
FREEZE_MS = 200.0                     # a frame slower than this counts as a freeze
WARMUP_FRAMES = 25                    # the first second is left out of the statistics: OpenVINO prepares the
                                      # models on their first use (that time is in the start-up time)

# rows of the report's timing table: (label, stages that are added up)
REPORT_ROWS = [("Capture", ["capture"]),
               ("Face tracking (T1)", ["t1_faces", "t1_effects"]),
               ("Pose estimation (T2)", ["t2_pose"]),
               ("Identity tracking (T3)", ["t3_identity"]),
               ("Interaction logic (T4)", ["t4_logic"]),
               ("Scene rendering (T5)", ["segmentation", "t5_render", "display"]),
               ("End-to-end (camera to screen)", ["latency"])]


def timed(function, *args):
    """function(*args), and how long it took in ms. Used for the work in the face and segmentation threads."""
    start = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter() - start) * 1000


class NoMeasurements:
    """Used when the game runs without --measure: every call does nothing."""
    enabled = False

    def stage(self, name):
        return nullcontext()

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Measurements:
    enabled = True

    def __init__(self, folder=None, program_start=None):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder = Path(folder or Path(__file__).resolve().parent.parent / "measurements" / stamp)
        self.program_start = program_start or time.perf_counter()
        self.game_ready = None          # Game.__init__ done (models loaded)
        self.first_frame = None         # the first image shown
        self.frames = {}                # frame index -> row (dict)
        self.shown = []                 # rows of the frames that were shown, in order
        self.resources = []
        self.last_shown = None
        self.last_sample = 0.0
        try:
            import psutil
            self.process = psutil.Process()
            self.process.cpu_percent(None)          # the first call only starts the measurement
            psutil.cpu_percent(None)
            self.psutil = psutil
        except ImportError:
            self.psutil = None

    # --- per frame -------------------------------------------------------------------------------
    def start_frame(self, frame, state):
        row = {"index": frame.index, "state": state, "capture": getattr(frame, "capture_ms", float("nan")),
               "captured_at": frame.time}
        self.frames[frame.index] = row
        self.current = row
        return row

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000)

    def add(self, name, ms):
        self.current[name] = self.current.get(name, 0.0) + ms

    def count(self, name, value):
        self.current[name] = value

    def frame_shown(self, index, display_ms):
        """Called by the main loop right after the image of frame `index` was given to the window."""
        now = time.perf_counter()
        row = self.frames.pop(index, None)
        if row is None:
            return
        row["display"] = display_ms
        row["latency"] = (now - row["captured_at"]) * 1000
        row["frame_time"] = (now - self.last_shown) * 1000 if self.last_shown is not None else float("nan")
        row["shown_at"] = now - self.program_start
        self.last_shown = now
        if self.first_frame is None:
            self.first_frame = now
        self.shown.append(row)
        if now - self.last_sample >= 1.0:
            self.sample_resources(now)

    def sample_resources(self, now):
        self.last_sample = now
        if self.psutil is None:
            return
        temps = self.psutil.sensors_temperatures() if hasattr(self.psutil, "sensors_temperatures") else {}
        cpu_temp = next((t.current for name in ("k10temp", "coretemp", "cpu_thermal", "acpitz")
                         for t in temps.get(name, [])[:1]), float("nan"))
        self.resources.append({
            "time": round(now - self.program_start, 1),
            "process_cpu_percent": self.process.cpu_percent(None),      # can be above 100: more cores
            "system_cpu_percent": self.psutil.cpu_percent(None),
            "memory_mb": round(self.process.memory_info().rss / 2**20),
            "cpu_temp_c": cpu_temp,
        })

    # --- at the end ------------------------------------------------------------------------------
    def finish(self, note=""):
        if not self.shown:
            print("measurements: no frames were shown, nothing written")
            return None
        self.folder.mkdir(parents=True, exist_ok=True)
        columns = ["index", "shown_at", "state"] + STAGES + COUNTS
        with open(self.folder / "frames.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self.shown:
                writer.writerow({key: (round(value, 3) if isinstance(value, float) else value)
                                 for key, value in row.items()})
        if self.resources:
            with open(self.folder / "resources.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.resources[0]))
                writer.writeheader()
                writer.writerows(self.resources)
        summary = self.summary(note)
        (self.folder / "summary.md").write_text(summary)
        (self.folder / "table.tex").write_text(self.latex_table())
        print(summary)
        print(f"measurements written to {self.folder}")
        return self.folder

    def values(self, stage, rows=None):
        rows = self.shown if rows is None else rows
        out = np.array([row.get(stage, np.nan) for row in rows], dtype=float)
        return out[np.isfinite(out)]

    def measured_rows(self):
        """All shown frames after the warm-up."""
        return self.shown[WARMUP_FRAMES:] if len(self.shown) > 2 * WARMUP_FRAMES else self.shown

    def playing_rows(self):
        """The frames of normal gameplay (state Playing), or all frames when the song was never played."""
        rows = [row for row in self.measured_rows() if row["state"] == "playing"]
        return rows if len(rows) >= 10 else self.measured_rows()

    def summary(self, note=""):
        rows = self.playing_rows()
        which = ("Playing" if all(row["state"] == "playing" for row in rows)
                 else "all states (Playing was not reached)")
        frame_times = self.values("frame_time", rows)
        fps = 1000.0 / frame_times if len(frame_times) else np.array([])
        duration = self.shown[-1]["shown_at"] - self.shown[0]["shown_at"]
        lines = [f"# Measurements {self.folder.name}", ""]
        if note:
            lines += [note, ""]
        lines += [f"- system: {self.system_info()}"]
        lines += [f"- frames shown: {len(self.shown)} in {duration:.1f} s; statistics below: {which}, {len(rows)} frames "
                  f"(the first {WARMUP_FRAMES} frames are left out: warm-up)",
                  f"- start-up: game ready after {self._since_start(self.game_ready)}, "
                  f"first image after {self._since_start(self.first_frame)}"]
        if len(fps):
            lines.append(f"- FPS: mean {len(frame_times) / (frame_times.sum() / 1000):.1f}, median {np.median(fps):.1f}, "
                         f"5th percentile {np.percentile(fps, 5):.1f}, "
                         f"1% low {np.mean(np.sort(fps)[:max(1, len(fps) // 100)]):.1f}")
        all_frame_times = self.values("frame_time", self.measured_rows())
        freezes = all_frame_times[all_frame_times > FREEZE_MS]
        lines.append(f"- freezes (frames slower than {FREEZE_MS:.0f} ms): {len(freezes)}"
                     + (f", longest {freezes.max():.0f} ms" if len(freezes) else ""))
        if self.resources:
            cpu = np.array([r["process_cpu_percent"] for r in self.resources])
            mem = np.array([r["memory_mb"] for r in self.resources])
            temp = np.array([r["cpu_temp_c"] for r in self.resources], dtype=float)
            lines.append(f"- CPU (this program, 100% = one core): mean {cpu.mean():.0f}%, max {cpu.max():.0f}%; "
                         f"memory {mem.min():.0f}-{mem.max():.0f} MB"
                         + (f"; CPU temperature {np.nanmin(temp):.0f}-{np.nanmax(temp):.0f} C"
                            if np.isfinite(temp).any() else ""))
        for name in COUNTS:
            counts = self.values(name, rows)
            if len(counts):
                lines.append(f"- {name} per frame: mean {counts.mean():.2f}; "
                             f"frames with 2 or more: {100 * np.mean(counts >= 2):.0f}%")
        lines += ["", "| stage | mean ms | median ms | p95 ms | max ms |", "|---|---|---|---|---|"]
        for stage in STAGES:
            v = self.values(stage, rows)
            if len(v):
                lines.append(f"| {stage} | {v.mean():.1f} | {np.median(v):.1f} | {np.percentile(v, 95):.1f} | {v.max():.1f} |")
        lines += ["", "t1_faces and segmentation run at the same time as t2_pose: the stages do not add up to frame_time.",
                  "latency starts when the camera image arrives in the program; the camera itself and the screen add "
                  "some more (not measured here).", ""]
        return "\n".join(lines)

    def latex_table(self):
        rows = self.playing_rows()
        lines = ["% generated by game_logic/measurements.py: mean time per stage during normal gameplay (ms)",
                 "% T1 face detection and the segmentation (in T5) run at the same time as T2, so the rows",
                 "% do not add up to the frame time"]
        for label, stages in REPORT_ROWS:
            per_frame = np.nansum([[row.get(s, np.nan) for s in stages] for row in rows], axis=1)
            lines.append(f"{label} & {np.mean(per_frame):.1f} \\\\")
        frame_times = self.values("frame_time", rows)
        if len(frame_times):
            fps = len(frame_times) / (frame_times.sum() / 1000)
            lines.append(f"% end-to-end frame rate: {fps:.1f} FPS; latency: {np.mean(self.values('latency', rows)):.0f} ms")
        return "\n".join(lines) + "\n"

    def system_info(self):
        """The laptop and the versions, for "report the PC/laptop tech specs"."""
        cpu = platform.processor() or platform.machine()
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        parts = [f"CPU {cpu}"]
        if self.psutil is not None:
            parts.append(f"{self.psutil.cpu_count(logical=False)} cores / {self.psutil.cpu_count()} threads, "
                         f"{self.psutil.virtual_memory().total / 2**30:.0f} GB memory")
        parts.append(f"{platform.system()} {platform.release()}, Python {platform.python_version()}")
        try:
            import cv2
            import openvino
            parts.append(f"OpenCV {cv2.__version__}, OpenVINO {openvino.__version__.split('-')[0]}")
        except ImportError:
            pass
        return "; ".join(parts)

    def _since_start(self, moment):
        return "-" if moment is None else f"{moment - self.program_start:.1f} s"
