"""Technical evaluation of task 3: multiplayer detection and identity tracking.

Runs the pose estimation and the IdentityTracker of the game (no game states, no stage) on the webcam
or on a recorded video, shows the measurements on screen and writes them to files. It measures:

    identity accuracy   % of evaluated person-frames that have the correct player
    identity switches   per trial and per minute
    success rate        % of trials where both players have the correct ID at the end
    recovery            % of trials that recover, and the time from the end of the event
                        (crossing, occlusion, re-entry) until both players are correct again
    speed               FPS, time per part (pose, tracking, faces) in ms, and latency
    conditions          mean brightness of the image (lighting) and player height in pixels (distance)

Spatial error (mm) is not measured: the game only gives positions in pixels.

Run (from the project folder):
    python tests/eval_identity.py --condition bright --subjects "A,B" --distance 2.5
    python tests/eval_identity.py --video tests/results/crossing.mp4 --condition dim
    python tests/eval_identity.py --record tests/results/crossing.mp4    (also save the camera images)

Keys while it runs:
    c / o / r   start a trial: crossing / occlusion / leave and re-enter
    e or SPACE  end the trial (press it when both players stand apart again, in their final places)
    s           the players swapped sides (outside a trial: now; in a trial: at the end of the trial).
                A crossing trial already expects a swap: press s only when they did NOT swap.
    k           calibrate: the IDs shown now are correct (when the start was wrong)
    x           throw the current trial away
    p           pause (video)
    q / ESC     stop and write the results

How the ground truth works (no labelling afterwards needed):
    A frame is "clean" when exactly two people are found and their boxes do not overlap. In a clean
    frame we know who is who by their side: the person on the left is the person who was on the left
    before, unless the players swapped sides (a crossing trial, or the s key). Only clean frames are
    evaluated. The first clean frame where both players have an ID sets the ground truth (check on
    screen that it is right, else press k).
    In a trial, the frames that are not clean are the "event" (players overlap, someone is hidden or
    out of the image). The frames before the event use the old sides, the frames after it the new
    sides. Recovery time = from the first clean frame after the event until both IDs are correct.

Output: tests/results/<date>_<condition>/  frames.csv (every frame), trials.csv (every trial),
summary.json, and one line per run in tests/results/summary.csv (to compare lighting, distance, ...).
"""
import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "tests" / "results"
TRIAL_KINDS = {"c": "crossing", "o": "occlusion", "r": "re-entry"}
EXPECTS_SWAP = {"crossing": True, "occlusion": False, "re-entry": False}
WARMUP_FRAMES = 10   # the first frames are slow (the model starts up): not in the timing results


# ------------------------------------------------------------------ the measurements (no camera needed)
@dataclass
class Trial:
    kind: str
    start: float
    swap: bool                      # do the players end on the other side?
    end: float = None
    frames: list = field(default_factory=list, repr=False)   # (time, clean, ids) of every frame in the trial
    event_start: float = None       # first frame that is not clean
    event_end: float = None         # first clean frame after the last frame that is not clean
    evaluated: int = 0              # person-frames
    correct: int = 0
    switches: int = 0
    recovered: bool = False
    recovery_time: float = None     # seconds from event_end until both IDs are correct
    success: bool = None            # both IDs correct at the end of the trial (None: trial not valid)
    note: str = ""


class IdentityEvaluator:
    """Counts correct IDs, switches and recoveries. Feed it one frame at a time with add().

    ids: (ID of the left person, ID of the right person), 0 = this person has no player.
    clean: exactly two people, apart from each other (only then the sides tell who is who).
    """

    def __init__(self):
        self.truth = None           # (correct ID of the left person, correct ID of the right person)
        self.trial = None           # the trial that is running
        self.trials = []
        self.evaluated = 0          # person-frames
        self.correct = 0
        self.switches = 0
        self.last_ok = None         # was the last evaluated frame correct?
        self.first_time = None
        self.last_time = None

    def add(self, t, clean, ids):
        if self.first_time is None:
            self.first_time = t
        self.last_time = t
        if self.trial is not None:
            self.trial.frames.append((t, clean, ids))   # scored at the end of the trial
            return
        if not clean:
            return
        if self.truth is None:
            if 0 in ids or ids[0] == ids[1]:
                return
            self.truth = ids                           # the start: the IDs of the registration are correct
        self._score(ids, self.truth)

    def frame_is_correct(self, ids):
        """For the screen: True / False, or None when it is not known yet."""
        if self.truth is None or self.trial is not None or ids is None:
            return None
        return ids == self.truth

    def calibrate(self, ids):
        """The IDs shown now are correct (key k)."""
        if ids is not None and 0 not in ids and ids[0] != ids[1]:
            self.truth = ids
            self.last_ok = True
            return True
        return False

    def swap_sides(self):
        """The players swapped sides (key s)."""
        if self.trial is not None:
            self.trial.swap = not self.trial.swap
        elif self.truth is not None:
            self.truth = self.truth[::-1]

    def start_trial(self, kind, t):
        if self.trial is not None:
            self.end_trial(t)
        self.trial = Trial(kind, t, EXPECTS_SWAP[kind])

    def discard_trial(self):
        self.trial = None

    def end_trial(self, t):
        trial, self.trial = self.trial, None
        if trial is None:
            return None
        trial.end = t
        frames = trial.frames
        not_clean = [i for i, (_, clean, _) in enumerate(frames) if not clean]
        if self.truth is None:
            trial.note = "no ground truth yet (calibrate first)"
        elif not not_clean:
            trial.note = "no event seen (the players never overlapped or disappeared)"
            for frame_time, _, ids in frames:
                self._score(ids, self.truth, trial)
        else:
            first, last = not_clean[0], not_clean[-1]
            old = self.truth
            new = old[::-1] if trial.swap else old
            trial.event_start = frames[first][0]
            if last + 1 < len(frames):
                trial.event_end = frames[last + 1][0]
            for i, (frame_time, clean, ids) in enumerate(frames):
                if first <= i <= last:
                    continue                           # during the event nobody knows who is who
                ok = self._score(ids, old if i < first else new, trial)
                if i > last and ok and not trial.recovered:
                    trial.recovered = True
                    trial.recovery_time = frame_time - trial.event_end
            if trial.event_end is None:
                trial.note = "ended during the event (press e when both players stand apart)"
            else:
                trial.success = self.last_ok is True
            self.truth = new
        self.trials.append(trial)
        return trial

    def _score(self, ids, truth, trial=None):
        """Count one evaluated frame. Returns True when both IDs are correct."""
        good = sum(a == b for a, b in zip(ids, truth))
        ok = good == 2
        self.evaluated += 2
        self.correct += good
        if trial is not None:
            trial.evaluated += 2
            trial.correct += good
        if self.last_ok is True and not ok:            # it was correct and now it is not: a switch
            self.switches += 1
            if trial is not None:
                trial.switches += 1
        self.last_ok = ok
        return ok

    @property
    def accuracy(self):
        return 100.0 * self.correct / self.evaluated if self.evaluated else None

    def summary(self):
        minutes = ((self.last_time or 0) - (self.first_time or 0)) / 60
        result = {
            "duration_s": round(minutes * 60, 1),
            "evaluated_person_frames": self.evaluated,
            "identity_accuracy_pct": _round(self.accuracy),
            "identity_switches": self.switches,
            "switches_per_minute": _round(self.switches / minutes) if minutes > 0 else None,
            "trials": {},
        }
        for kind in EXPECTS_SWAP:
            trials = [t for t in self.trials if t.kind == kind]
            valid = [t for t in trials if t.success is not None]
            times = [t.recovery_time for t in valid if t.recovery_time is not None]
            result["trials"][kind] = {
                "trials": len(trials),
                "valid_trials": len(valid),
                "success_rate_pct": _round(100 * sum(t.success for t in valid) / len(valid)) if valid else None,
                "recovered_pct": _round(100 * sum(t.recovered for t in valid) / len(valid)) if valid else None,
                "switches": sum(t.switches for t in valid),
                "switches_per_trial": _round(sum(t.switches for t in valid) / len(valid)) if valid else None,
                "recovery_time_mean_s": _round(statistics.mean(times), 3) if times else None,
                "recovery_time_max_s": _round(max(times), 3) if times else None,
                "event_time_mean_s": _round(statistics.mean(t.event_end - t.event_start for t in valid), 2)
                if valid else None,
            }
        return result


def _round(value, digits=1):
    return None if value is None else round(float(value), digits)


# ------------------------------------------------------------------ from the tracker to (clean, ids)
def left_right_ids(people, players, max_overlap=0.25):
    """(clean, ids): ids = (player ID of the left person, of the right person), 0 = no player.

    clean is False (and ids None) when there are not exactly two people, or when their boxes overlap
    more than `max_overlap` of the narrower box (then the side does not say who is who)."""
    if len(people) != 2:
        return False, None
    left, right = sorted(people, key=lambda p: p.box[0] + p.box[2] / 2)
    overlap = left.box[0] + left.box[2] - right.box[0]
    if overlap > max_overlap * min(left.box[2], right.box[2]):
        return False, None
    by_box = {tuple(entry["box"]): player_id for player_id, entry in players.items()
              if entry["visible"] and entry["box"] is not None}
    return True, tuple(by_box.get(tuple(p.box), 0) for p in (left, right))


def brightness(image):
    """Mean brightness 0..255 (the V of HSV): one number for the lighting of the room."""
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2].mean())


# ------------------------------------------------------------------ on screen
GREEN, RED, WHITE, YELLOW = (80, 220, 80), (60, 60, 230), (255, 255, 255), (0, 215, 255)
GREY_TEXT = (200, 200, 200)


def draw_panel(image, lines, origin=(10, 10)):
    """Lines of (text, colour) on a dark box."""
    font, scale, thickness, step = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 22
    width = max(cv2.getTextSize(text, font, scale, thickness)[0][0] for text, _ in lines) + 20
    x, y = origin
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + step * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    for i, (text, color) in enumerate(lines):
        cv2.putText(image, text, (x + 10, y + 22 + i * step), font, scale, color, thickness, cv2.LINE_AA)


def draw_truth(image, people, ok):
    """OK / WRONG above both people in a clean frame."""
    if ok is None or len(people) != 2:
        return
    for person in people:
        x, y, w, _ = person.box
        cv2.putText(image, "OK" if ok else "WRONG", (x + w - 90, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN if ok else RED, 2, cv2.LINE_AA)


# ------------------------------------------------------------------ the run
def ms_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(statistics.mean(values), 2), "median": round(statistics.median(values), 2),
            "p95": round(float(np.percentile(values, 95)), 2), "max": round(max(values), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="a recorded video instead of the webcam")
    parser.add_argument("--condition", default="normal", help="name of the test condition, e.g. bright, dim")
    parser.add_argument("--subjects", default="", help="who played, e.g. \"A,B\"")
    parser.add_argument("--distance", type=float, help="distance players-camera in metres (measured)")
    parser.add_argument("--no-faces", action="store_true", help="do not run face tracking (only pose + identity)")
    parser.add_argument("--record", help="save the camera images to this video (to replay with --video)")
    parser.add_argument("--save-video", action="store_true", help="save the screen with the measurements")
    parser.add_argument("--overlap", type=float, default=0.25, help="boxes overlapping more than this: not clean")
    args = parser.parse_args()

    from camera import Camera, FPSCounter
    from identity_tracking.identity import IdentityTracker, draw_labels, people_from_pose
    from pose_tracking.pose_processing import PoseEstimator
    if not args.no_faces:
        from face_tracking.face_tracking import get_face_properties, smooth_face_properties
        from face_tracking.game_effects import faces_for_players

    camera = Camera(args.video if args.video else args.camera)
    video_fps = camera.cap.get(cv2.CAP_PROP_FPS) or 25.0
    pose_estimator = PoseEstimator(CONFIDENCE_THRESHOLD=0.1, KEYPOINTS_SMOOTHING=0.7, MOTIONS_SMOOTHING=0.5)
    tracker = IdentityTracker(max_players=2)
    evaluator = IdentityEvaluator()
    fps = FPSCounter()

    out_dir = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_file = open(out_dir / "frames.csv", "w", newline="")
    frames_csv = csv.writer(frames_file)
    frames_csv.writerow(["frame", "time_s", "people", "clean", "left_id", "right_id", "correct", "in_trial",
                         "pose_ms", "tracking_ms", "faces_ms", "processing_ms", "latency_ms", "fps",
                         "brightness", "player_height_px"])
    recorder = screen_writer = None

    window = "Identity evaluation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    message, message_until = "", 0.0
    timings = {"pose_ms": [], "tracking_ms": [], "faces_ms": [], "processing_ms": [], "latency_ms": [], "fps": []}
    light, heights = [], []
    paused, image, t = False, None, 0.0
    run_start = time.perf_counter()

    while True:
        if not paused:
            frame = camera.read()
            if frame is None:
                break
            # video: the time in the video (so recovery times do not depend on the speed of this computer)
            t = frame.index / video_fps if args.video else frame.time - run_start

            start = time.perf_counter()
            keypoints, _ = pose_estimator.process(frame.image)
            after_pose = time.perf_counter()
            people = people_from_pose(keypoints, min_confidence=0.1)
            players = tracker.update(frame.image, people)
            after_tracking = time.perf_counter()
            faces_ms = None
            if not args.no_faces:
                faces = smooth_face_properties(get_face_properties(frame.image))
                faces_for_players(players, faces)
                faces_ms = (time.perf_counter() - after_tracking) * 1000
            done = time.perf_counter()
            fps.update()

            pose_ms = (after_pose - start) * 1000
            tracking_ms = (after_tracking - after_pose) * 1000
            processing_ms = (done - start) * 1000
            latency_ms = (done - frame.time) * 1000 if not args.video else None   # camera image -> result
            clean, ids = left_right_ids(people, players, args.overlap)
            evaluator.add(t, clean, ids)
            ok = evaluator.frame_is_correct(ids) if clean else None
            light.append(brightness(frame.image))
            height = statistics.mean(p.box[3] for p in people) if people else None
            if height:
                heights.append(height)
            if frame.index >= WARMUP_FRAMES:
                for name, value in (("pose_ms", pose_ms), ("tracking_ms", tracking_ms), ("faces_ms", faces_ms),
                                    ("processing_ms", processing_ms), ("latency_ms", latency_ms), ("fps", fps.fps)):
                    timings[name].append(value)
            frames_csv.writerow([frame.index, round(t, 3), len(people), int(clean),
                                 ids[0] if ids else "", ids[1] if ids else "", "" if ok is None else int(ok),
                                 int(evaluator.trial is not None), round(pose_ms, 2), round(tracking_ms, 3),
                                 "" if faces_ms is None else round(faces_ms, 2), round(processing_ms, 2),
                                 "" if latency_ms is None else round(latency_ms, 1), round(fps.fps, 1),
                                 round(light[-1], 1), "" if height is None else round(height)])

            if args.record:
                if recorder is None:
                    h, w = frame.image.shape[:2]
                    recorder = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                recorder.write(cv2.flip(frame.image, 1))   # un-mirror: the Camera mirrors it again on replay

            image = frame.image.copy()
            draw_labels(image, players)
            draw_truth(image, people, ok)
            trial = evaluator.trial
            truth = evaluator.truth
            lines = [
                (f"FPS {fps.fps:5.1f}   processing {processing_ms:5.1f} ms   "
                 + (f"latency {latency_ms:5.1f} ms" if latency_ms is not None else "(video: no latency)"), WHITE),
                (f"pose {pose_ms:5.1f} ms   tracking {tracking_ms:5.2f} ms   "
                 + (f"faces {faces_ms:5.1f} ms" if faces_ms is not None else "faces off"), WHITE),
                (f"brightness {light[-1]:5.1f}   people {len(people)}   "
                 + ("clean" if clean else "not clean (overlap / missing)"), WHITE),
                (f"accuracy {evaluator.accuracy or 0:5.1f} %   switches {evaluator.switches}   "
                 f"trials {len(evaluator.trials)}", WHITE),
                ("ground truth: " + (f"left = Player {truth[0]}, right = Player {truth[1]}" if truth
                                     else "not set (needs 2 players apart)"), YELLOW if truth is None else WHITE),
                ((f"TRIAL {trial.kind}  {t - trial.start:4.1f} s  expects swap: {'yes' if trial.swap else 'no'}"
                  "   (e = end, x = throw away, s = toggle swap)") if trial else
                 "no trial   c = crossing   o = occlusion   r = re-entry   k = calibrate   q = stop",
                 YELLOW if trial else GREY_TEXT),
            ]
            if message and time.perf_counter() < message_until:
                lines.append((message, YELLOW))
            draw_panel(image, lines)
            if args.save_video:
                if screen_writer is None:
                    h, w = image.shape[:2]
                    screen_writer = cv2.VideoWriter(str(out_dir / "screen.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                    25, (w, h))
                screen_writer.write(image)

        cv2.imshow(window, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            paused = not paused
        elif chr(key) in TRIAL_KINDS:
            evaluator.start_trial(TRIAL_KINDS[chr(key)], t)
            message = f"started trial: {TRIAL_KINDS[chr(key)]}"
        elif key in (ord("e"), ord(" ")) and evaluator.trial is not None:
            done_trial = evaluator.end_trial(t)
            if done_trial.success is None:
                message = f"trial not valid: {done_trial.note}"
            else:
                recovery = f"{done_trial.recovery_time:.2f} s" if done_trial.recovered else "not recovered"
                message = (f"{done_trial.kind}: {'SUCCESS' if done_trial.success else 'FAILED'}, "
                           f"{done_trial.switches} switches, recovery {recovery}")
            print(message)
        elif key == ord("s"):
            evaluator.swap_sides()
            message = "sides swapped"
        elif key == ord("k"):
            message = "calibrated" if evaluator.calibrate(ids) else "calibrate needs 2 players apart, both with an ID"
        elif key == ord("x"):
            evaluator.discard_trial()
            message = "trial thrown away"
        else:
            key = None
        if key is not None:
            message_until = time.perf_counter() + 3.0

    if evaluator.trial is not None:
        evaluator.end_trial(t)
    camera.release()
    for writer in (recorder, screen_writer):
        if writer is not None:
            writer.release()
    cv2.destroyAllWindows()
    frames_file.close()
    write_results(out_dir, args, evaluator, timings, light, heights)


def write_results(out_dir, args, evaluator, timings, light, heights):
    with open(out_dir / "trials.csv", "w", newline="") as file:
        columns = [name for name in Trial.__dataclass_fields__ if name != "frames"]
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        for trial in evaluator.trials:
            row = asdict(trial)
            row.pop("frames")
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})

    summary = {
        "condition": args.condition,
        "subjects": args.subjects,
        "distance_m": args.distance,
        "source": args.video or f"camera {args.camera}",
        "faces": not args.no_faces,
        "brightness_mean": _round(statistics.mean(light)) if light else None,
        "player_height_px_mean": _round(statistics.mean(heights)) if heights else None,
        **evaluator.summary(),
        "timing_ms": {name: ms_stats(values) for name, values in timings.items() if name != "fps"},
        "fps_mean": _round(statistics.mean(timings["fps"])) if timings["fps"] else None,
        "spatial_error_mm": "not measured (positions are only in pixels)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # one line per run: compare conditions (lighting, distance, subjects) in one table
    row = {
        "folder": out_dir.name, "condition": args.condition, "subjects": args.subjects,
        "distance_m": args.distance, "brightness": summary["brightness_mean"],
        "player_height_px": summary["player_height_px_mean"], "duration_s": summary["duration_s"],
        "accuracy_pct": summary["identity_accuracy_pct"], "switches": summary["identity_switches"],
        "switches_per_min": summary["switches_per_minute"], "fps": summary["fps_mean"],
        "processing_ms": (summary["timing_ms"]["processing_ms"] or {}).get("mean"),
        "tracking_ms": (summary["timing_ms"]["tracking_ms"] or {}).get("mean"),
        "latency_ms": (summary["timing_ms"]["latency_ms"] or {}).get("mean"),
    }
    for kind, stats in summary["trials"].items():
        row[f"{kind}_trials"] = stats["valid_trials"]
        row[f"{kind}_success_pct"] = stats["success_rate_pct"]
        row[f"{kind}_recovery_s"] = stats["recovery_time_mean_s"]
    table = RESULTS_DIR / "summary.csv"
    new_table = not table.exists()
    with open(table, "a", newline="") as file:
        writer = csv.DictWriter(file, list(row))
        if new_table:
            writer.writeheader()
        writer.writerow(row)

    print(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
