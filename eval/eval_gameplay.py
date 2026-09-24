"""Task 4 evaluation: move classifier accuracy, scoring validity, false triggers.

    python -m eval.eval_gameplay

Experiments (synthetic, from the same 2D body model as the reference dancer):
  1. confusion matrix: every move pose with random joint-angle perturbations (+-15 deg),
     keypoint noise, body size, position and camera aspect ratio
  2. scoring validity: grades of a "good" dancer (follows the model with 250 ms lag + noise),
     a "sloppy" dancer (+-35 deg errors), a "random" dancer and a dancer standing still
  3. false triggers: recognised moves per minute during free-style dancing (confirmation +
     cooldown are meant to keep this low)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from core.config import Config
from core.mock_source import body_pose, dance_pose, to_image
from core.types import Grade, MoveType, Player, PoseObs
from gameplay.choreography import MOVE_POSES, Choreography
from gameplay.move_detector import MoveDetector, classify_pose
from gameplay.scorer import Scorer
from pose.features import LIMBS

RNG = np.random.default_rng(0)
CLASSES = [m for m in MOVE_POSES if not m.is_interaction]


def perturbed(move: MoveType, jitter_deg: float) -> np.ndarray:
    params = dict(MOVE_POSES[move])
    for side in ("left", "right"):
        u, f = params[side]
        params[side] = (u + RNG.uniform(-jitter_deg, jitter_deg), f + RNG.uniform(-jitter_deg, jitter_deg))
    if "squat" in params:
        params["squat"] = float(np.clip(params["squat"] + RNG.uniform(-0.2, 0.1), 0, 1))
    return body_pose(**params)


def to_obs(body: np.ndarray, t: float = 0.0, noise: float = 0.003) -> PoseObs:
    aspect = RNG.choice([4 / 3, 16 / 9])
    scale = RNG.uniform(0.15, 0.25)
    kp = to_image(body, (RNG.uniform(0.3, 0.7), 0.6), scale, aspect) + RNG.normal(0, noise, (17, 2))
    return PoseObs(kp.astype(np.float32), np.full(17, 0.9, np.float32), t, aspect=float(aspect))


def confusion(n: int = 200) -> None:
    names = [m.name for m in CLASSES]
    mat = np.zeros((len(CLASSES), len(CLASSES)), int)
    for i, move in enumerate(CLASSES):
        for _ in range(n):
            pred = classify_pose(to_obs(perturbed(move, 15)))
            mat[i, CLASSES.index(pred) if pred in CLASSES else 0] += 1
    print("1. move classifier confusion (rows = true, columns = predicted)")
    print(" " * 12 + "".join(f"{n[:9]:>10}" for n in names))
    for i, row in enumerate(mat):
        print(f"{names[i][:11]:<12}" + "".join(f"{v:>10}" for v in row))
    print(f"   accuracy {np.trace(mat) / mat.sum():.1%}\n")


def scoring() -> None:
    cfg = Config()
    choreo = Choreography.synthetic(64.0)
    dancers = {
        "good (250 ms lag)": lambda t: _follow(choreo, t, lag=0.25, err_deg=8),
        "sloppy (+-35 deg)": lambda t: _follow(choreo, t, lag=0.25, err_deg=35),
        "late (700 ms lag)": lambda t: _follow(choreo, t, lag=0.7, err_deg=8),
        "random": lambda t: to_obs(perturbed(CLASSES[int(t / 1.3) % len(CLASSES)], 40), t),
        "standing still": lambda t: to_obs(perturbed(MoveType.NONE, 5), t),
    }
    print("2. scoring validity: grade distribution over one song")
    for name, dancer in dancers.items():
        scorer = Scorer(cfg)
        player = Player(1, "P1", (0, 0, 0))
        grades = Counter()
        for i in range(int(choreo.duration * 30)):
            t = i / 30
            player.pose = dancer(t)
            for ev in scorer.update({1: player}, t, choreo):
                grades[ev.grade] += 1
        total = sum(grades.values())
        print(f"   {name:<20}" + "  ".join(f"{g.name} {grades[g] / total:4.0%}" for g in Grade))
    print()


def _follow(choreo: Choreography, t: float, lag: float, err_deg: float) -> PoseObs:
    """The reference pose `lag` seconds late with limb-angle errors. The errors are *systematic*:
    fixed per bar (a sloppy dancer holds a wrong pose), plus a little per-frame noise."""
    ref = choreo.reference_at(max(0.0, t - lag))
    kp = ref.keypoints.copy()
    bar_rng = np.random.default_rng(int(max(0.0, t - lag) / 2.0) + int(err_deg) * 1000)
    for a, b, _ in LIMBS:  # rotate each limb (and the limb attached to its end) around its start
        th = np.radians(bar_rng.uniform(-err_deg, err_deg) + RNG.normal(0, 2))
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        chain = [b] + [bb for aa, bb, _ in LIMBS if aa == b]
        kp[chain] = kp[a] + (kp[chain] - kp[a]) @ rot.T
    return PoseObs(kp.astype(np.float32), np.full(17, 0.9, np.float32), t, aspect=choreo.aspect)


def false_triggers(seconds: float = 120.0) -> None:
    cfg = Config()
    det = MoveDetector(cfg)
    player = Player(1, "P1", (0, 0, 0))
    counts = Counter()
    for i in range(int(seconds * 30)):
        t = i / 30
        player.pose = to_obs(dance_pose(t), t)
        for ev in det.update({1: player}, t, t):
            counts[ev.move.name] += 1
    per_min = sum(counts.values()) / (seconds / 60)
    print(f"3. free-style dancing: {per_min:.1f} recognised moves / min  {dict(counts)}")


def main() -> None:
    confusion()
    scoring()
    false_triggers()


if __name__ == "__main__":
    main()
