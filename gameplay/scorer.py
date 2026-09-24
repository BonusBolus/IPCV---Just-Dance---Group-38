"""Task 4: grade how well each player follows the reference choreography.

Per frame, for every visible player:
    live limb directions (8 limbs, isotropic, confidence-weighted)
    vs. reference limb directions in the window [t - tolerance, t + 0.1 s]
      (players react to the model, so they lag it by 100-300 ms)
    limb score = 1 - angle_error / 60 deg (clipped to [0, 1]), weighted mean over limbs
    frame similarity = best match inside the window
Per move segment (pose moves only; HIGH_FIVE/SWAP are scored by interactions.py):
    segment score = 75th percentile of the frame similarities (robust to a few bad frames
    and to the transition into the pose) -> Perfect / Good / OK / Miss
    graded once, `tolerance` after the segment ends, for every player (active or not)
    a player that was not visible during the segment gets a MISS: leaving costs points
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.config import Config
from core.types import EventType, GameEvent, Grade, Player, PoseObs
from gameplay.choreography import Choreography
from pose.features import limb_vectors, limb_vectors_arr

ANGLE_TOLERANCE_DEG = 60.0   # a limb this far off scores 0
# Similarity thresholds for grades (1.0 = identical limb directions).
# 0.80 ~ 12 deg mean limb error, 0.65 ~ 21 deg, 0.45 ~ 33 deg.
GRADE_THRESHOLDS = ((0.80, Grade.PERFECT), (0.65, Grade.GOOD), (0.45, Grade.OK))
MIN_WEIGHT = 1.5      # at least ~2 reliable limbs are needed to judge a pose
MIN_SAMPLES = 3       # frames needed in a segment to grade it


def similarity_to_window(live: PoseObs, ref_vec: np.ndarray, ref_w: np.ndarray, min_conf: float) -> float | None:
    """Best similarity of a live pose to N reference frames (ref_vec (N, L, 2), ref_w (N, L))."""
    vec, w = limb_vectors(live, min_conf)
    weight = ref_w * w[None]                                   # (N, L)
    cos = np.clip(np.einsum("nld,ld->nl", ref_vec, vec), -1.0, 1.0)
    limb_score = np.clip(1.0 - np.degrees(np.arccos(cos)) / ANGLE_TOLERANCE_DEG, 0.0, 1.0)
    total = weight.sum(axis=1)
    valid = total >= MIN_WEIGHT
    if not valid.any():
        return None
    sims = (weight * limb_score).sum(axis=1) / np.maximum(total, 1e-6)
    return float(sims[valid].max())


def pose_similarity(live: PoseObs, ref: PoseObs, min_conf: float = 0.5) -> float | None:
    """Similarity in [0, 1] between two poses; None if too few limbs are visible in both."""
    vec, w = limb_vectors(ref, min_conf)
    return similarity_to_window(live, vec[None], w[None], min_conf)


def grade_from_similarity(similarity: float) -> Grade:
    for threshold, grade in GRADE_THRESHOLDS:
        if similarity >= threshold:
            return grade
    return Grade.MISS


class Scorer:
    def __init__(self, cfg: Config):
        self.gc = cfg.gameplay
        self.min_conf = cfg.pose.min_keypoint_conf
        self.live_similarity: dict[int, float | None] = {}
        self._samples: dict[tuple[int, int], list[float]] = defaultdict(list)
        self._next_to_grade = 0

    def update(self, players: dict[int, Player], song_t: float, choreo: Choreography) -> list[GameEvent]:
        tol = self.gc.timing_tolerance_s
        times, kps, conf = choreo.window(song_t - tol, song_t + 0.1)
        ref = limb_vectors_arr(kps, conf, choreo.aspect, self.min_conf) if len(times) else None
        seg_i, seg = choreo.segment_at(song_t - 0.5 * tol)  # the player's time lags the song

        for pid, p in players.items():
            sim = None
            if p.is_active and p.pose is not None and ref is not None:
                sim = similarity_to_window(p.pose, ref[0], ref[1], self.min_conf)
            self.live_similarity[pid] = sim
            if sim is not None and seg is not None and not seg.move.is_interaction:
                self._samples[(pid, seg_i)].append(sim)

        events = []
        while self._next_to_grade < len(choreo.moves):
            i = self._next_to_grade
            seg = choreo.moves[i]
            if seg.end + tol > song_t:
                break
            self._next_to_grade += 1
            if seg.move.is_interaction:
                continue
            for pid in players:
                events.append(self._grade(pid, i, seg, song_t))
        return events

    def _grade(self, pid: int, i: int, seg, song_t: float) -> GameEvent:
        samples = self._samples.pop((pid, i), [])
        if len(samples) < MIN_SAMPLES:
            grade, score = Grade.MISS, 0.0
        else:
            score = float(np.percentile(samples, self.gc.score_percentile))
            grade = grade_from_similarity(score)
        return GameEvent(EventType.GRADE, song_t, pid=pid, grade=grade, move=seg.move,
                         data={"similarity": score, "segment": i, "duet": seg.duet, "samples": len(samples)})

    def reset(self) -> None:
        self.live_similarity.clear()
        self._samples.clear()
        self._next_to_grade = 0
