"""Task 4: grade how well each player follows the reference choreography."""
from __future__ import annotations

from core.config import Config
from core.types import GameEvent, Grade, Player, PoseObs
from gameplay.choreography import Choreography

# Similarity thresholds for grades. TODO(T4): tune on recorded sessions and justify in the report.
GRADE_THRESHOLDS = ((0.90, Grade.PERFECT), (0.75, Grade.GOOD), (0.55, Grade.OK))


class Scorer:
    """Emits one GRADE event per player per move segment.

    TODO(T4):
      - During a move segment, compare each active player's pose to the reference inside a timing
        tolerance window (choreo.window(t - tol, t + tol)); humans lag the model by 100-300 ms.
      - Keep the best (or average) similarity over the segment; when the segment ends, emit
        GameEvent(EventType.GRADE, t, pid, grade=grade_from_similarity(s)). Emit it exactly once
        (state per pid + segment), so a move can never be scored twice.
      - LOST players during a segment: decide (MISS? not scored?) and document it.
      - Gold moves: add data={"points": bonus}.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def update(self, players: dict[int, Player], song_t: float, choreo: Choreography) -> list[GameEvent]:
        return []

    def reset(self) -> None:
        pass


def pose_similarity(live: PoseObs, ref: PoseObs) -> float:
    """TODO(T4, with features from T2): similarity in [0, 1] between two poses.

    Suggestion: mean cosine similarity of limb direction vectors (pose.features.limb_vectors),
    weighted by keypoint confidence, so height, distance and position don't matter.
    """
    raise NotImplementedError


def grade_from_similarity(similarity: float) -> Grade:
    for threshold, grade in GRADE_THRESHOLDS:
        if similarity >= threshold:
            return grade
    return Grade.MISS
