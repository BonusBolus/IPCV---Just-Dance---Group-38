"""Task 4: recognise discrete moves/gestures (the ">= 2 distinguishable actions" requirement)."""
from __future__ import annotations

from core.config import Config
from core.types import GameEvent, MoveType, Player, PoseObs


class Cooldown:
    """Prevents an action from firing repeatedly. Keys are anything hashable, e.g. (pid, move)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict = {}

    def ready(self, key, t: float) -> bool:
        return t - self._last.get(key, float("-inf")) >= self.seconds

    def trigger(self, key, t: float) -> None:
        self._last[key] = t

    def try_trigger(self, key, t: float) -> bool:
        """Trigger and return True if ready, else return False."""
        if self.ready(key, t):
            self.trigger(key, t)
            return True
        return False

    def reset(self) -> None:
        self._last.clear()


class MoveDetector:
    """TODO(T4): classify each player's pose into a MoveType and emit MOVE_DETECTED events.

    - Rule-based on joint angles/positions is fine and explainable (e.g. ARMS_UP = both wrists
      above the nose; T_POSE = both arms horizontal; CLAP = wrists close together with high
      approach velocity; SQUAT = hips drop relative to standing height).
    - Confirmation: require the move for N consecutive frames before firing.
    - Cooldown: use `Cooldown` so a held pose fires once, not every frame.
    - Used for gold moves during the song, and could be used for gestures in the menus
      (e.g. ARMS_UP to start instead of pressing SPACE).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cooldown = Cooldown(seconds=1.0)

    def update(self, players: dict[int, Player], song_t: float) -> list[GameEvent]:
        return []

    def classify(self, pose: PoseObs) -> MoveType:
        raise NotImplementedError

    def reset(self) -> None:
        self.cooldown.reset()
