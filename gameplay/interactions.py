"""Task 4: player-player interaction (required: "meaningful player-player or player-object interaction")."""
from __future__ import annotations

from core.config import Config
from core.types import GameEvent, Player
from gameplay.choreography import Choreography


class InteractionDetector:
    """TODO(T4): detect interactions between players and emit INTERACTION events.

    Ideas:
      - duet moves (MoveSegment.duet): both players must hit the move together for a bonus
      - high five: wrists of the two players close together, in metres (Player.position_m, Task 3)
      - "swap places" move: players cross (this also demonstrates Task 3's identity tracking)
    Resolve simultaneous/conflicting actions deterministically (e.g. by pid order).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def update(self, players: dict[int, Player], song_t: float, choreo: Choreography) -> list[GameEvent]:
        return []

    def reset(self) -> None:
        pass
