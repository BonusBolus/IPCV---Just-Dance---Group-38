"""Task 3: assign anonymous detections to persistent player identities."""
from __future__ import annotations

import numpy as np

from core.config import Config
from core.types import KP, FaceObs, Player, PoseObs, TrackState


class PlayerTracker:
    """Keeps `players` (pid -> Player) up to date every frame.

    Contract: after `update`, every Player that was matched this frame is ACTIVE and holds this
    frame's pose/face. Unmatched players are LOST and keep their last observation.
    Pids start at 1 and never change meaning during a game.

    TODO(T3): replace the placeholder logic in `update` with real tracking, e.g.
      - predict each track (Kalman / constant velocity), then match with the Hungarian algorithm
        (scipy.optimize.linear_sum_assignment) on a cost of position distance + appearance distance
      - LOST tracks: keep predicting for a while; on re-entry match against stored appearance
      - associate faces to bodies (face centre vs. nose/ear keypoints)
      - evaluate: identity switches during scripted crossings (MockPoseSource(scenario="cross")
        and recorded videos), recovery time after occlusion, time per update
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_players = cfg.game.num_players
        self.players: dict[int, Player] = {}

    def update(self, poses: list[PoseObs], faces: list[FaceObs], t: float) -> dict[int, Player]:
        # ---- PLACEHOLDER (replace in Task 3) -------------------------------------------------
        # Naive: sort detections left -> right and call them P1, P2. Identities swap as soon as
        # players cross, which is exactly the failure case Task 3 has to solve.
        ordered = sorted(poses, key=lambda p: (p.center() or (0.5, 0.5))[0])[: self.max_players]
        seen = set()
        for i, pose in enumerate(ordered):
            player = self._get_or_create(i + 1)
            player.pose = pose
            player.state = TrackState.ACTIVE
            player.last_seen = t
            seen.add(player.pid)
        for player in self.players.values():
            if player.pid not in seen:
                player.state = TrackState.LOST
        self._associate_faces_placeholder(faces)
        return self.players

    def active_players(self) -> list[Player]:
        return [p for p in self.players.values() if p.is_active]

    def reset(self) -> None:
        self.players.clear()

    # ------------------------------------------------------------------ helpers
    def _get_or_create(self, pid: int) -> Player:
        if pid not in self.players:
            i = (pid - 1) % len(self.cfg.player_colors)
            name = self.cfg.player_names[pid - 1] if pid <= len(self.cfg.player_names) else f"Player {pid}"
            self.players[pid] = Player(pid=pid, name=name, color=self.cfg.player_colors[i])
        return self.players[pid]

    def _associate_faces_placeholder(self, faces: list[FaceObs]) -> None:
        """Greedy: each active player gets the face nearest to their nose. TODO(T3): do this properly."""
        free = list(faces)
        for player in self.active_players():
            player.face = None
            if not free or player.pose is None or player.pose.confidence[KP["nose"]] < 0.3:
                continue
            nose = player.pose.keypoints[KP["nose"]]
            dists = [np.hypot(*(np.asarray(f.center) - nose)) for f in free]
            player.face = free.pop(int(np.argmin(dists)))
