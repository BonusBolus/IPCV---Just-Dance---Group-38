"""Task 4: player-player interaction.

Three interactions, all tied to the choreography:
  DUET      a pose segment marked duet: if *both* players get GOOD or better, both get a bonus
  HIGH_FIVE during a HIGH_FIVE segment, a wrist of one player within high_five_distance_m of
            a wrist of the other, measured in metres at the players' depth (Task 3's spatial
            output), and at similar depths (|dZ| < 0.8 m, so a hand in front of the other
            person's face does not count)
  SWAP      during a SWAP segment the players' left-right order (lateral X in metres) flips
            and stays flipped for 0.3 s. This needs correct identities through the crossing,
            so it also demonstrates Task 3.
Each interaction rewards each player at most once per segment. Both players always get the
same result, so simultaneous actions never conflict.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from core.config import Config
from core.types import KP, EventType, GameEvent, Grade, MoveType, Player
from gameplay.choreography import Choreography
from identity.spatial import SpatialEstimator

WRISTS = (KP["left_wrist"], KP["right_wrist"])


class InteractionDetector:
    def __init__(self, cfg: Config, spatial: SpatialEstimator):
        self.gc = cfg.gameplay
        self.min_conf = cfg.pose.min_keypoint_conf
        self.spatial = spatial
        self._done: set[int] = set()                     # segment indices already rewarded
        self._swap_start_order: dict[int, tuple[int, ...]] = {}
        self._swap_flipped_since: dict[int, float] = {}

    def update(self, players: dict[int, Player], song_t: float, choreo: Choreography,
               grade_events: list[GameEvent] = ()) -> list[GameEvent]:
        events = self._duets(grade_events, song_t)
        seg_i, seg = choreo.segment_at(song_t)
        active = [p for p in players.values() if p.is_active and p.pose is not None]
        if seg is None or seg_i in self._done or len(active) < 2:
            return events
        if seg.move is MoveType.HIGH_FIVE:
            events += self._high_five(seg_i, active, song_t)
        elif seg.move is MoveType.SWAP:
            events += self._swap(seg_i, active, song_t)
        return events

    def reset(self) -> None:
        self._done.clear()
        self._swap_start_order.clear()
        self._swap_flipped_since.clear()

    # ------------------------------------------------------------------ interactions
    def _duets(self, grade_events, song_t: float) -> list[GameEvent]:
        by_segment: dict[int, list[GameEvent]] = defaultdict(list)
        for ev in grade_events:
            if ev.type is EventType.GRADE and ev.data.get("duet"):
                by_segment[ev.data["segment"]].append(ev)
        out = []
        for grades in by_segment.values():
            if len(grades) >= 2 and all(g.grade in (Grade.PERFECT, Grade.GOOD) for g in grades):
                out += [GameEvent(EventType.INTERACTION, song_t, pid=g.pid,
                                  data={"points": self.gc.duet_bonus, "label": "DUET BONUS!"}) for g in grades]
        return out

    def _high_five(self, seg_i: int, active: list[Player], song_t: float) -> list[GameEvent]:
        for a, b in combinations(active, 2):
            pa, pb = a.position_m, b.position_m
            if pa is None or pb is None or abs(pa[1] - pb[1]) > 0.8:
                continue
            for wa in WRISTS:
                for wb in WRISTS:
                    if a.pose.confidence[wa] < self.min_conf or b.pose.confidence[wb] < self.min_conf:
                        continue
                    ma = self.spatial.point_m(a, a.pose.keypoints[wa])
                    mb = self.spatial.point_m(b, b.pose.keypoints[wb])
                    if ma is None or mb is None:
                        continue
                    if np.linalg.norm(ma[:2] - mb[:2]) < self.gc.high_five_distance_m:
                        self._done.add(seg_i)
                        mid = tuple(float(v) for v in (a.pose.keypoints[wa] + b.pose.keypoints[wb]) / 2)
                        return [GameEvent(EventType.INTERACTION, song_t, pid=p.pid, move=MoveType.HIGH_FIVE,
                                          data={"points": self.gc.high_five_bonus, "label": "HIGH FIVE!", "pos": mid})
                                for p in (a, b)]
        return []

    def _swap(self, seg_i: int, active: list[Player], song_t: float) -> list[GameEvent]:
        xs = {}
        for p in active:  # lateral position in metres, or the image position as a fallback
            if p.position_m is not None:
                xs[p.pid] = p.position_m[0]
            elif (c := p.pose.center()) is not None:
                xs[p.pid] = c[0]
        if len(xs) < 2:
            return []
        order = tuple(sorted(xs, key=xs.get))
        start = self._swap_start_order.setdefault(seg_i, order)
        if order == start or set(order) != set(start):
            self._swap_flipped_since.pop(seg_i, None)
            return []
        since = self._swap_flipped_since.setdefault(seg_i, song_t)
        if song_t - since < 0.3:
            return []
        self._done.add(seg_i)
        return [GameEvent(EventType.INTERACTION, song_t, pid=pid, move=MoveType.SWAP,
                          data={"points": self.gc.swap_bonus, "label": "SWAP!"}) for pid in order]
