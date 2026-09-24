"""Task 3: assign anonymous detections to persistent player identities.

Pipeline per frame:
    1. predict every track's torso centre with a constant-velocity Kalman filter
       (velocity is damped while the track is lost)
    2. describe every detection: torso centre, body size, torso colour histogram
    3. cost matrix track x detection:
           w_pos  * distance(predicted, measured) / gate   (gate widens with Kalman uncertainty)
         + w_app  * Bhattacharyya(histogram_track, histogram_detection)
         + w_size * |log(size_track / size_detection)|
         + w_shape * mean keypoint distance(last pose shifted by the track velocity, detection) / size
       solved with the Hungarian algorithm; pairs with cost > max_cost are rejected
    4. unmatched detection:
         - fewer players than expected -> new player (new players are numbered left to right)
         - otherwise -> re-entry of a LOST player if the appearance matches, else ignored
           (spectators walking by cannot steal an identity)
    5. unmatched track -> LOST: keeps its last pose, pid and score. Nothing is reassigned.
    6. faces -> players: Hungarian on face centre vs. head keypoints

Crossing: while two players pass each other, the torso positions coincide for a few frames.
Velocity prediction carries each track through the crossing, and the pose-shape term (where
are the arms and legs compared to the last frame) plus appearance decide the ambiguous frames.
During partial occlusion the hidden player's appearance model is frozen, so it is not
polluted by the other player's colours.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.config import Config
from core.filters import ConstantVelocityKalman
from core.image_utils import bbox_iou
from core.types import KP, EventType, FaceObs, FrameData, Player, PoseObs, TrackState
from identity.appearance import AppearanceModel, histogram_distance
from pose.features import torso_length

log = logging.getLogger(__name__)

_TORSO = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
_HEAD = [KP["nose"], KP["left_eye"], KP["right_eye"], KP["left_ear"], KP["right_ear"]]
_BIG = 1e6


@dataclass
class _Detection:
    pose: PoseObs
    center: np.ndarray     # normalized
    size: float            # torso length, isotropic units
    hist: np.ndarray | None
    bbox: tuple | None


@dataclass
class _Track:
    kf: ConstantVelocityKalman
    size: float
    hist: np.ndarray | None
    last_pose: PoseObs | None = None
    last_t: float = 0.0
    lost_since: float | None = None


def body_center(pose: PoseObs, min_conf: float = 0.5) -> np.ndarray | None:
    """Torso centre (more stable than the centre of all keypoints: arms swing while dancing)."""
    ok = pose.confidence >= min_conf
    torso = [i for i in _TORSO if ok[i]]
    if len(torso) >= 2:
        return pose.keypoints[torso].mean(axis=0)
    c = pose.center(min_conf)
    return None if c is None else np.asarray(c, np.float32)


class PlayerTracker:
    """Keeps `players` (pid -> Player) up to date every frame. Pids start at 1 and never change
    meaning during a game; `reset()` (on entering the lobby) starts a new game."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tc = cfg.tracking
        self.min_conf = cfg.pose.min_keypoint_conf
        self.max_players = cfg.game.num_players
        self.players: dict[int, Player] = {}
        self.appearance = AppearanceModel()
        self.stats: Counter[str] = Counter()  # matches, new, lost, reentries: for evaluation
        self._tracks: dict[int, _Track] = {}
        self._events: list[tuple[EventType, int]] = []
        self._last_t: float | None = None

    # ------------------------------------------------------------------ public API
    def update(self, frame: FrameData, poses: list[PoseObs], faces: list[FaceObs], t: float) -> dict[int, Player]:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        for tr in self._tracks.values():
            damping = 1.0 if tr.lost_since is None else 0.8 ** (dt * 30)
            tr.kf.predict(dt, damping)

        dets = [d for d in (self._describe(frame, p) for p in poses) if d is not None]
        self._flag_occlusions(dets)
        matches, unmatched = self._match(dets, t)
        matches += self._assign_unmatched(dets, unmatched, {pid for _, pid in matches}, t)

        matched_pids = set()
        for i, pid in matches:
            self._apply(pid, dets[i], t)
            matched_pids.add(pid)
        for pid, player in self.players.items():
            if pid not in matched_pids and player.state is TrackState.ACTIVE:
                player.state = TrackState.LOST
                player.face = None
                self._tracks[pid].lost_since = t
                self._events.append((EventType.PLAYER_LOST, pid))
                self.stats["lost"] += 1
        self._associate_faces(faces)
        return self.players

    def active_players(self) -> list[Player]:
        return [p for p in self.players.values() if p.is_active]

    def pop_events(self) -> list[tuple[EventType, int]]:
        events, self._events = self._events, []
        return events

    def reset(self) -> None:
        self.players.clear()
        self._tracks.clear()
        self._events.clear()
        self._last_t = None

    # ------------------------------------------------------------------ steps
    def _describe(self, frame: FrameData, pose: PoseObs) -> _Detection | None:
        center = body_center(pose, self.min_conf)
        if center is None:
            return None
        size = torso_length(pose, self.min_conf)
        if size is None:
            box = pose.bbox(self.min_conf)
            size = 0.3 * box[3] if box else 0.1
        return _Detection(pose, center, max(size, 1e-3), self.appearance.extract(frame, pose),
                          pose.bbox(self.min_conf))

    def _flag_occlusions(self, dets: list[_Detection]) -> None:
        """Overlapping bodies: don't learn appearance from them (the colours are mixed)."""
        for i, a in enumerate(dets):
            for b in dets[i + 1:]:
                if a.bbox and b.bbox and bbox_iou(a.bbox, b.bbox) > 0.15:
                    a.hist = b.hist = None

    def _cost(self, det: _Detection, tr: _Track, t: float) -> float:
        aspect = det.pose.aspect
        lost_for = 0.0 if tr.lost_since is None else t - tr.lost_since
        if lost_for > self.tc.lost_predict_s:
            pos_cost = 0.5  # position no longer informative: let appearance decide
        else:
            d = (det.center - tr.kf.position) * np.array([aspect, 1.0])
            gate = self.tc.position_gate + 2.0 * tr.kf.position_std()
            pos_cost = float(np.linalg.norm(d)) / gate
        app_cost = 0.3 if det.hist is None or tr.hist is None else histogram_distance(det.hist, tr.hist)
        size_cost = abs(np.log(det.size / tr.size))
        shape_cost = self._shape_cost(det, tr, t) if lost_for <= self.tc.lost_predict_s else 0.5
        return (self.tc.w_position * pos_cost + self.tc.w_appearance * app_cost
                + self.tc.w_size * size_cost + self.tc.w_shape * shape_cost)

    def _shape_cost(self, det: _Detection, tr: _Track, t: float) -> float:
        """Mean distance (in torso lengths) between the detection's keypoints and the track's
        last pose moved along with the track velocity. Distinguishes overlapping bodies by the
        position of their limbs."""
        last = tr.last_pose
        if last is None:
            return 0.5
        ok = (det.pose.confidence >= self.min_conf) & (last.confidence >= self.min_conf)
        if ok.sum() < 4:
            return 0.5
        predicted = last.keypoints[ok] + tr.kf.velocity * max(0.0, t - tr.last_t)
        d = (det.pose.keypoints[ok] - predicted) * np.array([det.pose.aspect, 1.0])
        return float(min(2.0, np.linalg.norm(d, axis=1).mean() / tr.size))

    def _match(self, dets: list[_Detection], t: float) -> tuple[list[tuple[int, int]], list[int]]:
        pids = list(self._tracks)
        if not dets or not pids:
            return [], list(range(len(dets)))
        cost = np.array([[self._cost(d, self._tracks[pid], t) for pid in pids] for d in dets])
        rows, cols = linear_sum_assignment(np.minimum(cost, _BIG))
        matches = [(int(i), pids[j]) for i, j in zip(rows, cols) if cost[i, j] <= self.tc.max_cost]
        matched = {i for i, _ in matches}
        return matches, [i for i in range(len(dets)) if i not in matched]

    def _assign_unmatched(self, dets, unmatched: list[int], taken: set[int], t: float) -> list[tuple[int, int]]:
        out = []
        # biggest detections first (nearest people are the players), then left to right
        order = sorted(unmatched, key=lambda i: -dets[i].size)[: self.max_players]
        for i in sorted(order, key=lambda i: dets[i].center[0]):
            if len(self._tracks) < self.max_players:
                pid = next(p for p in range(1, self.max_players + 1) if p not in self._tracks)
                self._tracks[pid] = _Track(ConstantVelocityKalman(dets[i].center, process_noise=1.0),
                                           dets[i].size, dets[i].hist)
                self._create_player(pid)
                self.stats["new"] += 1
                log.info("New player %d", pid)
                out.append((i, pid))
                taken.add(pid)
                continue
            # re-entry: best-matching LOST track by appearance
            best, best_d = None, self.tc.reentry_max_appearance
            for pid, tr in self._tracks.items():
                if pid in taken or tr.lost_since is None:
                    continue
                d = 0.3 if dets[i].hist is None or tr.hist is None else histogram_distance(dets[i].hist, tr.hist)
                if d < best_d:
                    best, best_d = pid, d
            if best is not None:
                self._tracks[best].kf = ConstantVelocityKalman(dets[i].center, process_noise=1.0)
                self.stats["reentries"] += 1
                out.append((i, best))
                taken.add(best)
        return out

    def _apply(self, pid: int, det: _Detection, t: float) -> None:
        tr = self._tracks[pid]
        tr.kf.update(det.center)
        tr.size = 0.9 * tr.size + 0.1 * det.size
        if det.hist is not None:
            a = self.tc.appearance_alpha if tr.hist is not None else 1.0
            tr.hist = det.hist if tr.hist is None else (1 - a) * tr.hist + a * det.hist
        player = self.players[pid]
        if player.state is TrackState.LOST:
            self._events.append((EventType.PLAYER_RETURNED, pid))
        tr.lost_since = None
        tr.last_pose, tr.last_t = det.pose, t
        player.state = TrackState.ACTIVE
        player.pose = det.pose
        player.last_seen = t
        self.stats["matches"] += 1

    def _create_player(self, pid: int) -> None:
        i = (pid - 1) % len(self.cfg.player_colors)
        name = self.cfg.player_names[pid - 1] if pid <= len(self.cfg.player_names) else f"Player {pid}"
        self.players[pid] = Player(pid=pid, name=name, color=self.cfg.player_colors[i])

    def _associate_faces(self, faces: list[FaceObs]) -> None:
        active = [p for p in self.active_players() if p.pose is not None]
        for p in active:
            p.face = None
        if not faces or not active:
            return
        cost = np.full((len(active), len(faces)), _BIG)
        for i, p in enumerate(active):
            head = _head_point(p.pose, self.min_conf)
            if head is None:
                continue
            for j, f in enumerate(faces):
                d = (np.asarray(f.center) - head) * np.array([p.pose.aspect, 1.0])
                scale = max(f.bbox[2] * p.pose.aspect, f.bbox[3], 1e-3)
                cost[i, j] = np.linalg.norm(d) / scale
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < 1.0:  # face centre within one face size of the head keypoints
                active[i].face = faces[j]


def _head_point(pose: PoseObs, min_conf: float) -> np.ndarray | None:
    ok = pose.confidence >= min_conf * 0.6
    head = [i for i in _HEAD if ok[i]]
    return pose.keypoints[head].mean(axis=0) if head else None

