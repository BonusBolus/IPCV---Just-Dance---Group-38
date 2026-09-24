"""Unit tests of the technical modules on synthetic data (ground truth known)."""
import numpy as np

from core.config import Config
from core.filters import ConstantVelocityKalman, OneEuroFilter
from core.mock_source import MockPoseSource, body_pose, to_image
from core.types import EventType, FrameData, Grade, MoveType, Player, PoseObs
from gameplay.choreography import MOVE_POSES, Choreography
from gameplay.move_detector import MoveDetector, classify_pose
from gameplay.scorer import Scorer, pose_similarity
from identity.player_tracker import PlayerTracker
from pose.features import joint_angles, normalize_pose
from pose.keypoint_filter import PoseSmoother


def _obs(body, center=(0.5, 0.6), scale=0.2, aspect=16 / 9, t=0.0, conf=0.9):
    kp = to_image(body, center, scale, aspect).astype(np.float32)
    return PoseObs(kp, np.full(17, conf, np.float32), t, aspect=aspect)


def test_one_euro_reduces_jitter_and_follows_motion():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=5.0)
    still = [f(np.array([0.5]) + rng.normal(0, 0.01, 1), i / 30) for i in range(60)]
    assert np.std(still[10:]) < 0.5 * 0.01
    for i in range(60, 90):  # fast move to 0.9: must get there without large lag
        y = f(np.array([0.9]), i / 30)
    assert abs(y[0] - 0.9) < 0.02


def test_kalman_predicts_constant_motion():
    kf = ConstantVelocityKalman((0.0, 0.0))
    for i in range(1, 30):
        kf.predict(1 / 30)
        kf.update((i * 0.01, 0.0))
    p = kf.predict(10 / 30)
    assert abs(p[0] - 0.39) < 0.03


def test_pose_smoother_holds_and_drops_missing_keypoints():
    cfg = Config()
    sm = PoseSmoother(cfg)
    pose = _obs(body_pose())
    sm.update(1, pose, 0.0)
    missing = PoseObs(pose.keypoints.copy(), pose.confidence.copy(), 0.1, aspect=pose.aspect)
    missing.confidence[9] = 0.0
    missing.keypoints[9] = (0.0, 0.0)  # garbage position for an undetected wrist
    out = sm.update(1, missing, 0.1)
    assert np.linalg.norm(out.keypoints[9] - pose.keypoints[9]) < 0.01  # held, no jump
    assert out.confidence[9] < cfg.pose.min_keypoint_conf               # but not trusted
    out = sm.update(1, missing, 0.1 + cfg.pose.hold_s + 0.05)
    assert out.confidence[9] == 0.0


def test_features_are_scale_and_aspect_invariant():
    a = normalize_pose(_obs(body_pose((90, 90), (90, 90)), scale=0.15, aspect=4 / 3))
    b = normalize_pose(_obs(body_pose((90, 90), (90, 90)), scale=0.25, aspect=16 / 9, center=(0.3, 0.5)))
    assert np.allclose(a, b, atol=1e-3)
    angles = joint_angles(_obs(body_pose((90, 90), (90, 90))))
    # T-pose: straight elbow; shoulder angle vs. the (slightly tapered) torso side is ~100 deg
    assert abs(angles["left_elbow"] - 180) < 1 and 95 < angles["left_shoulder"] < 105


def test_move_classifier_on_templates():
    for move, params in MOVE_POSES.items():
        if move.is_interaction:
            continue
        assert classify_pose(_obs(body_pose(**params))) is move, move


def test_move_detector_confirmation_and_cooldown():
    det = MoveDetector(Config())
    p = Player(1, "P1", (0, 0, 0), pose=_obs(body_pose(**MOVE_POSES[MoveType.ARMS_UP])))
    fired = []
    for i in range(90):  # hold ARMS_UP for 3 s
        fired += det.update({1: p}, i / 30, i / 30)
    assert len(fired) == 3  # confirmed after 0.2 s, then once per 1 s cooldown
    assert det.held_for(1, MoveType.ARMS_UP, 89 / 30) > 2.5


def test_similarity_orders_poses():
    t = _obs(body_pose((90, 90), (90, 90)))
    close = _obs(body_pose((80, 95), (100, 85)))
    far = _obs(body_pose((170, 170), (20, 20)))
    assert pose_similarity(t, t) > 0.99
    assert pose_similarity(close, t) > 0.8 > pose_similarity(far, t)


def test_scorer_grades_follower_high_and_absentee_miss():
    choreo = Choreography.synthetic(20.0)
    scorer = Scorer(Config())
    good, gone = Player(1, "P1", (0, 0, 0)), Player(2, "P2", (0, 0, 0))
    from core.types import TrackState
    gone.state = TrackState.LOST
    grades = {1: [], 2: []}
    for i in range(20 * 30):
        t = i / 30
        ref = choreo.reference_at(max(0.0, t - 0.2))
        good.pose = PoseObs(ref.keypoints, ref.confidence, t, aspect=1.0)
        for ev in scorer.update({1: good, 2: gone}, t, choreo):
            grades[ev.pid].append(ev.grade)
    assert grades[1] and all(g is Grade.PERFECT for g in grades[1])
    assert grades[2] and all(g is Grade.MISS for g in grades[2])


def test_tracker_keeps_identities_through_crossings():
    cfg = Config()
    tracker = PlayerTracker(cfg)
    mock = MockPoseSource(scenario="cross", dropout=0.1, seed=3)
    frame = FrameData(np.zeros((720, 1280, 3), np.uint8), 0.0, 0)
    mapping, switches = {}, 0
    for i in range(30 * 40):  # 40 s = 8 crossings
        t = i / 30
        frame.index = i
        poses, faces = mock.generate(t, (720, 1280))
        players = tracker.update(frame, poses, faces, t)
        for tid, pose in zip(mock.true_ids, poses):
            pid = next((p.pid for p in players.values() if p.pose is pose), None)
            if pid is None:
                continue
            switches += int(tid in mapping and mapping[tid] != pid)
            mapping[tid] = pid
    assert switches == 0
    assert len(tracker.players) == 2


def test_tracker_ignores_spectator_while_player_lost():
    cfg = Config()
    tracker = PlayerTracker(cfg)
    frame = FrameData(np.zeros((720, 1280, 3), np.uint8), 0.0, 0)
    a, b = _obs(body_pose(), center=(0.3, 0.6)), _obs(body_pose(), center=(0.7, 0.6))
    tracker.update(frame, [a, b], [], 0.0)
    tracker.update(frame, [a], [], 0.1)          # player 2 leaves
    assert not tracker.players[2].is_active
    events = tracker.pop_events()
    assert (EventType.PLAYER_LOST, 2) in events
    tracker.update(frame, [a, b], [], 3.0)       # and comes back: same pid
    assert tracker.players[2].is_active and tracker.players[2].pose is b
