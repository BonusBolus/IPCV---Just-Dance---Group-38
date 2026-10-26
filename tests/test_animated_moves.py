"""Animated moves: a list of poses that the players follow. Every step is graded; correct steps in a row
make a chain with extra points, and the motion per player must go the right way (direction check)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import config
from game_logic.motion import PlayerMotion, flow
from game_logic.song import Song, animated_move, repeat_moves
from game_logic.states.playing import MIRROR_SWAP, PlayingState
from identity_tracking.identity import KEYPOINT_NAMES, new_player_entry
from scene.pose_figure import animated_pose

POSES = json.load(open(config.POSES_PATH))
KAZACHOK = ["kazachok_left", "kazachok_right", "kazachok_left", "kazachok_right"]
SCALE = 60                                      # pixels per pose unit (shoulder width = 60 px)


def keypoints_raw(pose, center_x=640, nose_y=150):
    """(17, 3) keypoints like YOLO gives them on the mirrored camera image (as in the other tests)."""
    kp = np.zeros((17, 3), np.float32)
    for i, name in enumerate(KEYPOINT_NAMES):
        point = pose.get(name)
        if point is not None:
            kp[i] = (center_x + point[0] * SCALE, nose_y + point[1] * SCALE, 0.9)
    return kp[MIRROR_SWAP]


class SongFormatTests(unittest.TestCase):
    def steps(self):
        return animated_move({"beat": 0, "poses": KAZACHOK, "name": "Kazachok"}, 3, 1.0, 0.5, POSES, "test")

    def test_every_pose_is_one_step_of_one_music_beat(self):
        steps = self.steps()
        self.assertEqual([s.key for s in steps], KAZACHOK)
        self.assertEqual([s.start for s in steps], [1.0, 1.5, 2.0, 2.5])
        self.assertEqual([s.end - s.start for s in steps], [0.5] * 4)
        self.assertEqual({(s.group, s.steps, s.group_name) for s in steps}, {(3, 4, "Kazachok")})
        self.assertEqual([s.step for s in steps], [0, 1, 2, 3])

    def test_the_steps_follow_each_other_exactly(self):
        # 127 bpm: one beat is 0.47244... s, which is not exact in a computer; no gap or overlap anyway
        steps = animated_move({"beat": 0, "poses": KAZACHOK}, 0, 68.031, 60 / 127, POSES, "test")
        for a, b in zip(steps, steps[1:]):
            self.assertEqual(a.end, b.start)

    def test_each_step_knows_the_pose_before_and_after(self):
        steps = self.steps()
        self.assertIsNone(steps[0].prev_pose)
        self.assertIs(steps[1].prev_pose, POSES["kazachok_left"]["keypoints"])
        self.assertIs(steps[2].next_pose, POSES["kazachok_right"]["keypoints"])
        self.assertIsNone(steps[3].next_pose)

    def test_repeated_songs_keep_the_animated_moves(self):
        moves = self.steps()
        repeated = repeat_moves(moves, music_length=10.0)
        self.assertGreater(len(repeated), len(moves))
        self.assertEqual([m.step for m in repeated[4:8]], [0, 1, 2, 3])
        self.assertIs(repeated[5].prev_pose, moves[1].prev_pose)

    def test_wrong_animated_moves_stop_at_start_up(self):
        for poses in (["kazachok_left"], ["kazachok_left", "no_such_pose"], ["kazachok_left", "high_five"]):
            with self.assertRaises(ValueError):
                animated_move({"beat": 0, "poses": poses}, 0, 0.0, 0.5, POSES, "test")

    def test_a_song_file_with_an_animated_move(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.json"
            path.write_text(json.dumps({"title": "T", "bpm": 120, "move_bpm": 360, "moves": [
                {"beat": 12, "pose": "cactus", "beats": 4}, {"beat": 24, "poses": KAZACHOK, "name": "Kazachok"},
                {"beat": 48, "pose": "t_pose", "beats": 4}]}))
            song = Song(path)
        self.assertEqual(len(song.moves), 6)
        self.assertAlmostEqual(song.moves[1].start, 4.0)                 # beat 24 at 360 move-bpm
        self.assertAlmostEqual(song.moves[2].start - song.moves[1].start, 0.5)   # one beat of the music (120 bpm)


class AnimationTests(unittest.TestCase):
    left, right = POSES["kazachok_left"]["keypoints"], POSES["kazachok_right"]["keypoints"]

    def test_holds_the_pose_first_then_moves_to_the_next(self):
        self.assertIs(animated_pose(self.left, self.right, 0.3), self.left)
        end = animated_pose(self.left, self.right, 1.0)
        np.testing.assert_allclose(end["left_wrist"], self.right["left_wrist"])
        # halfway the second half of the step, the figure is halfway between the two poses
        # (checked on the body point that moves the most between them)
        name = max((n for n in self.left if self.left[n] is not None and self.right.get(n) is not None),
                   key=lambda n: np.hypot(*np.subtract(self.right[n], self.left[n])))
        a, b = np.array(self.left[name]), np.array(self.right[name])
        middle = np.array(animated_pose(self.left, self.right, 0.75)[name])
        self.assertAlmostEqual(float((middle - a) @ (b - a) / ((b - a) @ (b - a))), 0.5)

    def test_the_last_step_does_not_move(self):
        self.assertIs(animated_pose(self.left, None, 0.9), self.left)


class FlowTests(unittest.TestCase):
    left, right = POSES["kazachok_left"]["keypoints"], POSES["kazachok_right"]["keypoints"]

    def figure_motion(self, factor):
        """The displacement of the figure from left to right, times factor, as a (17, 2) array."""
        out = np.zeros((17, 2))
        for i, name in enumerate(KEYPOINT_NAMES):
            if self.left.get(name) is not None and self.right.get(name) is not None:
                out[i] = factor * (np.array(self.right[name]) - self.left[name])
        return out

    def test_same_way_opposite_way_and_not_moving(self):
        self.assertAlmostEqual(flow(self.figure_motion(1.0), self.left, self.right), 1.0)
        self.assertAlmostEqual(flow(self.figure_motion(-1.0), self.left, self.right), -1.0)
        self.assertEqual(flow(np.zeros((17, 2)), self.left, self.right), 0.0)
        self.assertEqual(flow(None, self.left, self.right), 0.0)

    def test_a_smaller_movement_in_the_right_direction_counts(self):
        self.assertAlmostEqual(flow(self.figure_motion(0.5), self.left, self.right), 1.0)

    def test_nothing_to_check_when_the_poses_are_the_same(self):
        self.assertIsNone(flow(np.zeros((17, 2)), self.left, self.left))


class PlayerMotionTests(unittest.TestCase):
    def test_motion_is_in_shoulder_widths(self):
        pose = POSES["t_pose"]["keypoints"]
        for scale in (60, 120):                         # far from and close to the camera
            motion = PlayerMotion(smoothing=1.0)
            a = keypoints_raw(pose)[MIRROR_SWAP]
            a[:, :2] = (a[:, :2] - 640) * scale / SCALE + 640
            b = a.copy()
            b[:, 0] += scale                                # everything moves one shoulder width to the right
            motion.update(1, a, 0.0)
            motion.update(1, b, 0.04)
            seen = a[:, 2] > 0                             # the points the pose has (no eyes, ears, ...)
            np.testing.assert_allclose(motion.displacement(1, 0.0, 1.0)[seen, 0], 1.0, atol=1e-5)

    def test_a_player_out_of_view_does_not_make_a_jump(self):
        motion = PlayerMotion(smoothing=1.0)
        a = keypoints_raw(POSES["t_pose"]["keypoints"])[MIRROR_SWAP]
        motion.update(1, a, 0.0)
        motion.update(1, None, 0.04)                       # not seen
        b = a.copy()
        b[:, 0] += 300
        motion.update(1, b, 0.08)                          # back, somewhere else
        self.assertIsNone(motion.displacement(1, 0.0, 1.0))


class ChainTests(unittest.TestCase):
    """Two players get the same pose score; player 1 follows the moving figure, player 2 stands still."""

    def play(self, scores=None):
        steps = animated_move({"beat": 0, "poses": KAZACHOK, "name": "Kazachok"}, 0, 1.0, 0.5, POSES, "test")
        song = Mock()
        song.length = 100.0
        song.is_finished.return_value = False
        song.grading_move.side_effect = lambda t: next((s for s in steps if s.start <= t < s.end), None)
        grader = Mock(processed_poses={key: None for key in KAZACHOK})
        score_list = list(scores or [90.0] * 4)
        grader.grade_pose.side_effect = lambda kp, key, player=0: score_list[
            next(i for i, s in enumerate(steps) if s.start <= song.time.return_value < s.end)]
        players = {1: new_player_entry(1), 2: new_player_entry(2)}
        game = SimpleNamespace(song=song, pose_grader=grader, now=0.0, players=players, debug=False)
        song.time.return_value = 0.0
        state = PlayingState()
        state.enter(game)
        t = 0.0
        while t < 3.6:
            song.time.return_value = t
            game.now = t
            follow = next((s for s in steps if s.start <= t - 0.08 < s.end), None)   # 80 ms reaction time
            pose_1 = (animated_pose(follow.pose, follow.next_pose, (t - 0.08 - follow.start) / 0.5)
                      if follow else (steps[-1].pose if t > 2 else steps[0].pose))
            for player_id, pose, x in ((1, pose_1, 400), (2, steps[0].pose, 900)):
                players[player_id].update(visible=True, registered=True, keypoints_raw=keypoints_raw(pose, x),
                                          box=(x - 90, 100, 180, 400))
            state.update(game)
            t += 0.04
        return state, players

    def test_following_the_figure_builds_a_chain(self):
        state, players = self.play()
        self.assertEqual(state.chain[1], 4)
        self.assertEqual(players[1]["score"], 4 * 90 + config.CHAIN_BONUS * (0 + 1 + 2 + 3))

    def test_standing_still_breaks_the_chain(self):
        state, players = self.play()
        self.assertLess(state.flows[2], config.FLOW_MIN)          # the right pose score, but no movement
        self.assertEqual(players[2]["score"], 4 * 90)              # no chain bonus

    def test_a_bad_step_breaks_the_chain(self):
        state, players = self.play(scores=[90, 40, 90, 90])
        self.assertEqual(players[1]["score"], 90 + 40 + 90 + 90 + config.CHAIN_BONUS * 1)   # only steps 3-4 in a row

    def test_without_the_direction_check_standing_still_is_enough(self):
        old = config.FLOW_CHECK
        config.FLOW_CHECK = False
        try:
            state, players = self.play()
        finally:
            config.FLOW_CHECK = old
        self.assertEqual(players[2]["score"], 4 * 90 + config.CHAIN_BONUS * 6)


if __name__ == "__main__":
    unittest.main()
