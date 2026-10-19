"""The rating pop, the move timeline and the full-body hint at registration."""
import unittest
from types import SimpleNamespace

import numpy as np

from game_logic.states.common import full_body_visible
from identity_tracking.identity import new_player_entry
from scene import timeline
from scene.registration_screen import HINT, draw_registration
from scene.scene import POP_SIZE, POP_TIME, pop

POSE = {"nose": (0, 0), "left_shoulder": (-0.5, 0.5), "right_shoulder": (0.5, 0.5), "left_elbow": (-1.2, 0.5),
        "right_elbow": (1.2, 0.5), "left_wrist": (-2.0, 0.5), "right_wrist": (2.0, 0.5), "left_hip": (-0.35, 1.5),
        "right_hip": (0.35, 1.5), "left_knee": (-0.37, 2.3), "right_knee": (0.37, 2.3),
        "left_ankle": (-0.37, 3.1), "right_ankle": (0.37, 3.1)}


def move(start, end, gold=False, players=1):
    return SimpleNamespace(name="Test", key="t_pose", pose=POSE, pose2=POSE if players == 2 else None,
                           start=start, end=end, gold=gold, players=players)


class PopTests(unittest.TestCase):
    def test_starts_big_and_shrinks_quickly(self):
        self.assertAlmostEqual(pop(0.0, 1.0)[0], POP_SIZE)
        self.assertAlmostEqual(pop(POP_TIME, 1.0)[0], 1.0)
        self.assertGreater(pop(0.05, 1.0)[0], pop(0.1, 1.0)[0])

    def test_fades_out_at_the_end(self):
        self.assertEqual(pop(0.5, 1.0)[1], 1.0)
        self.assertLess(pop(0.9, 1.0)[1], 0.5)
        self.assertEqual(pop(1.0, 1.0)[1], 0.0)


class TimelineTests(unittest.TestCase):
    moves = [move(2.0, 3.0), move(4.0, 5.0, gold=True), move(9.0, 10.0, players=2)]

    def test_visible_moves(self):
        self.assertEqual(len(timeline.visible_moves(self.moves, 0.0)), 1)          # only the move at 2 s
        self.assertEqual(len(timeline.visible_moves(self.moves, 2.5)), 2)          # active + the gold move
        self.assertEqual(len(timeline.visible_moves(self.moves, 10.5)), 0)         # the song is over

    def draw(self, t, size=(1280, 720)):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        return timeline.draw_timeline(frame, self.moves, t, 0.0)

    def test_draws_at_both_camera_sizes(self):
        for size in ((1280, 720), (640, 480)):
            for t in (-3.0, 0.0, 2.5, 4.2, 9.5, 11.0):
                self.draw(t, size)                                                  # no crash

    def test_active_move_sits_on_the_hit_line(self):
        frame = self.draw(4.2)                                                      # the gold move is active
        (left, top, right, bottom), hit_x = timeline.timeline_layout(1280, 720)
        gold = np.all(frame == timeline.GOLD, axis=2)
        xs = np.nonzero(gold)[1]
        self.assertLess(abs(xs.min() + (xs[xs < hit_x + 100].max() - xs.min()) / 2 - hit_x), 60)

    def test_next_move_waits_at_the_right_end(self):
        # the only move is a duet 10 s away (more than LOOKAHEAD): it still waits at the right end
        frame = np.zeros((720, 1280, 3), np.uint8)
        timeline.draw_timeline(frame, [move(10.0, 11.0, players=2)], 0.0, 0.0)
        (left, top, right, bottom), hit_x = timeline.timeline_layout(1280, 720)
        xs = np.nonzero(np.all(frame == timeline.DUET, axis=2))[1]
        self.assertTrue(len(xs) > 0)
        self.assertGreater(xs.min(), right - 150)                                   # only near the right end


class FullBodyTests(unittest.TestCase):
    def player(self, ankles):
        entry = new_player_entry(1)
        entry["visible"] = True
        entry["box"] = (500, 150, 200, 500)
        entry["keypoints"].update(nose=(600, 200), left_ankle=(580, 640) if ankles else None,
                                  right_ankle=(620, 640) if ankles else None)
        return entry

    def test_full_body(self):
        self.assertTrue(full_body_visible(self.player(ankles=True)))
        self.assertFalse(full_body_visible(self.player(ankles=False)))

    def test_step_back_hint_only_when_feet_are_not_seen(self):
        for ankles, expected in ((False, True), (True, False)):
            frame = np.zeros((720, 1280, 3), np.uint8)
            players = {1: self.player(ankles), 2: new_player_entry(2)}
            draw_registration(frame, players, {1: None, 2: None}, {1: False, 2: False}, {1: 0.0, 2: 0.0},
                              False, {1: full_body_visible(players[1]), 2: True})
            self.assertEqual(bool(np.all(frame == HINT, axis=2).any()), expected)


if __name__ == "__main__":
    unittest.main()
