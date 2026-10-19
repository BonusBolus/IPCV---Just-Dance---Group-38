"""Duet awards, move boundaries, and feedback timing without a camera or music."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import config
from game_logic.states.playing import PlayingState
from game_logic.states.results import ResultsState
from identity_tracking.identity import new_player_entry
from scene.functions import draw_duet_bonus


class DuetFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.move = SimpleNamespace(key="duet", players=2, gold=False)
        self.song = Mock()
        self.song.time.return_value = 2.0
        self.song.is_finished.return_value = False
        self.song.grading_move.return_value = self.move
        self.grader = Mock()
        self.grader.processed_poses = {"duet": {}, "solo": {}}
        self.grader.grade_pose.side_effect = lambda points, key, player=0: (80, 60)[player]
        self.grader.interaction_met.return_value = True
        self.game = SimpleNamespace(song=self.song, pose_grader=self.grader, now=10.0,
                                    players={i: new_player_entry(i) for i in (1, 2)},
                                    debug=False, scene=Mock())
        for player in self.game.players.values():
            player.update(visible=True, keypoints_raw=np.zeros((17, 3)))
        self.state = PlayingState()
        self.state.enter(self.game)

    def finish(self):
        self.song.grading_move.return_value = None
        self.game.now += 1.0
        self.state.update(self.game)

    def test_success_awards_once_and_starts_popup_when_points_are_added(self):
        self.state.update(self.game)
        self.state.update(self.game)
        self.assertIsNone(self.state.duet_bonus_since)
        self.finish()
        self.assertEqual([self.game.players[i]["score"] for i in (1, 2)], [100, 75])
        self.assertEqual(self.game.players[1]["last_move_points"], 100)
        self.assertEqual(self.state.duet_bonus_since, self.game.now)
        self.state.finish_move(self.game)
        self.assertEqual(self.game.players[1]["score"], 100)

    def test_gold_duet_bonus_uses_doubled_points(self):
        self.move.gold = True
        self.state.update(self.game)
        self.finish()
        self.assertEqual([self.game.players[i]["score"] for i in (1, 2)], [200, 150])

    def test_bonus_requires_simultaneous_good_scores_and_interaction(self):
        for case in ("no interaction", "low score", "missing player", "nonfinite score"):
            with self.subTest(case=case):
                self.setUp()
                if case == "no interaction":
                    self.grader.interaction_met.return_value = False
                elif case == "low score":
                    self.grader.grade_pose.side_effect = [80, config.DUET_MIN_SCORE - 1]
                elif case == "missing player":
                    self.game.players[2]["visible"] = False
                else:
                    self.grader.grade_pose.side_effect = [80, float("nan")]
                self.state.update(self.game)
                self.finish()
                self.assertEqual(self.game.players[1]["score"], 80)
                self.assertIsNone(self.state.duet_bonus_since)

    def test_success_does_not_carry_into_next_duet_or_solo(self):
        for players in (1, 2):
            with self.subTest(players=players):
                self.setUp()
                self.state.update(self.game)
                following = SimpleNamespace(key="solo" if players == 1 else "duet",
                                            players=players, gold=False)
                self.song.grading_move.return_value = following
                self.grader.interaction_met.return_value = False
                self.state.update(self.game)
                self.assertFalse(self.state.interacted)
                self.finish()
                self.assertEqual(self.game.players[1]["score"], 180)

    def test_popup_expires_and_reentry_clears_it(self):
        self.state.update(self.game)
        self.finish()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.game.scene.render.side_effect = lambda image, players: image
        self.game.scene.draw_ratings.side_effect = lambda image, ratings, players=None: image
        with patch("game_logic.states.playing.draw_move_cards", side_effect=lambda game, image, t: image), \
             patch("game_logic.states.playing.draw_duet_bonus", side_effect=lambda image, progress: image) as popup:
            self.state.draw(self.game, frame)
            popup.assert_called_once()
            self.game.now += config.DUET_BONUS_TIME
            self.state.draw(self.game, frame)
            popup.assert_called_once()
        self.state.enter(self.game)
        self.assertIsNone(self.state.duet_bonus_since)

    def test_popup_fades_to_original_frame(self):
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        self.assertFalse(np.array_equal(draw_duet_bonus(frame.copy(), 0), frame))
        np.testing.assert_array_equal(draw_duet_bonus(frame.copy(), 1), frame)

    def test_final_move_bonus_popup_survives_transition_to_results(self):
        self.song.is_finished.return_value = True
        self.assertEqual(self.state.update(self.game), "results")
        self.game.states = {"playing": self.state}
        results = ResultsState()
        results.enter(self.game)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("game_logic.states.results.draw_duet_bonus", side_effect=lambda image, progress, center_y: image) as popup:
            results.draw(self.game, frame)
            popup.assert_called_once()
            self.game.now += config.DUET_BONUS_TIME
            results.draw(self.game, frame)
            popup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
