"""Points of a move: one lucky frame is not enough, holding the pose is."""
import unittest

import config
from game_logic.states.playing import PlayingState


def state_with(scores_1, scores_2=()):
    state = PlayingState()
    state.frame_scores = {1: list(scores_1), 2: list(scores_2)}
    return state


class MoveScoreTests(unittest.TestCase):
    def test_one_lucky_frame_is_not_enough(self):
        # 40 frames: wrong pose (20 points), with one perfect frame in the middle
        state = state_with([20] * 20 + [100] + [20] * 19)
        self.assertLess(state.move_score(1), 40)          # "Miss", with the best frame it was 100

    def test_holding_the_pose_counts(self):
        # moving into the pose for 60% of the move, then holding it well for 40%
        state = state_with([30] * 24 + [90] * 16)
        self.assertGreaterEqual(state.move_score(1), 85)  # "Perfect"

    def test_player_that_was_not_seen_gets_zero(self):
        state = state_with([80, 90], [])
        self.assertEqual(state.move_score(2), 0.0)

    def test_constant_score_stays_the_same(self):
        state = state_with([65] * 10)
        self.assertAlmostEqual(state.move_score(1), 65)

    def test_percentile_comes_from_config(self):
        self.assertTrue(0 < config.SCORE_PERCENTILE < 100)


if __name__ == "__main__":
    unittest.main()
