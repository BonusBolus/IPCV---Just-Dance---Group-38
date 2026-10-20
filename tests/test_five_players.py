"""1 to 5 players: registration waits for everyone, duets in pairs of neighbours, the score badges and
the ranking. With exactly 2 players the game stays as it was (see the other tests)."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import config
from game_logic.pose_grading import PoseGrader
from game_logic.states.playing import PlayingState
from game_logic.states.registration import RegistrationState
from game_logic.states.results import rank_players
from identity_tracking.identity import new_player_entry
from scene.functions import RATING_COLORS
from scene.scene import Scene
from test_points_go_to_the_right_player import POSES, SCALE, visible_player


def all_slots():
    return {i: new_player_entry(i) for i in range(1, 6)}


class RegistrationTests(unittest.TestCase):
    """hands_up() is replaced by player["hands"], so the test only checks who the game waits for."""

    def setUp(self):
        self.players = all_slots()
        self.game = SimpleNamespace(now=0.0, players=None,
                                    tracker=SimpleNamespace(reset=lambda: None, entries=self.players))
        self.state = RegistrationState()
        self.state.enter(self.game)
        patcher = patch("game_logic.states.registration.hands_up", lambda player: player.get("hands", False))
        patcher.start()
        self.addCleanup(patcher.stop)

    def step(self, seconds):
        result = None
        for _ in range(int(seconds / 0.1)):
            self.game.now += 0.1
            result = self.state.update(self.game) or result
        return result

    def join(self, *ids, hands=False):
        for i in ids:
            self.players[i].update(registered=True, visible=True, hands=hands)

    def test_nobody_there_does_not_start(self):
        self.assertIsNone(self.step(2 * config.HOLD_TIME))

    def test_one_player_alone_can_start(self):
        self.join(1, hands=True)
        self.assertEqual(self.step(config.HOLD_TIME + 0.3), "countdown")

    def test_waits_until_all_three_are_ready(self):
        self.join(1, 2, hands=True)
        self.join(3)
        self.assertIsNone(self.step(config.HOLD_TIME + 0.3))      # Player 3 is not ready yet
        self.players[3]["hands"] = True
        self.assertEqual(self.step(config.HOLD_TIME + 0.3), "countdown")

    def test_a_player_who_walked_away_is_not_waited_for(self):
        self.join(1, 2, hands=True)
        self.join(4)
        self.step(0.2)
        self.players[4]["visible"] = False                        # walked past, never ready
        self.assertIsNone(self.step(config.DROP_TIME - 1.0))      # still waiting for Player 4 ...
        self.assertEqual(self.step(1.5), "countdown")             # ... until gone for DROP_TIME


class DuetPairsTests(unittest.TestCase):
    """5 players, the REAL PoseGrader: from left to right the 1st+2nd and the 3rd+4th are a pair; the 5th
    does the left part alone. The player ids stand in a random order."""
    grader = PoseGrader()

    def play_duet(self, xs_and_parts, frames=10):
        """xs_and_parts: {player_id: (center x, "keypoints" or "keypoints_2")}.
        Returns (who met the interaction during the move, scores after the move, the state)."""
        pose = POSES["high_five"]
        players = {i: visible_player(i, pose[part], x) for i, (x, part) in xs_and_parts.items()}
        move = SimpleNamespace(key="high_five", players=2, gold=False)
        song = Mock()
        song.time.return_value = 1.0
        song.is_finished.return_value = False
        song.grading_move.return_value = move
        game = SimpleNamespace(song=song, pose_grader=self.grader, now=0.0, players=players, debug=False)
        state = PlayingState()
        state.enter(game)
        for _ in range(frames):
            game.now += 0.04
            state.update(game)
        interacted = set(state.interacted)
        song.grading_move.return_value = None   # the move ends: points are added
        state.update(game)
        return interacted, {i: players[i]["score"] for i in players}, state

    GAP = 3.1 * SCALE                           # the raised hands of a pair touch in the middle

    def test_pairs_of_neighbours_get_the_bonus_and_the_fifth_does_part_1(self):
        interacted, scores, state = self.play_duet({
            3: (150, "keypoints"), 1: (150 + self.GAP, "keypoints_2"),       # pair 1
            5: (650, "keypoints"), 2: (650 + self.GAP, "keypoints_2"),       # pair 2
            4: (1150, "keypoints"),                                          # alone: the left part
        })
        self.assertEqual(state.pairs, [(3, 1), (5, 2), (4,)])
        self.assertEqual(state.roles, {3: 0, 1: 1, 5: 0, 2: 1, 4: 0})
        self.assertEqual(interacted, {1, 2, 3, 5})                          # no bonus for the 5th
        self.assertGreater(min(scores.values()), 80)                        # everyone did their part well
        self.assertGreater(min(scores[i] for i in (1, 2, 3, 5)), scores[4])  # the bonus

    def test_only_the_pair_whose_hands_touch_gets_the_bonus(self):
        interacted, _, _ = self.play_duet({
            1: (150, "keypoints"), 2: (150 + self.GAP, "keypoints_2"),       # hands touch
            3: (650, "keypoints"), 4: (650 + 3 * self.GAP, "keypoints_2"),   # too far apart
        })
        self.assertEqual(interacted, {1, 2})

    def test_pairs_follow_where_the_players_stand(self):
        _, _, state = self.play_duet({
            1: (1150, "keypoints"), 2: (650 + self.GAP, "keypoints_2"),
            3: (650, "keypoints"), 4: (150, "keypoints"), 5: (150 + self.GAP, "keypoints_2"),
        })
        self.assertEqual(state.pairs, [(4, 5), (3, 2), (1,)])


class RankingTests(unittest.TestCase):
    def game_with_scores(self, scores):
        players = all_slots()
        for i, score in scores.items():
            players[i].update(registered=True, score=score)
        return SimpleNamespace(players=players)

    def test_ranking_from_high_to_low_with_shared_places(self):
        ranking = rank_players(self.game_with_scores({1: 300, 2: 900, 3: 300, 5: 50}))
        self.assertEqual([(place, player["id"]) for place, player in ranking], [(1, 2), (2, 1), (2, 3), (4, 5)])

    def test_only_players_in_the_game(self):
        self.assertEqual([player["id"] for _, player in rank_players(self.game_with_scores({4: 10}))], [4])


class BadgeTests(unittest.TestCase):
    WIDTH, HEIGHT = 1280, 720

    def players_at(self, xs):
        """xs: {player_id: x}; these players are registered and visible."""
        players = all_slots()
        for i, x in xs.items():
            players[i].update(registered=True, visible=True, score=100 * i, box=(x - 60, 200, 120, 400))
        return list(players.values())

    def test_badges_in_the_order_the_players_stand(self):
        scene = Scene()
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        scene.render(frame, self.players_at({1: 900, 2: 200, 3: 600}))
        self.assertEqual(scene.sides, [2, 3, 1])
        self.assertEqual(sorted(scene.badges, key=lambda i: scene.badges[i][0]), [2, 3, 1])

    def test_rating_under_the_players_badge(self):
        scene = Scene()
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        frame = scene.render(frame, self.players_at({1: 900, 2: 200, 3: 600, 4: 1100, 5: 400}))
        frame = scene.draw_ratings(frame, {4: "Perfect"}, {4: 0.5}, 1.0)
        xs = np.nonzero(np.all(frame == RATING_COLORS["Perfect"], axis=2))[1]
        badge_x, _, badge_w = scene.badges[4]
        self.assertLess(abs(xs.mean() - badge_x), badge_w / 4)
        self.assertEqual(sorted(scene.badges, key=lambda i: scene.badges[i][0])[-1], 4)   # 4 stands right

    def test_two_players_keep_the_score_bar(self):
        scene = Scene()
        scene.render(np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8), self.players_at({1: 900, 2: 200}))
        self.assertIsNone(scene.badges)
        self.assertEqual(scene.sides, [2, 1])

    def test_neighbours_only_swap_when_clearly_the_other_way_round(self):
        scene = Scene()
        scene.update_sides(self.players_at({1: 200, 2: 600, 3: 1000})[:3])
        self.assertEqual(scene.update_sides(self.players_at({1: 200, 2: 620, 3: 600})[:3]), [1, 2, 3])
        self.assertEqual(scene.update_sides(self.players_at({1: 200, 2: 700, 3: 600})[:3]), [1, 3, 2])

    def test_a_player_out_of_view_keeps_their_place(self):
        scene = Scene()
        scene.update_sides(self.players_at({1: 200, 2: 600, 3: 1000})[:3])
        players = self.players_at({1: 200, 3: 1000})[:3]
        players[1]["registered"] = True                           # Player 2 is in the game but not visible
        self.assertEqual(scene.update_sides(players), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
