"""The bonus star: it appears now and then, floats across at hand height, and the first player who
touches it with a hand gets the bonus once."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import config
from game_logic.bonus_star import BonusStar
from game_logic.states.playing import PlayingState
from identity_tracking.identity import new_player_entry
from scene.star import draw_star, draw_star_burst

SIZE = (1280, 720)
SONG_LENGTH = 200.0


def player(player_id, center_x, wrist=None):
    entry = new_player_entry(player_id)
    entry.update(visible=True, registered=True, box=(center_x - 90, 150, 180, 520))
    entry["keypoints"].update(left_shoulder=(center_x - 40, 260), right_shoulder=(center_x + 40, 260))
    if wrist is not None:
        entry["keypoints"]["right_wrist"] = wrist
    return entry


def run(star, players, start, end, step=0.04):
    """Update the star from song time start to end; returns [(time, player id)] of the grabs."""
    grabs, t = [], start
    while t < end:
        player_id = star.update(t, players, SIZE, SONG_LENGTH)
        if player_id is not None:
            grabs.append((t, player_id))
        t += step
    return grabs


class StarTests(unittest.TestCase):
    def setUp(self):
        self.star = BonusStar(seed=1)
        self.players = [player(1, 350), player(2, 930)]

    def test_no_star_before_the_first_one_is_due(self):
        run(self.star, self.players, 0.0, self.star.next_start - 0.1)
        self.assertIsNone(self.star.position)

    def test_the_star_floats_across_at_hand_height(self):
        t0 = self.star.next_start
        self.star.update(t0, self.players, SIZE, SONG_LENGTH)
        xs, ys = [], []
        for k in range(1, 120):
            self.star.update(t0 + k * 0.04, self.players, SIZE, SONG_LENGTH)
            xs.append(self.star.position[0])
            ys.append(self.star.position[1])
        self.assertGreater(abs(xs[-1] - xs[0]), 0.7 * SIZE[0] * (119 * 0.04) / config.STAR_TIME)   # across the screen
        self.assertGreaterEqual(min(ys), 150 - 0.04 * SIZE[1] - 1)       # not higher than just above the heads
        self.assertLessEqual(max(ys), 261)                              # not lower than the shoulders

    def test_nobody_touches_it_and_it_goes_away(self):
        t0 = self.star.next_start
        self.assertEqual(run(self.star, self.players, t0, t0 + config.STAR_TIME + 0.1), [])
        self.assertIsNone(self.star.position)
        self.assertGreater(self.star.next_start, t0 + config.STAR_TIME)  # the next one comes later

    def test_a_hand_on_the_star_gives_the_bonus_once(self):
        t0 = self.star.next_start
        self.star.update(t0, self.players, SIZE, SONG_LENGTH)
        self.star.update(t0 + 1.0, self.players, SIZE, SONG_LENGTH)
        x, y = self.star.position
        self.players[1]["keypoints"]["left_wrist"] = (x + 10, y)       # player 2 reaches for it
        self.assertEqual(self.star.update(t0 + 1.0, self.players, SIZE, SONG_LENGTH), 2)
        self.assertIsNone(self.star.position)                          # gone
        self.assertEqual(self.star.grab[0], 2)
        self.assertEqual(run(self.star, self.players, t0 + 1.04, t0 + 4.0), [])   # not a second time

    def test_the_closest_hand_wins(self):
        t0 = self.star.next_start
        self.star.update(t0, self.players, SIZE, SONG_LENGTH)
        self.star.update(t0 + 1.0, self.players, SIZE, SONG_LENGTH)
        x, y = self.star.position
        self.players[0]["keypoints"]["right_wrist"] = (x + 30, y)
        self.players[1]["keypoints"]["left_wrist"] = (x + 5, y)
        self.assertEqual(self.star.update(t0 + 1.0, self.players, SIZE, SONG_LENGTH), 2)

    def test_a_hand_far_away_does_not_grab_it(self):
        t0 = self.star.next_start
        self.star.update(t0, self.players, SIZE, SONG_LENGTH)
        x, y = self.star.position
        self.players[0]["keypoints"]["right_wrist"] = (x, y + config.STAR_REACH * SIZE[1] + 20)
        self.assertIsNone(self.star.update(t0 + 0.04, self.players, SIZE, SONG_LENGTH))

    def test_no_star_at_the_end_of_the_song_or_when_nobody_is_seen(self):
        late = BonusStar(seed=1)
        late.next_start = SONG_LENGTH - 2.0
        late.update(SONG_LENGTH - 2.0, self.players, SIZE, SONG_LENGTH)
        self.assertIsNone(late.position)
        nobody = BonusStar(seed=1)
        gone = [dict(p, visible=False) for p in self.players]
        nobody.update(nobody.next_start, gone, SIZE, SONG_LENGTH)
        self.assertIsNone(nobody.start)

    def test_the_next_star_comes_from_the_other_side(self):
        first = self.star.from_left
        t0 = self.star.next_start
        run(self.star, self.players, t0, t0 + config.STAR_TIME + 0.1)
        self.assertNotEqual(self.star.from_left, first)


class StarInTheGameTests(unittest.TestCase):
    def test_the_bonus_goes_to_the_score(self):
        song = Mock()
        song.time.return_value = 0.0
        song.length = SONG_LENGTH
        song.is_finished.return_value = False
        song.grading_move.return_value = None
        players = {1: player(1, 350), 2: player(2, 930)}
        game = SimpleNamespace(song=song, pose_grader=Mock(processed_poses={}), now=0.0, players=players, debug=False)
        state = PlayingState()
        state.enter(game)
        t0 = state.star.next_start
        song.time.return_value = t0 + 1.0
        state.update(game)                                             # the star appears
        song.time.return_value = t0 + 1.04
        state.update(game)
        x, y = state.star.position
        players[1]["keypoints"]["right_wrist"] = (x, y)
        state.update(game)
        self.assertEqual(players[1]["score"], config.STAR_BONUS)
        self.assertEqual(players[2]["score"], 0)


class DrawStarTests(unittest.TestCase):
    def test_drawing_at_the_edge_does_not_crash(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        for center in ((0, 0), (1279, 719), (-30, 300), (1300, 360)):
            draw_star(frame, center, 25, 1.0, 0.5)
            draw_star_burst(frame, center, "+100", (255, 255, 255), 0.5)

    def test_the_star_is_gold(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        draw_star(frame, (640, 360), 25, 0.0)
        self.assertGreater(frame[360, 640, 2], 200)                    # red + green = gold (BGR)
        self.assertGreater(frame[360, 640, 1], 150)


if __name__ == "__main__":
    unittest.main()
