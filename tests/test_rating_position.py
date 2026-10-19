"""The rating ("Perfect", "Miss", ...) is drawn above the player's own place on the screen."""
import unittest

import numpy as np

from identity_tracking.identity import new_player_entry
from scene.functions import RATING_COLORS
from scene.scene import Scene

WIDTH, HEIGHT = 1280, 720


def players_at(x_1, x_2):
    players = {1: new_player_entry(1), 2: new_player_entry(2)}
    for player_id, x in ((1, x_1), (2, x_2)):
        if x is not None:
            players[player_id].update(visible=True, box=(x - 90, 170, 180, 500))
    return players


def text_x(ratings, players):
    """Middle (x) of the drawn rating text, for one rating."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    Scene().draw_ratings(frame, ratings, players)
    xs = np.nonzero(frame.any(axis=2))[1]
    return xs.mean()


class RatingPositionTests(unittest.TestCase):
    def test_rating_is_above_the_player_also_when_player_1_stands_right(self):
        players = players_at(950, 330)                                  # Player 1 on the right
        self.assertAlmostEqual(text_x({1: "Miss"}, players), 950, delta=25)
        self.assertAlmostEqual(text_x({2: "Perfect"}, players), 330, delta=25)

    def test_player_that_is_not_visible_gets_the_fixed_side(self):
        players = players_at(None, 330)
        self.assertAlmostEqual(text_x({1: "Good"}, players), WIDTH // 2 - WIDTH // 8, delta=25)

    def test_players_close_together_do_not_overlap(self):
        # both ratings at the same time, in two colours, so each text can be found
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        Scene().draw_ratings(frame, {1: "Perfect", 2: "Miss"}, players_at(640, 660))
        middle = {}
        for rating in ("Perfect", "Miss"):
            mask = np.all(frame == RATING_COLORS[rating], axis=2)
            middle[rating] = np.nonzero(mask)[1].mean()
        self.assertGreater(middle["Miss"] - middle["Perfect"], 150)      # Player 1 left of Player 2, apart


if __name__ == "__main__":
    unittest.main()
