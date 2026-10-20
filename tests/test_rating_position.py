"""The score bar follows where the players stand, and each rating is right next to the score bar on
the side of that player's score (not above the player)."""
import unittest

import numpy as np

from identity_tracking.identity import new_player_entry
from scene.functions import RATING_COLORS, player_color
from scene.scene import SIDE_SWAP_MARGIN, Scene

WIDTH, HEIGHT = 1280, 720
BAR_LEFT, BAR_RIGHT = WIDTH // 2 - WIDTH // 8, WIDTH // 2 + WIDTH // 8


def players_at(x_1, x_2, score_1=100, score_2=100):
    players = {1: new_player_entry(1), 2: new_player_entry(2)}
    for player_id, x, score in ((1, x_1, score_1), (2, x_2, score_2)):
        players[player_id]["score"] = score
        if x is not None:
            players[player_id].update(visible=True, box=(x - 90, 170, 180, 500))
    return players


def draw(scene, players, ratings):
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame = scene.render(frame, list(players.values()))
    return scene.draw_ratings(frame, ratings, {i: 0.5 for i in ratings}, 1.0)


def middle_x(frame, color):
    xs = np.nonzero(np.all(frame == color, axis=2))[1]
    return xs.mean()


class ScoreSideTests(unittest.TestCase):
    def test_player_1_on_the_right_gets_the_right_part_of_the_bar(self):
        scene = Scene()
        players = players_at(950, 330, score_1=300, score_2=100)        # Player 1 stands right
        self.assertEqual(scene.update_sides(list(players.values())), [2, 1])
        frame = draw(scene, players, {})
        bar = frame[scene.score_top + 4, BAR_LEFT + 10:BAR_RIGHT - 10]
        color_1 = np.array(player_color(players[1]))
        self.assertTrue(np.all(bar[-20:] == color_1))                   # the right end is Player 1's colour

    def test_start_and_not_visible_keep_the_side(self):
        scene = Scene()
        self.assertEqual(scene.update_sides(list(players_at(None, None).values())), [1, 2])   # start: by number
        scene.update_sides(list(players_at(950, 330).values()))                                # swapped
        self.assertEqual(scene.update_sides(list(players_at(None, 330).values())), [2, 1])    # 1 out of view: kept

    def test_no_flipping_when_players_stand_close(self):
        scene = Scene()
        scene.update_sides(list(players_at(600, 700).values()))
        self.assertEqual(scene.update_sides(list(players_at(640 + SIDE_SWAP_MARGIN // 2, 640).values())), [1, 2])


class RatingNextToTheBarTests(unittest.TestCase):
    def test_rating_is_next_to_the_bar_on_the_players_side(self):
        scene = Scene()
        frame = draw(scene, players_at(950, 330), {1: "Miss", 2: "Perfect"})   # Player 1 right, Player 2 left
        perfect, miss = middle_x(frame, RATING_COLORS["Perfect"]), middle_x(frame, RATING_COLORS["Miss"])
        self.assertLess(perfect, BAR_LEFT)                                      # Player 2's rating left of the bar
        self.assertGreater(perfect, BAR_LEFT - 250)                             # ... right next to it
        self.assertGreater(miss, BAR_RIGHT)                                     # Player 1's rating right of it
        self.assertLess(miss, BAR_RIGHT + 250)

    def test_rating_is_not_above_the_player(self):
        scene = Scene()
        frame = draw(scene, players_at(1150, 200), {1: "Good"})                 # Player 1 far right
        self.assertLess(middle_x(frame, RATING_COLORS["Good"]), BAR_RIGHT + 250)  # at the bar, not at x = 1150

    def test_rating_stays_on_the_screen(self):
        scene = Scene()
        frame = np.zeros((480, 640, 3), np.uint8)                               # small camera image
        frame = scene.render(frame, list(players_at(500, 100).values()))
        frame = scene.draw_ratings(frame, {2: "Perfect", 1: "Perfect"}, {1: 0.0, 2: 0.0}, 1.0)   # popping: big
        gold = np.nonzero(np.all(frame == RATING_COLORS["Perfect"], axis=2))[1]
        self.assertGreaterEqual(gold.min(), 0)
        self.assertLess(gold.max(), 640)


if __name__ == "__main__":
    unittest.main()
