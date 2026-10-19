"""Song selection: which hand is up, browsing with one hand, choosing with both hands."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import config
from game_logic.states.common import raised_hands
from game_logic.states.song_select import SongSelectState
from identity_tracking.identity import new_player_entry

NOSE = (300, 200)
SCREEN_LEFT_UP, SCREEN_RIGHT_UP = (240, 100), (360, 100)     # wrists above the nose, left / right of it
DOWN_LEFT, DOWN_RIGHT = (240, 330), (360, 330)                 # wrists below the nose


def player(screen_left_wrist, screen_right_wrist):
    """A visible player. Names like YOLO in the mirrored camera image: the wrist on the LEFT of
    the screen is called right_wrist."""
    entry = new_player_entry(1)
    entry["visible"] = True
    entry["keypoints"].update(nose=NOSE, right_wrist=screen_left_wrist, left_wrist=screen_right_wrist)
    return entry


class RaisedHandsTests(unittest.TestCase):
    def test_sides_come_from_the_screen_not_from_the_names(self):
        self.assertEqual(raised_hands(player(SCREEN_LEFT_UP, DOWN_RIGHT)), "left")
        self.assertEqual(raised_hands(player(DOWN_LEFT, SCREEN_RIGHT_UP)), "right")

    def test_both_and_none(self):
        self.assertEqual(raised_hands(player(SCREEN_LEFT_UP, SCREEN_RIGHT_UP)), "both")
        self.assertIsNone(raised_hands(player(DOWN_LEFT, DOWN_RIGHT)))

    def test_missing_points(self):
        entry = player(SCREEN_LEFT_UP, None)
        self.assertEqual(raised_hands(entry), "left")
        entry["keypoints"]["nose"] = None
        self.assertIsNone(raised_hands(entry))


class SongSelectTests(unittest.TestCase):
    def setUp(self):
        self.songs = [Mock(name=f"song{i}", music_playing=Mock(return_value=True)) for i in range(3)]
        self.game = SimpleNamespace(songs=self.songs, song_index=0, song=self.songs[0], now=0.0,
                                    players={1: player(DOWN_LEFT, DOWN_RIGHT), 2: new_player_entry(2)})
        self.state = SongSelectState()
        self.state.enter(self.game)

    def hold(self, screen_left_wrist, screen_right_wrist, seconds, fps=25):
        """Hold a pose for some seconds; returns what update() returned last."""
        self.game.players[1] = player(screen_left_wrist, screen_right_wrist)
        result = None
        for _ in range(int(seconds * fps)):
            self.game.now += 1 / fps
            result = self.state.update(self.game)
        return result

    def test_enter_plays_a_preview(self):
        self.songs[0].preview.assert_called()

    def test_right_hand_goes_to_the_next_song_once(self):
        self.hold(DOWN_LEFT, SCREEN_RIGHT_UP, config.BROWSE_HOLD_TIME * 4)    # keep it up a long time
        self.assertEqual(self.game.song_index, 1)                              # still only one step
        self.assertIs(self.game.song, self.songs[1])
        self.songs[1].preview.assert_called()

    def test_lower_the_hand_for_the_next_step(self):
        for _ in range(2):
            self.hold(DOWN_LEFT, SCREEN_RIGHT_UP, config.BROWSE_HOLD_TIME + 0.2)
            self.hold(DOWN_LEFT, DOWN_RIGHT, 0.2)
        self.assertEqual(self.game.song_index, 2)

    def test_left_hand_goes_back_and_wraps_around(self):
        self.hold(SCREEN_LEFT_UP, DOWN_RIGHT, config.BROWSE_HOLD_TIME + 0.2)
        self.assertEqual(self.game.song_index, 2)                              # from the first to the last song

    def test_short_hand_up_does_nothing(self):
        self.hold(DOWN_LEFT, SCREEN_RIGHT_UP, config.BROWSE_HOLD_TIME / 2)
        self.hold(DOWN_LEFT, DOWN_RIGHT, 0.2)
        self.assertEqual(self.game.song_index, 0)

    def test_both_hands_choose_the_song(self):
        self.hold(DOWN_LEFT, SCREEN_RIGHT_UP, config.BROWSE_HOLD_TIME + 0.2)
        self.hold(DOWN_LEFT, DOWN_RIGHT, 0.2)
        self.assertIsNone(self.hold(SCREEN_LEFT_UP, SCREEN_RIGHT_UP, config.HOLD_TIME / 2))
        self.assertEqual(self.hold(SCREEN_LEFT_UP, SCREEN_RIGHT_UP, config.HOLD_TIME), "registration")
        self.assertEqual(self.game.song_index, 1)                              # both hands do not browse

    def test_exit_stops_the_preview(self):
        self.state.exit(self.game)
        self.songs[0].stop.assert_called()

    def test_preview_starts_again_at_the_end(self):
        self.songs[0].preview.reset_mock()
        self.songs[0].music_playing.return_value = False
        self.hold(DOWN_LEFT, DOWN_RIGHT, 0.1)
        self.songs[0].preview.assert_called()


if __name__ == "__main__":
    unittest.main()
