"""The debug drawings turn on by themselves DEBUG_SHOW_AT seconds into the song, for DEBUG_SHOW_FOR seconds."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import config
from game_logic.game import Game


def fake_game(state, song_time, debug=False):
    song = Mock()
    song.time.return_value = song_time
    return SimpleNamespace(state_name=state, song=song, debug=debug)


class DebugMomentTests(unittest.TestCase):
    def shown(self, state, t):
        return Game.debug_shown(fake_game(state, t))

    def test_on_from_25_to_30_seconds_into_the_song(self):
        at, length = config.DEBUG_SHOW_AT, config.DEBUG_SHOW_FOR
        self.assertFalse(self.shown("playing", at - 0.1))
        self.assertTrue(self.shown("playing", at))
        self.assertTrue(self.shown("playing", at + length - 0.1))
        self.assertFalse(self.shown("playing", at + length))

    def test_only_while_playing(self):
        self.assertFalse(self.shown("song_select", config.DEBUG_SHOW_AT + 1))
        self.assertFalse(self.shown("results", config.DEBUG_SHOW_AT + 1))

    def test_can_be_switched_off(self):
        old = config.DEBUG_SHOW_AT
        config.DEBUG_SHOW_AT = None
        try:
            self.assertFalse(self.shown("playing", 27.0))
        finally:
            config.DEBUG_SHOW_AT = old


if __name__ == "__main__":
    unittest.main()
