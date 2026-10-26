"""The debug drawings turn on by themselves on the results screen (the end of the demo video), and optionally
for a moment in the song (DEBUG_SHOW_AT). With --record the song stops so that the results fill the end."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from game_logic.game import Game
from game_logic.states.playing import PlayingState
from game_logic.states.results import ResultsState


def fake_game(state, song_time, debug=False):
    song = Mock()
    song.time.return_value = song_time
    return SimpleNamespace(state_name=state, song=song, debug=debug)


class DebugMomentTests(unittest.TestCase):
    def shown(self, state, t=10.0):
        return Game.debug_shown(fake_game(state, t))

    def test_on_the_results_screen(self):
        with patch.object(config, "DEBUG_IN_RESULTS", True), patch.object(config, "DEBUG_SHOW_AT", None):
            self.assertTrue(self.shown("results"))
            self.assertFalse(self.shown("playing"))          # not during the song
            self.assertFalse(self.shown("song_select"))

    def test_results_without_debug(self):
        with patch.object(config, "DEBUG_IN_RESULTS", False), patch.object(config, "DEBUG_SHOW_AT", None):
            self.assertFalse(self.shown("results"))

    def test_a_moment_in_the_song_when_asked(self):
        with patch.object(config, "DEBUG_SHOW_AT", 3.0), patch.object(config, "DEBUG_SHOW_FOR", 5.0):
            self.assertFalse(self.shown("playing", 2.9))
            self.assertTrue(self.shown("playing", 3.0))
            self.assertTrue(self.shown("playing", 7.9))
            self.assertFalse(self.shown("playing", 8.0))


class RecordingEndTests(unittest.TestCase):
    """--record: the song stops at the deadline (RECORD_RESULTS_TIME before the video ends) and the results stay."""

    def playing_game(self, song_time, now, deadline):
        song = Mock()
        song.time.return_value = song_time
        song.play_length = song.length = 100.0
        song.is_finished.return_value = False
        song.grading_move.return_value = None
        game = SimpleNamespace(song=song, pose_grader=Mock(processed_poses={}), now=now, debug=False,
                               players={}, song_deadline=deadline)
        return game

    def test_the_song_stops_at_the_deadline(self):
        with patch("game_logic.states.playing.PLAYER_IDS", ()):
            state = PlayingState()
            game = self.playing_game(10.0, 100.0, deadline=105.0)
            state.enter(game)
            self.assertIsNone(state.update(game))                          # 5 s before the deadline: still playing
            game.now, game.song.time.return_value = 104.5, 14.5
            state.update(game)
            game.song.fade_out.assert_called_once()                        # the music fades out
            game.now, game.song.time.return_value = 105.0, 15.0
            self.assertEqual(state.update(game), "results")                # the deadline: the results

    def test_while_recording_there_is_no_play_again_hint(self):
        state = ResultsState()
        game = SimpleNamespace(song=Mock(), now=0.0, recorder=object(), players={1: {"score": 2}, 2: {"score": 1}},
                               states={"playing": SimpleNamespace(duet_bonus_since=None)})
        state.enter(game)
        game.now = 50.0
        with patch("game_logic.states.results.draw_results", return_value="image") as draw:
            state.draw(game, "image")
        self.assertFalse(draw.call_args.args[4])                 # show_hint (image, players, winner, progress, show_hint)

    def test_while_recording_the_results_stay(self):
        state = ResultsState()
        game = SimpleNamespace(song=Mock(), now=0.0, recorder=object(), players={},
                               states={"playing": SimpleNamespace(duet_bonus_since=None)})
        state.enter(game)
        for t in (5.0, 30.0, 100.0):
            game.now = t
            self.assertIsNone(state.update(game))


if __name__ == "__main__":
    unittest.main()
