"""Faster moves with "move_bpm" (the music keeps its own "bpm"), and the moves start again when the
music is longer than the choreography."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from game_logic import song as song_module
from game_logic.song import Song, music_length, repeat_moves

MOVES = [{"beat": 8, "pose": "cactus", "beats": 4}, {"beat": 16, "pose": "t_pose", "beats": 4},
         {"beat": 24, "pose": "pencil", "beats": 4, "gold": True}]


def make_song(folder, **fields):
    path = Path(folder) / "song.json"
    path.write_text(json.dumps({"title": "Test", "bpm": 120, "offset": 0.0, "moves": MOVES, **fields}))
    return Song(path)


class MoveBpmTests(unittest.TestCase):
    def test_move_bpm_makes_the_moves_shorter_and_more_often(self):
        with tempfile.TemporaryDirectory() as folder:
            normal, fast = make_song(folder), make_song(folder, move_bpm=240)
        self.assertAlmostEqual(normal.moves[1].start - normal.moves[0].start, 4.0)    # 8 beats at 120 bpm
        self.assertAlmostEqual(fast.moves[1].start - fast.moves[0].start, 2.0)        # 8 beats at 240 bpm
        self.assertAlmostEqual(fast.moves[0].end - fast.moves[0].start, 1.0)

    def test_the_beat_pulse_keeps_the_music_tempo(self):
        with tempfile.TemporaryDirectory() as folder:
            fast = make_song(folder, move_bpm=240)
        self.assertEqual((fast.bpm, fast.move_bpm), (120, 240))
        self.assertAlmostEqual(fast.beat_phase(0.25), 0.5)                            # 120 bpm: a beat is 0.5 s

    def test_without_move_bpm_nothing_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(make_song(folder).move_bpm, 120)


class RepeatMovesTests(unittest.TestCase):
    def moves(self):
        with tempfile.TemporaryDirectory() as folder:
            return make_song(folder).moves               # no music file: not repeated

    def test_moves_start_again_with_the_same_step(self):
        moves = repeat_moves(self.moves(), music_length=31.0)                         # moves: 4, 8, 12 s
        self.assertEqual([m.key for m in moves], ["cactus", "t_pose", "pencil"] * 2 + ["cactus"])
        self.assertEqual([m.start for m in moves], [4, 8, 12, 16, 20, 24, 28])
        self.assertEqual([m.gold for m in moves[3:6]], [False, False, True])           # gold moves stay gold

    def test_no_move_after_the_music(self):
        moves = repeat_moves(self.moves(), music_length=21.0)
        self.assertLessEqual(moves[-1].end + config.GRADE_GRACE, 21.0)
        self.assertEqual(len(moves), 4)

    def test_music_shorter_than_the_moves_or_unknown(self):
        self.assertEqual(len(repeat_moves(self.moves(), music_length=10.0)), 3)
        self.assertEqual(len(repeat_moves(self.moves(), music_length=None)), 3)

    def test_a_song_with_music_is_repeated_until_the_music_ends(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(Song, "_load_audio", return_value=True), \
                patch.object(song_module, "music_length", return_value=60.0):
            song = make_song(folder, audio="song.ogg")
        self.assertEqual(len(song.moves), 14)                                         # 4, 8, ..., 56 s
        self.assertLessEqual(song.length, 60.0)


class MusicLengthTests(unittest.TestCase):
    def test_reads_the_length_of_an_ogg_file(self):
        path = Path(config.CHOREOGRAPHY_DIR) / "song_1.ogg"
        if not path.exists():
            self.skipTest("no song_1.ogg")
        self.assertAlmostEqual(music_length(path), 64.0, delta=0.5)

    def test_unknown_files_give_none(self):
        self.assertIsNone(music_length("does_not_exist.ogg"))
        self.assertIsNone(music_length(config.POSES_PATH))


if __name__ == "__main__":
    unittest.main()
