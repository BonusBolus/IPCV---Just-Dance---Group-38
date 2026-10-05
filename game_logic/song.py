"""The song: which pose the players must do at which time, and the music.

Song file format: see choreography/song_1.json and docs/game_structure.md, section 5.
The moves are on beats, the poses come from choreography/poses.json.

All methods that need the song time get it as argument `t` (seconds). Call `song.time()`
once per frame and pass that value on, so everything in one frame uses the same time.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import config

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # no "Hello from pygame" message
try:
    import pygame
except ImportError:
    pygame = None


@dataclass(eq=False)
class Move:
    name: str      # shown on screen, for example "Cactus"
    key: str       # name in poses.json, for example "cactus" (also used by the PoseGrader)
    pose: dict     # from poses.json: {"nose": (x, y), ...}, nose at (0, 0), None = not used
    start: float   # seconds: the move becomes active, the player must be in the pose
    end: float     # seconds: the move is over
    gold: bool     # gold move: double points


def moves_from_beats(data, poses, file_name):
    """Beat format: {"bpm": 120, "offset": 0.0, "moves": [{"beat": 4, "pose": "cactus", "beats": 2}, ...]}"""
    beat_time = 60 / data["bpm"]
    offset = data.get("offset", 0.0)
    moves = []
    for i, item in enumerate(data["moves"]):
        if item["pose"] not in poses:
            raise ValueError(f'{file_name} move {i}: pose "{item["pose"]}" not found in poses.json')
        pose = poses[item["pose"]]
        start = offset + item["beat"] * beat_time
        end = start + item.get("beats", config.DEFAULT_MOVE_BEATS) * beat_time
        moves.append(Move(pose["name"], item["pose"], pose["keypoints"], start, end, item.get("gold", False)))
    return moves


def check_moves(moves, file_name):
    """Stop at start-up with a clear message, instead of crashing in the middle of a song."""
    if not moves:
        raise ValueError(f"{file_name}: the song has no moves")
    moves.sort(key=lambda move: move.start)
    for i in range(1, len(moves)):
        if moves[i].start < moves[i - 1].end:
            raise ValueError(f"{file_name} move {i}: starts before move {i - 1} has ended")


class Song:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        poses = json.loads(Path(config.POSES_PATH).read_text(encoding="utf-8"))
        self.moves = moves_from_beats(data, poses, self.path.name)
        check_moves(self.moves, self.path.name)

        self.title = data.get("title", self.path.stem)
        self.bpm = data["bpm"]
        self.offset = data.get("offset", 0.0)
        audio_name = data.get("audio")
        self.audio_path = self.path.parent / audio_name if audio_name else None
        self.has_audio = self._load_audio()
        self.started_at = None   # time.perf_counter() at start(), None when not playing
        self.last_time = 0.0

    def _load_audio(self):
        if self.audio_path is None or not self.audio_path.exists():
            print(f"Warning: music file {self.audio_path} not found, the song uses a timer instead")
            return False
        if pygame is None:
            print("Warning: pygame is not installed, the song uses a timer instead")
            return False
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(str(self.audio_path))
        except pygame.error as error:
            print(f"Warning: cannot play {self.audio_path} ({error}), the song uses a timer instead")
            return False
        return True

    # ------------------------------------------------------------------ music
    def start(self):
        self.started_at = time.perf_counter()
        self.last_time = 0.0
        if self.has_audio:
            pygame.mixer.music.play()

    def stop(self):
        if self.has_audio:
            pygame.mixer.music.stop()
        self.started_at = None
        self.last_time = 0.0

    def time(self):
        """Song time in seconds. 0 before start(). With music it comes from the audio clock,
        so the moves always stay in sync with the music."""
        if self.started_at is None:
            return 0.0
        if self.has_audio:
            position = pygame.mixer.music.get_pos()   # milliseconds since play(), -1 when stopped
            if position >= 0:
                self.last_time = position / 1000
            return self.last_time
        return time.perf_counter() - self.started_at

    def is_finished(self, t):
        if self.started_at is None:
            return False
        if self.has_audio:
            return not pygame.mixer.music.get_busy()
        return t > self.moves[-1].end + 2.0

    # ------------------------------------------------------------------ moves
    def current_move(self, t):
        """The move that is active now, or None between moves."""
        for move in self.moves:
            if move.start <= t < move.end:
                return move
            if move.start > t:
                break
        return None

    def _next_index(self, t):
        for i, move in enumerate(self.moves):
            if move.start > t:
                return i
        return None

    def next_move(self, t):
        """The first move that has not started yet, or None at the end of the song."""
        i = self._next_index(t)
        return None if i is None else self.moves[i]

    def grading_move(self, t):
        """The move we grade now. Players need some time to react, so a move is graded until
        GRADE_GRACE seconds after its end (but never after the next move starts)."""
        for i, move in enumerate(self.moves):
            end = move.end + config.GRADE_GRACE
            if i + 1 < len(self.moves):
                end = min(end, self.moves[i + 1].start)
            if move.start <= t < end:
                return move
            if move.start > t:
                break
        return None

    def progress(self, move, t):
        """0 at the start of the move, 1 at the end."""
        return min(max((t - move.start) / (move.end - move.start), 0.0), 1.0)

    def next_progress(self, t):
        """0 -> 1 from the start of the previous move (or the song start) to the start of the next move."""
        i = self._next_index(t)
        if i is None:
            return 0.0
        begin = self.moves[i - 1].start if i > 0 else 0.0
        return min(max((t - begin) / (self.moves[i].start - begin), 0.0), 1.0)

    def beat_phase(self, t):
        """0 on the beat, going up to 1 just before the next beat."""
        return ((t - self.offset) * self.bpm / 60) % 1.0
