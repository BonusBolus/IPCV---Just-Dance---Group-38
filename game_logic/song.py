"""The song: which pose the players must do at which time, and the music.

Song file format: see choreography/song_1.json and docs/game_structure.md, section 5.
The moves are on beats, the poses come from choreography/poses.json.
"bpm" is the tempo of the music (the beat pulse); the optional "move_bpm" is the tempo of the moves:
higher means shorter moves that come more often. When the music is longer than the moves, the moves
start again from the first one (see repeat_moves).

All methods that need the song time get it as argument `t` (seconds). Call `song.time()`
once per frame and pass that value on, so everything in one frame uses the same time.
"""
import json
import os
import statistics
import time
from dataclasses import dataclass, field
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
    pose: dict     # from poses.json: {"nose": (x, y), ...}, nose at (0, 0), None = not used, when duet move this holds only keypoints player 1
    start: float   # seconds: the move becomes active, the player must be in the pose
    end: float     # seconds: the move is over
    gold: bool     # gold move: double points
    players: int = 1 #1 is single move, 2 is duet
    pose2: dict | None = None #if duet store keypoints for p2
    interaction: list = field(default_factory=list)  # list of dicts: {"pose": "pose_name", "keypoints": {...}, "players": 1, "interaction": [...]}


def moves_from_beats(data, poses, file_name):
    """Beat format: {"bpm": 120, "offset": 0.0, "moves": [{"beat": 4, "pose": "cactus", "beats": 2}, ...]}
    The beats of the moves use "move_bpm" when the song has it (faster moves), else "bpm"."""
    beat_time = 60 / data.get("move_bpm", data["bpm"])
    offset = data.get("offset", 0.0)
    moves = []
    for i, item in enumerate(data["moves"]):
        if item["pose"] not in poses:
            raise ValueError(f'{file_name} move {i}: pose "{item["pose"]}" not found in poses.json')
        pose = poses[item["pose"]]
        
        # if two players move get both keypoints
        players = pose.get("players", 1)
        if players == 2 and ("keypoints_2" not in pose or not pose.get("interaction")):
            raise ValueError(f'{file_name} move {i}: duet pose "{item["pose"]}" needs keypoints_2 and interaction')
        
        start = offset + item["beat"] * beat_time
        end = start + item.get("beats", config.DEFAULT_MOVE_BEATS) * beat_time
        moves.append(Move(pose["name"], item["pose"], pose["keypoints"], start, end,
                          item.get("gold", False),
                          players=players,
                          pose2=pose.get("keypoints_2"),
                          interaction=pose.get("interaction", [])))
    return moves


def check_moves(moves, file_name):
    """Stop at start-up with a clear message, instead of crashing in the middle of a song."""
    if not moves:
        raise ValueError(f"{file_name}: the song has no moves")
    moves.sort(key=lambda move: move.start)
    for i in range(1, len(moves)):
        if moves[i].start < moves[i - 1].end:
            raise ValueError(f"{file_name} move {i}: starts before move {i - 1} has ended")


def repeat_moves(moves, music_length):
    """When the moves run out before the music ends, start again with the first move, until the music
    ends. The next round starts one normal step (the usual time between two moves) after the last move."""
    if music_length is None or len(moves) < 2:
        return moves
    steps = [b.start - a.start for a, b in zip(moves, moves[1:])]
    first, last = moves[0], moves[-1]
    shift = max(last.start - first.start + statistics.median(steps), last.end - first.start)
    round_moves, repeated = list(moves), list(moves)
    while True:
        round_moves = [Move(m.name, m.key, m.pose, m.start + shift, m.end + shift, m.gold, m.players,
                            m.pose2, m.interaction) for m in round_moves]
        fitting = [m for m in round_moves if m.end + config.GRADE_GRACE <= music_length]
        repeated += fitting
        if len(fitting) < len(round_moves):
            return repeated


def music_length(path):
    """Seconds of music in an .ogg file (Vorbis or Opus), read from the file itself; None when unknown."""
    path = Path(path)
    if path.suffix.lower() not in (".ogg", ".oga", ".opus") or not path.exists():
        return None
    with open(path, "rb") as f:
        head = f.read(4096)
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 65536))
        tail = f.read()
    if b"\x01vorbis" in head:
        i = head.index(b"\x01vorbis")
        rate, skip = int.from_bytes(head[i + 12:i + 16], "little"), 0
    elif b"OpusHead" in head:
        i = head.index(b"OpusHead")
        rate, skip = 48000, int.from_bytes(head[i + 10:i + 12], "little")
    else:
        return None
    last_page = tail.rfind(b"OggS")                     # the last page holds the number of the last sample
    if last_page < 0 or rate <= 0:
        return None
    samples = int.from_bytes(tail[last_page + 6:last_page + 14], "little", signed=True) - skip
    return samples / rate if samples > 0 else None


def find_songs(folder=None):
    """All song files in the choreography folder (every .json except poses.json), sorted by title."""
    folder = Path(folder or config.CHOREOGRAPHY_DIR)
    songs = [Song(path) for path in folder.glob("*.json") if path.name != Path(config.POSES_PATH).name]
    return sorted(songs, key=lambda song: song.title.lower())


class Song:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        poses = json.loads(Path(config.POSES_PATH).read_text(encoding="utf-8"))
        self.moves = moves_from_beats(data, poses, self.path.name)
        check_moves(self.moves, self.path.name)

        self.title = data.get("title", self.path.stem)
        self.bpm = data["bpm"]                       # tempo of the music: the beat pulse
        self.move_bpm = data.get("move_bpm", self.bpm)   # tempo of the moves
        self.offset = data.get("offset", 0.0)
        audio_name = data.get("audio")
        self.audio_path = self.path.parent / audio_name if audio_name else None
        self.has_audio = self._load_audio()
        if self.has_audio:
            self.moves = repeat_moves(self.moves, music_length(self.audio_path))
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

    @property
    def length(self):
        """Seconds until the last move ends."""
        return self.moves[-1].end

    @property
    def duet_moves(self):
        return sum(move.players == 2 for move in self.moves)

    @property
    def gold_moves(self):
        return sum(move.gold for move in self.moves)

    def start(self):
        self.started_at = time.perf_counter()
        self.last_time = 0.0
        if self.has_audio:
            # pygame has one music player for all songs (the song selection plays previews),
            # so load this song's music again before playing it
            pygame.mixer.music.load(str(self.audio_path))
            pygame.mixer.music.play()

    def preview(self):
        """Play a part of the music for the song selection, from PREVIEW_START (a part of the song)."""
        if self.has_audio:
            pygame.mixer.music.load(str(self.audio_path))
            pygame.mixer.music.play(start=self.length * config.PREVIEW_START, fade_ms=500)

    def music_playing(self):
        return self.has_audio and pygame.mixer.music.get_busy()

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
