"""The bonus star: during the song a star floats across the screen at hand height now and then.
The first player who touches it with a hand gets STAR_BONUS points.

- About every STAR_INTERVAL seconds (a bit random) a star appears on one side and floats to the other side
  in STAR_TIME seconds, gently up and down. The next star comes from the other side.
- "Hand height": between just above the heads and the shoulders of the players, so a player can reach it
  with a raised arm. It follows the players when they move.
- No star in the last seconds of the song, and none when nobody is seen.

All times are song times (seconds), like the moves.
"""
import math
import random

import config

WRISTS = ("left_wrist", "right_wrist")


class BonusStar:
    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.next_start = self._gap()   # song time of the next star
        self.start = None               # song time the current star appeared; None: no star now
        self.from_left = self.random.random() < 0.5
        self.band = None                # (top y, bottom y) of hand height, in image pixels
        self.position = None            # (x, y) of the current star, None when there is none
        self.grab = None                # the last grab: (player id, (x, y), song time), for the "+100"

    def _gap(self):
        return config.STAR_INTERVAL + self.random.uniform(-config.STAR_JITTER, config.STAR_JITTER)

    def age(self, t):
        """Seconds since the current star appeared (None when there is no star)."""
        return None if self.start is None else t - self.start

    def update(self, t, players, image_size, song_length):
        """Call every frame. Returns the id of the player who grabbed the star now, else None.

        players:    the player dicts (box and keypoints in image pixels)
        image_size: (width, height) of the game image"""
        width, height = image_size
        visible = [player for player in players if player["visible"] and player["box"] is not None]
        if self.start is None:
            if t >= self.next_start and visible:
                if t + config.STAR_TIME > song_length - config.STAR_END_FREE:
                    self.next_start = math.inf          # too close to the end of the song: no more stars
                    return None
                self.start = t
            else:
                return None
        age = t - self.start
        if age >= config.STAR_TIME:                     # nobody got it
            self._next(t)
            return None
        self._follow_players(visible, height)
        self.position = self.position_at(age, width)
        player_id = self.touched_by(visible, height)
        if player_id is not None:
            self.grab = (player_id, self.position, t)
            self._next(t)
        return player_id

    def _next(self, t):
        self.start = None
        self.position = None
        self.next_start = t + self._gap()
        self.from_left = not self.from_left

    def _follow_players(self, visible, height):
        """Hand height from the players: from just above the highest head to the shoulders (smoothed)."""
        if not visible:
            target = (0.2 * height, 0.4 * height) if self.band is None else self.band
        else:
            top = min(player["box"][1] for player in visible) - 0.04 * height
            shoulders = [point[1] for player in visible for name in ("left_shoulder", "right_shoulder")
                         if (point := player["keypoints"].get(name)) is not None]
            bottom = (sum(shoulders) / len(shoulders) if shoulders
                      else sum(p["box"][1] + 0.25 * p["box"][3] for p in visible) / len(visible))
            target = (max(0.05 * height, top), max(top + 0.05 * height, bottom))
        if self.band is None:
            self.band = target
        else:
            self.band = tuple(0.9 * old + 0.1 * new for old, new in zip(self.band, target))

    def position_at(self, age, width):
        """Where the star is `age` seconds after it appeared: across the screen, gently up and down."""
        progress = age / config.STAR_TIME
        x = (0.1 + 0.8 * progress) if self.from_left else (0.9 - 0.8 * progress)
        top, bottom = self.band
        y = top + (bottom - top) * (0.5 + 0.45 * math.sin(2 * math.pi * 0.6 * age))
        return int(x * width), int(y)

    def touched_by(self, visible, height):
        """The player with a wrist closest to the star, within STAR_REACH; None when nobody touches it."""
        reach = config.STAR_REACH * height
        best, best_distance = None, reach
        for player in visible:
            for name in WRISTS:
                point = player["keypoints"].get(name)
                if point is None:
                    continue
                distance = math.hypot(point[0] - self.position[0], point[1] - self.position[1])
                if distance <= best_distance:
                    best, best_distance = player["id"], distance
        return best
