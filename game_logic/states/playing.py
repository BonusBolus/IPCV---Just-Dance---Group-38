"""Playing: the music plays, the players are graded once per move, and lost players get a warning."""
import math

import numpy as np

import config
from game_logic.states.common import PLAYER_IDS, State, draw_move_cards
from identity_tracking.identity import KEYPOINT_NAMES
from scene.functions import GREEN, draw_banner, draw_big_text, draw_text, player_color

RATINGS = [(85, "Perfect"), (65, "Good"), (40, "OK")]   # below 40: "Miss"


def swap_side(name):
    """left_wrist -> right_wrist and the other way around. Other names stay the same."""
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


# YOLO names left and right by how the body LOOKS: for a person facing the camera, the arm on
# the left of the image is called right_*. Our camera image is mirrored, so YOLO calls the
# player's left arm right_*. The PoseGrader poses have left_* on the left of the screen, so we
# swap left and right before grading. Without this, even a perfect pose scores below 20.
MIRROR_SWAP = [KEYPOINT_NAMES.index(swap_side(name)) for name in KEYPOINT_NAMES]


def rating(score):
    """Score 0-100 -> "Perfect", "Good", "OK" or "Miss"."""
    for minimum, name in RATINGS:
        if score >= minimum:
            return name
    return "Miss"


class PlayingState(State):
    def enter(self, game):
        game.song.start()
        self.move = None                                 # the move we are grading now
        self.graded = False                              # False if the PoseGrader does not know the pose
        self.best = {i: None for i in PLAYER_IDS}        # best score of each player for this move
        self.live = {i: None for i in PLAYER_IDS}        # score in this frame (debug only)
        self.ratings = {}                                # player id -> ("Perfect", time)
        self.last_seen = {i: game.now for i in PLAYER_IDS}

    def update(self, game):
        t = game.song.time()

        for i in PLAYER_IDS:
            if game.players[i]["visible"]:
                self.last_seen[i] = game.now

        move = game.song.grading_move(t)
        if move is not self.move:
            self.finish_move(game)
            self.move = move
            self.graded = move is not None and move.key in game.pose_grader.processed_poses

        self.live = {i: None for i in PLAYER_IDS}
        if self.graded:
            for i in PLAYER_IDS:
                player = game.players[i]
                if not player["visible"] or player["keypoints_raw"] is None:
                    continue                             # a lost player gets no points
                with np.errstate(divide="ignore", invalid="ignore"):
                    keypoints = player["keypoints_raw"][MIRROR_SWAP]    # see MIRROR_SWAP
                    score = float(game.pose_grader.grade_pose(keypoints, move.key))
                if not math.isfinite(score):
                    continue                             # for example shoulders on top of each other
                self.live[i] = score
                if self.best[i] is None or score > self.best[i]:
                    self.best[i] = score

        if game.song.is_finished(t):
            self.finish_move(game)
            return "results"
        return None

    def finish_move(self, game):
        """Points once per move: the best score during the move, x2 for a gold move."""
        if self.move is None:
            return
        if not self.graded:                              # the PoseGrader does not know this pose
            self.move = None
            return
        for i in PLAYER_IDS:
            score = self.best[i] or 0.0
            points = round(score * 2 if self.move.gold else score)
            game.players[i]["score"] += points
            game.players[i]["last_move_points"] = points
            self.ratings[i] = (rating(score), game.now)
        self.move = None
        self.best = {i: None for i in PLAYER_IDS}

    def is_lost(self, game, player_id):
        player = game.players[player_id]
        return player["registered"] and game.now - self.last_seen[player_id] > config.LOST_TIME

    def draw(self, game, image):
        t = game.song.time()
        image = game.scene.render(image, list(game.players.values()))    # title + score bar
        image = draw_move_cards(game, image, t)

        ratings = {i: text for i, (text, since) in self.ratings.items()
                   if game.now - since < config.RATING_TIME}
        image = game.scene.draw_ratings(image, ratings)

        lost = [game.players[i] for i in PLAYER_IDS if self.is_lost(game, i)]
        for row, player in enumerate(lost):
            image = draw_banner(image, f"Player {player['id']} lost - step back in!", player_color(player), row)

        if t < 1.0:
            image = draw_big_text(image, "Dance!")
        if game.debug:
            lines = []
            for i in PLAYER_IDS:
                live_str = '-' if self.live[i] is None else f'{self.live[i]:.0f}'
                last_pts = game.players[i].get("last_move_points")
                last_str = '-' if last_pts is None else f'{last_pts} pts'
                lines.append(f"P{i} live: {live_str:>3} | last move: {last_str}")
            for row, line in enumerate([f"t = {t:.1f} s"] + lines):
                image = draw_text(image, line, (90, 140 + 25 * row), 0.6, GREEN)
        return image
