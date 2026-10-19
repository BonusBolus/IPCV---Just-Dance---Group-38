"""Playing: the music plays, the players are graded once per move, and lost players get a warning."""
import math

import numpy as np

import config
from game_logic.states.common import State, draw_move_cards
from identity_tracking.identity import KEYPOINT_NAMES
from scene.functions import GREEN, draw_banner, draw_big_text, draw_duet_bonus, draw_text, player_color

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
        self.frame_scores = {i: [] for i in game.players}  # score of every graded frame of this move, per player
        self.live = {i: None for i in game.players}          # score in this frame (debug only)
        self.interacted = set()                          # players whose duet pair met this move's interaction
        self.duet_bonus_since = None
        self.ratings = {}                                # player id -> ("Perfect", time)
        self.last_seen = {i: game.now for i in game.players}
        # duet part per player (0 = left, 1 = right) and the duet pairs, see _update_roles
        ids = self.players_in_move(game) or list(game.players)   # by number until we see where they stand
        self.roles = {i: index % 2 for index, i in enumerate(ids)}
        self.pairs = [tuple(ids[k:k + 2]) for k in range(0, len(ids), 2)]

    def update(self, game):
        t = game.song.time()

        for i in game.players:
            if game.players[i]["visible"]:
                self.last_seen[i] = game.now

        move = game.song.grading_move(t)
        if move is not self.move:
            self.finish_move(game)
            self.move = move
            self.graded = move is not None and move.key in game.pose_grader.processed_poses
            self.interacted = set()

        self.live = {i: None for i in game.players}
        if self.graded:
            duet = move.players == 2
            if duet:
                self._update_roles(game)
            kps = {}
            for i in game.players:
                player = game.players[i]
                if not player["visible"] or player["keypoints_raw"] is None:
                    continue                             # a lost player gets no points
                with np.errstate(divide="ignore", invalid="ignore"):
                    keypoints = player["keypoints_raw"][MIRROR_SWAP]    # see MIRROR_SWAP
                    score = float(game.pose_grader.grade_pose(
                        keypoints, move.key, player=self.roles.get(i, 0) if duet else 0))
                if not math.isfinite(score):
                    continue                             # for example shoulders on top of each other
                kps[i] = keypoints
                self.live[i] = score
                self.frame_scores[i].append(score)

            # duet bonus: per pair of neighbours, both good at the same time and the interaction met
            for pair in self.pairs if duet else []:
                if len(pair) == 2 and all(i in kps and self.live[i] >= config.DUET_MIN_SCORE for i in pair):
                    ordered = [kps[i] for i in sorted(pair, key=self.roles.get)]   # left to right
                    if game.pose_grader.interaction_met(ordered, move.key):
                        self.interacted.update(pair)

        if game.song.is_finished(t):
            self.finish_move(game)
            return "results"
        return None

    def move_score(self, player_id):
        """Score of a player for the move: the SCORE_PERCENTILE (70th) percentile of the frame scores.

        With the best frame, one lucky frame was enough. Now the player must hold the pose for at
        least about 30% of the move. Frames where the player was not visible are not counted;
        a player who was not graded at all gets 0."""
        scores = self.frame_scores[player_id]
        return float(np.percentile(scores, config.SCORE_PERCENTILE)) if scores else 0.0

    def players_in_move(self, game):
        """The players who get points and a rating: registered, or graded in this move (slots that
        nobody took are left out)."""
        return [i for i in game.players if game.players[i]["registered"] or self.frame_scores[i]]

    def finish_move(self, game):
        """Points once per move (see move_score), x2 for gold, plus a successful duet's bonus."""
        if self.move is None:
            return
        if not self.graded:                              # the PoseGrader does not know this pose
            self.move = None
            return
        bonus_awarded = False
        for i in self.players_in_move(game):
            score = self.move_score(i)
            points = round(score * 2 if self.move.gold else score)
            bonus = round(points * config.DUET_BONUS) if self.move.players == 2 and i in self.interacted else 0
            points += bonus
            bonus_awarded = bonus_awarded or bonus > 0
            game.players[i]["score"] += points
            game.players[i]["last_move_points"] = points
            self.ratings[i] = (rating(score), game.now)
        if bonus_awarded:
            self.duet_bonus_since = game.now
        self.move = None
        self.frame_scores = {i: [] for i in game.players}
        self.interacted = set()

    def is_lost(self, game, player_id):
        player = game.players[player_id]
        return player["registered"] and game.now - self.last_seen[player_id] > config.LOST_TIME

    def draw(self, game, image):
        t = game.song.time()
        image = game.scene.render(image, list(game.players.values()))    # title + score bar
        image = draw_move_cards(game, image, t)

        ratings = {i: text for i, (text, since) in self.ratings.items()
                   if game.now - since < config.RATING_TIME}
        ages = {i: game.now - since for i, (text, since) in self.ratings.items()}
        image = game.scene.draw_ratings(image, ratings, ages, config.RATING_TIME)   # next to the score bar, pops

        lost = [game.players[i] for i in game.players if self.is_lost(game, i)]
        for row, player in enumerate(lost):
            image = draw_banner(image, f"Player {player['id']} lost - step back in!", player_color(player), row)

        if self.duet_bonus_since is not None:
            elapsed = game.now - self.duet_bonus_since
            if 0 <= elapsed < config.DUET_BONUS_TIME:
                image = draw_duet_bonus(image, elapsed / config.DUET_BONUS_TIME)

        if t < 1.0:
            image = draw_big_text(image, "Dance!")
        if game.debug:
            lines = []
            for i in self.players_in_move(game):
                live_str = '-' if self.live[i] is None else f'{self.live[i]:.0f}'
                last_pts = game.players[i].get("last_move_points")
                last_str = '-' if last_pts is None else f'{last_pts} pts'
                lines.append(f"P{i} live: {live_str:>3} | last move: {last_str}")
            for row, line in enumerate([f"t = {t:.1f} s"] + lines):
                image = draw_text(image, line, (90, 140 + 25 * row), 0.6, GREEN)
        return image
    
    def _update_roles(self, game):
        """Duets: pairs of neighbours, from left to right on the screen. The 1st and 2nd player are a
        pair, the 3rd and 4th, ...; in a pair, role 0 is the left part, role 1 the right part.
        A player without a partner (the 5th) does the left part alone, without duet bonus.
        Only updated while every player in the game is tracked; otherwise the last pairs stay."""
        in_game = self.players_in_move(game) or list(game.players)
        ids = [i for i in in_game
               if game.players[i]["visible"] and game.players[i]["keypoints_raw"] is not None]
        if len(ids) < len(in_game):
            return

        def shoulder_x(i):
            kp = game.players[i]["keypoints_raw"]
            return (kp[KEYPOINT_NAMES.index("left_shoulder"), 0]
                    + kp[KEYPOINT_NAMES.index("right_shoulder"), 0]) / 2

        left_to_right = sorted(ids, key=shoulder_x)
        self.roles = {i: index % 2 for index, i in enumerate(left_to_right)}
        self.pairs = [tuple(left_to_right[k:k + 2]) for k in range(0, len(left_to_right), 2)]
