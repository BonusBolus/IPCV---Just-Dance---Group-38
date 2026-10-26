"""Playing: the music plays, the players are graded once per move, and lost players get a warning.

Animated moves (a list of poses, see song.animated_move): every step is graded like a normal move. Each
correct step in a row (score >= CHAIN_MIN_SCORE, and from the second step on moving the right way, see
motion.py) makes the chain longer and gives CHAIN_BONUS extra points per step in the chain."""
import math

import numpy as np

import config
from game_logic.bonus_star import BonusStar
from game_logic.motion import PlayerMotion, flow
from game_logic.states.common import PLAYER_IDS, State, draw_move_cards
from identity_tracking.identity import KEYPOINT_NAMES
from scene.functions import FLOW, GREEN, draw_banner, draw_big_text, draw_duet_bonus, draw_text, player_color
from scene.star import draw_star, draw_star_burst

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
        self.frame_scores = {i: [] for i in PLAYER_IDS}  # score of every graded frame of this move, per player
        self.live = {i: None for i in PLAYER_IDS}          # score in this frame (debug only)
        self.interacted = False                          # both players met this move's interaction
        self.duet_bonus_since = None
        self.ratings = {}                                # player id -> ("Perfect", time)
        self.last_seen = {i: game.now for i in PLAYER_IDS}
        self.roles = {i: role for role, i in enumerate(PLAYER_IDS)}   # duet part per player, see _update_roles
        self.star = BonusStar()                          # the bonus star, see bonus_star.py
        self.image_size = config.GAME_SIZE               # (width, height), updated in draw()
        self.motion = PlayerMotion()                     # motion per player, for the direction check
        self.chain = {i: 0 for i in PLAYER_IDS}          # correct steps in a row in the current animated move
        self.flows = {i: None for i in PLAYER_IDS}       # direction match of the last step (debug)
        self.chains_shown = {}                           # player id -> (chain length, time), for "Chain x3"

    def update(self, game):
        t = game.song.time()

        for i in PLAYER_IDS:
            player = game.players[i]
            if player["visible"]:
                self.last_seen[i] = game.now
            # motion per player (by identity, with the names of the poses), for the direction check
            seen = player["visible"] and player["keypoints_raw"] is not None
            self.motion.update(i, player["keypoints_raw"][MIRROR_SWAP] if seen else None, t)

        # the bonus star: the first player who touches it gets STAR_BONUS points
        grabbed_by = self.star.update(t, game.players.values(), self.image_size, game.song.length)
        if grabbed_by is not None:
            game.players[grabbed_by]["score"] += config.STAR_BONUS

        move = game.song.grading_move(t)
        if move is not self.move:
            self.finish_move(game)
            self.move = move
            self.graded = move is not None and move.key in game.pose_grader.processed_poses
            self.interacted = False

        self.live = {i: None for i in PLAYER_IDS}
        if self.graded:
            duet = move.players == 2
            if duet:
                self._update_roles(game)
            kps = {}
            for i in PLAYER_IDS:
                player = game.players[i]
                if not player["visible"] or player["keypoints_raw"] is None:
                    continue                             # a lost player gets no points
                with np.errstate(divide="ignore", invalid="ignore"):
                    keypoints = player["keypoints_raw"][MIRROR_SWAP]    # see MIRROR_SWAP
                    score = float(game.pose_grader.grade_pose(
                        keypoints, move.key, player=self.roles[i] if duet else 0))
                if not math.isfinite(score):
                    continue                             # for example shoulders on top of each other
                kps[i] = keypoints
                self.live[i] = score
                self.frame_scores[i].append(score)

            if duet and len(kps) == len(PLAYER_IDS) and all(
                self.live[i] >= config.DUET_MIN_SCORE for i in PLAYER_IDS
            ):
                ordered = [kps[i] for i in sorted(kps, key=self.roles.get)]   # left to right
                if game.pose_grader.interaction_met(ordered, move.key):
                    self.interacted = True

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

    def finish_move(self, game):
        """Points once per move (see move_score), x2 for gold, plus a successful duet's bonus."""
        if self.move is None:
            return
        if not self.graded:                              # the PoseGrader does not know this pose
            self.move = None
            return
        duet_bonus = self.move.players == 2 and self.interacted
        bonus_awarded = False
        for i in PLAYER_IDS:
            score = self.move_score(i)
            points = round(score * 2 if self.move.gold else score)
            bonus = round(points * config.DUET_BONUS) if duet_bonus else 0
            points += bonus + self.chain_bonus(game, i, score, round(score * 2 if self.move.gold else score))
            bonus_awarded = bonus_awarded or bonus > 0
            game.players[i]["score"] += points
            game.players[i]["last_move_points"] = points
            self.ratings[i] = (rating(score), game.now)
        if bonus_awarded:
            self.duet_bonus_since = game.now
        self.move = None
        self.frame_scores = {i: [] for i in PLAYER_IDS}
        self.interacted = False

    def step_flow(self, player_id, move):
        """How well the player moved from the previous step's pose to this one (-1..1), or None when there
        is nothing to check. The movement counts from half a step before this step until half a step
        into it: the figure moves during the second half of the previous step, the player follows."""
        if move.prev_pose is None:
            return None
        half = (move.end - move.start) / 2
        displacement = self.motion.displacement(player_id, move.start - half, move.start + half)
        return flow(displacement, move.prev_pose, move.pose)

    def chain_bonus(self, game, player_id, score, points):
        """Animated moves: extra points for every correct step in a row. Returns the extra points."""
        move = self.move
        if getattr(move, "group", None) is None:            # a normal move (fake moves in tests have no group)
            self.chain[player_id] = 0
            return 0
        if move.step == 0:
            self.chain[player_id] = 0                    # a new animated move: a new chain
        correct = score >= config.CHAIN_MIN_SCORE
        if config.FLOW_CHECK and move.step > 0:
            self.flows[player_id] = self.step_flow(player_id, move)
            if self.flows[player_id] is not None and self.flows[player_id] < config.FLOW_MIN:
                correct = False                          # the right pose, but not the right movement
        if not correct:
            self.chain[player_id] = 0
            return 0
        self.chain[player_id] += 1
        if self.chain[player_id] >= 2:
            self.chains_shown[player_id] = (self.chain[player_id], game.now)
        return config.CHAIN_BONUS * (self.chain[player_id] - 1)

    def draw_chains(self, game, image):
        """ "Chain x3" above a player who did several steps of an animated move right in a row."""
        for player_id, (length, since) in self.chains_shown.items():
            player = game.players[player_id]
            if game.now - since >= config.CHAIN_TIME or not player["visible"] or player["box"] is None:
                continue
            x, y, w, h = player["box"]
            draw_text(image, f"Chain x{length}", (int(x + w / 2), max(30, int(y) - 20)), 0.9, FLOW, 2)
        return image

    def draw_star(self, game, image, t):
        """The floating star (fading in and out), and the burst with "+100" after a grab."""
        age = self.star.age(t)
        if age is not None and self.star.position is not None:
            alpha = min(1.0, age / 0.3, (config.STAR_TIME - age) / 0.5)
            draw_star(image, self.star.position, int(config.STAR_SIZE * image.shape[0]), t, alpha)
        if self.star.grab is not None:
            player_id, position, since = self.star.grab
            progress = (t - since) / config.STAR_BURST_TIME
            draw_star_burst(image, position, f"+{config.STAR_BONUS}", player_color(game.players[player_id]), progress)
        return image

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

        self.image_size = (image.shape[1], image.shape[0])
        image = self.draw_star(game, image, t)
        image = self.draw_chains(game, image)

        lost = [game.players[i] for i in PLAYER_IDS if self.is_lost(game, i)]
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
            for i in PLAYER_IDS:
                live_str = '-' if self.live[i] is None else f'{self.live[i]:.0f}'
                last_pts = game.players[i].get("last_move_points")
                last_str = '-' if last_pts is None else f'{last_pts} pts'
                flow_str = '-' if self.flows[i] is None else f'{self.flows[i]:+.2f}'
                lines.append(f"P{i} live: {live_str:>3} | last move: {last_str} | chain {self.chain[i]} flow {flow_str}")
            for row, line in enumerate([f"t = {t:.1f} s"] + lines):
                image = draw_text(image, line, (90, 140 + 25 * row), 0.6, GREEN)
        return image
    
    def _update_roles(self, game):
        """Role 0 = player on the left of the screen, role 1 = on the right.
        Only updated while both players are tracked; otherwise the last assignment stays."""
        ids = [i for i in PLAYER_IDS
               if game.players[i]["visible"] and game.players[i]["keypoints_raw"] is not None]
        if len(ids) < len(PLAYER_IDS):
            return

        def shoulder_x(i):
            kp = game.players[i]["keypoints_raw"]
            return (kp[KEYPOINT_NAMES.index("left_shoulder"), 0]
                    + kp[KEYPOINT_NAMES.index("right_shoulder"), 0]) / 2

        self.roles = {i: r for r, i in enumerate(sorted(ids, key=shoulder_x))}
