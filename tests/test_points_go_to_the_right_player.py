"""The points of a move go to the player who did the pose, with the REAL PoseGrader.

Duet poses: part 1 (keypoints) is for the player on the LEFT of the screen, part 2 (keypoints_2)
for the player on the right. Player 1 can stand on either side.
"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import config
from game_logic.pose_grading import PoseGrader
from game_logic.states.playing import MIRROR_SWAP, PlayingState
from identity_tracking.identity import KEYPOINT_NAMES, new_player_entry

POSES = json.load(open(config.POSES_PATH))
SCALE = 60                                   # pixels per pose unit (shoulder width = 60 px)


def keypoints_raw(pose, center_x, nose_y=150):
    """(17, 3) keypoints like YOLO gives them on the mirrored camera image."""
    kp = np.zeros((17, 3), np.float32)
    for i, name in enumerate(KEYPOINT_NAMES):
        point = pose.get(name)
        if point is not None:
            kp[i] = (center_x + point[0] * SCALE, nose_y + point[1] * SCALE, 0.9)
    return kp[MIRROR_SWAP]                   # YOLO swaps left and right in the mirrored image


def visible_player(player_id, pose, center_x):
    entry = new_player_entry(player_id)
    entry.update(visible=True, registered=True, keypoints_raw=keypoints_raw(pose, center_x),
                 box=(int(center_x - 1.5 * SCALE), 100, int(3 * SCALE), int(5 * SCALE)))
    return entry


class PointsGoToTheRightPlayerTests(unittest.TestCase):
    grader = PoseGrader()

    def play_one_move(self, key, players, frames=10):
        move = SimpleNamespace(key=key, players=POSES[key].get("players", 1), gold=False)
        song = Mock()
        song.time.return_value = 1.0
        song.is_finished.return_value = False
        song.grading_move.return_value = move
        game = SimpleNamespace(song=song, pose_grader=self.grader, now=0.0, players=players, debug=False)
        state = PlayingState()
        state.enter(game)
        for _ in range(frames):
            game.now += 0.04
            state.update(game)
        song.grading_move.return_value = None   # the move ends: points are added
        state.update(game)
        return state, {i: players[i]["score"] for i in players}

    def test_solo_move_points_go_to_the_player_doing_it(self):
        right_pose, wrong_pose = POSES["t_pose"]["keypoints"], POSES["cactus"]["keypoints"]
        players = {1: visible_player(1, right_pose, 900), 2: visible_player(2, wrong_pose, 300)}   # player 1 on the right
        _, scores = self.play_one_move("t_pose", players)
        self.assertGreater(scores[1], 80)
        self.assertLess(scores[2], scores[1] - 20)

    def duet(self, player_1_on_the_left):
        pose = POSES["high_five"]
        left_x, right_x = 400, 400 + 3.1 * SCALE          # the raised hands touch in the middle
        if player_1_on_the_left:
            players = {1: visible_player(1, pose["keypoints"], left_x), 2: visible_player(2, pose["keypoints_2"], right_x)}
        else:
            players = {1: visible_player(1, pose["keypoints_2"], right_x), 2: visible_player(2, pose["keypoints"], left_x)}
        return self.play_one_move("high_five", players)

    def test_duet_with_player_1_on_the_left(self):
        state, scores = self.duet(player_1_on_the_left=True)
        self.assertGreater(min(scores.values()), 80)

    def test_duet_with_player_1_on_the_right(self):
        state, scores = self.duet(player_1_on_the_left=False)
        self.assertGreater(min(scores.values()), 80)      # each player is graded on the part of their side

    def test_duet_bonus_with_player_1_on_the_right(self):
        bonus_with_1_left = self.duet(player_1_on_the_left=True)[1]
        bonus_with_1_right = self.duet(player_1_on_the_left=False)[1]
        self.assertEqual(sorted(bonus_with_1_left.values()), sorted(bonus_with_1_right.values()))


if __name__ == "__main__":
    unittest.main()
