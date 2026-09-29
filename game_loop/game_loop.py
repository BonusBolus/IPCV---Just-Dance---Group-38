import json
import math


# Pose keypoints are relative to the nose. Lengths used:
# upper/lower arms 0.8, upper/lower legs 0.75
with open("choreography/poses.json", "r", encoding="utf-8") as poses_file:
    poses = json.load(poses_file)


def load_song(song_path):
    with open(song_path, "r", encoding="utf-8") as song_file:
        return json.load(song_file)


def get_current_pose(song, song_time):
    current_beat = math.floor(song_time / 60 * song["bpm"])
    selected_move = None
    for move in song["move_list"]:
        if move["beat_index"] <= current_beat:
            selected_move = move["move"]
        else:
            break

    return poses.get(selected_move)