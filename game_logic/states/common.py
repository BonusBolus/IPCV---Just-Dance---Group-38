"""Parts that more than one state uses.

Every state has these methods, called by Game (game_logic/game.py):
    enter(game)          once, when the game switches to this state
    update(game)         every frame; returns the name of the next state, or None to stay
    draw(game, image)    every frame; returns the image to show
    exit(game)           once, when the game leaves this state (also with the debug key n)

Flow:  song_select -> registration -> countdown -> playing -> results -> song_select
A state decides WHAT happens; the files in scene/ decide HOW it looks.
"""

import config
from scene import move_cards, timeline
from scene.functions import player_color

PLAYER_IDS = tuple(range(1, config.MAX_PLAYERS + 1))   # every player slot (see players_in_game())


def players_in_game(game):
    """The ids of the players who really play: registered at Registration (slots nobody took are left out)."""
    return [i for i, player in game.players.items() if player["registered"]]


class State:
    def enter(self, game):
        pass

    def update(self, game):
        return None

    def draw(self, game, image):
        return image

    def exit(self, game):
        pass


def hands_up(player):
    """True if both wrists are above the nose. In the image, "above" means a smaller y."""
    points = player["keypoints"]
    nose, left, right = points["nose"], points["left_wrist"], points["right_wrist"]
    if nose is None or left is None or right is None:
        return False
    return left[1] < nose[1] and right[1] < nose[1]


# The two keypoints of each joint angle in the PoseGrader (see pose_grading._get_reference_angles)
ANGLE_POINTS = {"shoulder": ("shoulder", "elbow"), "elbow": ("elbow", "wrist"),
                "hip": ("hip", "knee"), "knee": ("knee", "ankle")}
LOWER_BODY = ("hip", "knee", "ankle")                  # from top to bottom
BODY_PART_WORDS = {"hip": "hips", "knee": "knees", "ankle": "feet"}


def lowest_needed_part(pose_grader):
    """The lowest body part the grading looks at in any pose, for example "knee" (the hip angle
    goes from the hip to the knee). Players must stand far enough back to show it. This follows
    the grader: if it starts using the knee angles, it becomes "ankle"."""
    parts = set()
    for pose in pose_grader.processed_poses.values():
        for angles in pose["angles"]:
            for joint in angles:                      # for example "left_hip"
                parts.update(ANGLE_POINTS[joint.split("_", 1)[1]])
    lower = [part for part in LOWER_BODY if part in parts]
    return lower[-1] if lower else None


def body_visible_down_to(player, part):
    """True if the camera sees the nose and the `part` (for example "knee") of at least one side.

    The tracker only keeps keypoints the model is sure about (confidence 0.5), so a guessed
    knee (legs out of the image) does not count."""
    points = player["keypoints"]
    if points["nose"] is None:
        return False
    return part is None or points[f"left_{part}"] is not None or points[f"right_{part}"] is not None


def raised_hands(player):
    """Which hands are up (above the nose): "both", "left", "right" or None.

    "left" and "right" are the sides of the SCREEN. The camera image is mirrored, so a player's
    own left hand is on the left of the screen. We look at which side of the nose the wrist is,
    not at YOLO's names: in the mirrored image YOLO calls the left hand right_wrist."""
    points = player["keypoints"]
    nose = points["nose"]
    if nose is None:
        return None
    up = [wrist for wrist in (points["left_wrist"], points["right_wrist"])
          if wrist is not None and wrist[1] < nose[1]]
    if len(up) == 2:
        return "both"
    if len(up) == 1:
        return "left" if up[0][0] < nose[0] else "right"
    return None


class HoldGesture:
    """A gesture that must be held for `hold_time` seconds, so nobody starts something by accident."""

    def __init__(self, hold_time):
        self.hold_time = hold_time
        self.started = None
        self.progress = 0.0   # 0..1, for the progress bar on screen

    def update(self, active, now):
        """Returns True when the gesture was held long enough."""
        if not active:
            self.started = None
            self.progress = 0.0
            return False
        if self.started is None:
            self.started = now
        self.progress = min((now - self.started) / self.hold_time, 1.0)
        return self.progress >= 1.0


def anyone_hands_up(game):
    return any(player["visible"] and hands_up(player) for player in game.players.values())

def draw_move_cards(game, image, t):
    """The NOW and NEXT cards (or the NOW card and the timeline), used in the countdown and while playing.
    t is the song time; during the countdown it is negative, so the timeline already moves."""
    song = game.song
    now_move = song.current_move(t)
    colors = tuple(player_color(player) for player in list(game.players.values())[:2])   # duet figures: 2 colours
    if config.SHOW_TIMELINE:
        image = timeline.draw_timeline(image, song.moves, t, song.beat_phase(t), colors)
    return move_cards.draw_move_cards(
        image,
        now_move=now_move,
        now_progress=song.progress(now_move, t) if now_move else 0.0,
        next_move=song.next_move(t),
        next_progress=song.next_progress(t),
        beat_phase=song.beat_phase(t),
        flash=now_move is not None and t - now_move.start < 0.15,
        player_colors=colors,
        show_next=not config.SHOW_TIMELINE,
    )
