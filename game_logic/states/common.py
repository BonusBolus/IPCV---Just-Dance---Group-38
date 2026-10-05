"""Parts that more than one state uses.

Every state has three methods, called by Game (game_logic/game.py):
    enter(game)          once, when the game switches to this state
    update(game)         every frame; returns the name of the next state, or None to stay
    draw(game, image)    every frame; returns the image to show

Flow:  registration -> countdown -> playing -> results -> registration
A state decides WHAT happens; the files in scene/ decide HOW it looks.
"""

from scene import move_cards

PLAYER_IDS = (1, 2)


class State:
    def enter(self, game):
        pass

    def update(self, game):
        return None

    def draw(self, game, image):
        return image


def hands_up(player):
    """True if both wrists are above the nose. In the image, "above" means a smaller y."""
    points = player["keypoints"]
    nose, left, right = points["nose"], points["left_wrist"], points["right_wrist"]
    if nose is None or left is None or right is None:
        return False
    return left[1] < nose[1] and right[1] < nose[1]


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
    """The NOW and NEXT cards, used in the countdown and while playing."""
    song = game.song
    now_move = song.current_move(t)
    return move_cards.draw_move_cards(
        image,
        now_move=now_move,
        now_progress=song.progress(now_move, t) if now_move else 0.0,
        next_move=song.next_move(t),
        next_progress=song.next_progress(t),
        beat_phase=song.beat_phase(t),
        flash=now_move is not None and t - now_move.start < 0.15,
    )
