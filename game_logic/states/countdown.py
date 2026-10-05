"""Countdown: 3, 2, 1. The NEXT card already shows the first move, so players can get ready."""
import math

import config
from game_logic.states.common import State, draw_move_cards
from scene.functions import draw_big_text


class CountdownState(State):
    def enter(self, game):
        self.start = game.now

    def update(self, game):
        if game.now - self.start >= config.COUNTDOWN_TIME:
            return "playing"
        return None

    def draw(self, game, image):
        seconds_left = config.COUNTDOWN_TIME - (game.now - self.start)
        image = game.scene.render(image, list(game.players.values()))    # title + score bar
        image = draw_move_cards(game, image, 0.0)    # the NEXT card already shows the first move
        return draw_big_text(image, str(max(1, math.ceil(seconds_left))))
