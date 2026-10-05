"""Results: scores and winner. Raise both hands (or wait) to play again with new players."""
import config
from game_logic.states.common import HoldGesture, State, anyone_hands_up
from scene.results_screen import draw_results


class ResultsState(State):
    def enter(self, game):
        game.song.stop()
        self.start = game.now
        self.gesture = HoldGesture(config.HOLD_TIME)

    def update(self, game):
        waited = game.now - self.start
        if waited >= config.RESULTS_MAX_TIME:
            return "registration"
        if waited < config.RESULTS_MIN_TIME:
            return None                                  # nobody skips the results by accident
        if self.gesture.update(anyone_hands_up(game), game.now):
            return "registration"
        return None

    def draw(self, game, image):
        show_hint = game.now - self.start >= config.RESULTS_MIN_TIME
        player_1, player_2 = game.players[1], game.players[2]
        if player_1["score"] == player_2["score"]:
            winner = None                                # a draw
        else:
            winner = player_1 if player_1["score"] > player_2["score"] else player_2
        return draw_results(image, game.players, winner, self.gesture.progress, show_hint)
