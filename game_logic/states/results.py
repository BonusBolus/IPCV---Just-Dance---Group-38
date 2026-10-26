"""Results: scores and winner. Raise both hands (or wait) to go back to the song selection."""
import config
from game_logic.states.common import HoldGesture, State, anyone_hands_up
from scene.results_screen import draw_results
from scene.functions import draw_duet_bonus


class ResultsState(State):
    def enter(self, game):
        game.song.stop()
        self.start = game.now
        self.gesture = HoldGesture(config.HOLD_TIME)
        self.duet_bonus_since = game.states["playing"].duet_bonus_since

    def update(self, game):
        if getattr(game, "recorder", None) is not None:
            return None                                  # recording: the results stay until the video ends
        waited = game.now - self.start
        if waited >= config.RESULTS_MAX_TIME:
            return "song_select"
        if waited < config.RESULTS_MIN_TIME:
            return None                                  # nobody skips the results by accident
        if self.gesture.update(anyone_hands_up(game), game.now):
            return "song_select"
        return None

    def draw(self, game, image):
        show_hint = game.now - self.start >= config.RESULTS_MIN_TIME
        player_1, player_2 = game.players[1], game.players[2]
        if player_1["score"] == player_2["score"]:
            winner = None                                # a draw
        else:
            winner = player_1 if player_1["score"] > player_2["score"] else player_2
        image = draw_results(image, game.players, winner, self.gesture.progress, show_hint)
        if self.duet_bonus_since is not None:
            elapsed = game.now - self.duet_bonus_since
            if 0 <= elapsed < config.DUET_BONUS_TIME:
                image = draw_duet_bonus(image, elapsed / config.DUET_BONUS_TIME, center_y=45)
        return image
