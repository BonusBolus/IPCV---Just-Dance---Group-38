"""Results: a ranking of the players and the winner. Raise both hands (or wait) to go back to the song selection."""
import config
from game_logic.states.common import HoldGesture, State, anyone_hands_up, players_in_game
from scene.results_screen import draw_results
from scene.functions import draw_duet_bonus


def rank_players(game):
    """[(place, player), ...] from the highest score down. Players with the same score share a place.
    Only the players in the game (when nobody registered, for example after the debug key: Player 1 and 2)."""
    ids = players_in_game(game) or list(game.players)[:2]
    players = sorted((game.players[i] for i in ids), key=lambda player: (-player["score"], player["id"]))
    ranking = []
    for index, player in enumerate(players):
        same_as_above = index > 0 and player["score"] == players[index - 1]["score"]
        ranking.append((ranking[-1][0] if same_as_above else index + 1, player))
    return ranking


class ResultsState(State):
    def enter(self, game):
        game.song.stop()
        self.start = game.now
        self.gesture = HoldGesture(config.HOLD_TIME)
        self.duet_bonus_since = game.states["playing"].duet_bonus_since

    def update(self, game):
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
        ranking = rank_players(game)
        image = draw_results(image, ranking, self.gesture.progress, show_hint)
        if self.duet_bonus_since is not None:
            elapsed = game.now - self.duet_bonus_since
            if 0 <= elapsed < config.DUET_BONUS_TIME:
                image = draw_duet_bonus(image, elapsed / config.DUET_BONUS_TIME, center_y=45)
        return image
