"""Registration: the game starts here. The tracker finds 2 players, and both raise their hands when ready."""
import config
from game_logic.states.common import PLAYER_IDS, HoldGesture, State, hands_up
from scene.registration_screen import draw_registration


class RegistrationState(State):
    def enter(self, game):
        game.tracker.reset()               # forget old players and old scores
        game.players = game.tracker.entries
        self.gestures = {i: HoldGesture(config.HOLD_TIME) for i in PLAYER_IDS}
        self.ready = {i: False for i in PLAYER_IDS}
        self.last_seen = {i: game.now for i in PLAYER_IDS}

    def update(self, game):
        for i in PLAYER_IDS:
            player = game.players[i]
            if player["visible"]:
                self.last_seen[i] = game.now
                if not self.ready[i]:
                    self.ready[i] = self.gestures[i].update(hands_up(player), game.now)
            elif game.now - self.last_seen[i] > config.LOST_TIME:
                # gone for a while (not just one missed frame): ready again from the start
                self.ready[i] = False
                self.gestures[i].update(False, game.now)
        if all(self.ready.values()):
            return "countdown"
        return None

    def draw(self, game, image):
        progress = {i: self.gestures[i].progress for i in PLAYER_IDS}
        return draw_registration(image, game.players, self.ready, progress, game.tracker.colors_are_similar())
