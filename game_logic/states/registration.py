"""Registration: the players step in (up to MAX_PLAYERS) and raise both hands when they are ready.

The game starts when everyone who is registered is ready, and there are at least MIN_PLAYERS.
A player who is gone for DROP_TIME (for example someone who walked past) is not waited for.
"""
import config
from game_logic.states.common import BODY_PART_WORDS, body_visible_down_to, lowest_needed_part, hands_up, HoldGesture, State
from scene.registration_screen import draw_registration


class RegistrationState(State):
    def enter(self, game):
        game.tracker.reset()               # forget old players and old scores
        game.players = game.tracker.entries
        self.gestures = {i: HoldGesture(config.HOLD_TIME) for i in game.players}
        self.ready = {i: False for i in game.players}
        self.last_seen = {i: game.now for i in game.players}

    def waiting_for(self, game):
        """The players the game waits for: registered, and not gone for DROP_TIME."""
        return [i for i, player in game.players.items()
                if player["registered"] and game.now - self.last_seen[i] <= config.DROP_TIME]

    def update(self, game):
        for i, player in game.players.items():
            if player["visible"]:
                self.last_seen[i] = game.now
                if not self.ready[i]:
                    self.ready[i] = self.gestures[i].update(hands_up(player), game.now)
            elif game.now - self.last_seen[i] > config.LOST_TIME:
                # gone for a while (not just one missed frame): ready again from the start
                self.ready[i] = False
                self.gestures[i].update(False, game.now)
        waiting = self.waiting_for(game)
        if len(waiting) >= config.MIN_PLAYERS and all(self.ready[i] for i in waiting):
            return "countdown"
        return None

    def draw(self, game, image):
        progress = {i: self.gestures[i].progress for i in game.players}
        # "Step back" when the camera does not see the lowest body part that the grading needs
        part = lowest_needed_part(game.pose_grader)
        full_body = {i: body_visible_down_to(game.players[i], part) for i in game.players}
        hint = f"Step back - show your {BODY_PART_WORDS[part]}" if part else None
        image = draw_registration(image, game.players, game.heads, self.ready, progress,
                                  game.tracker.colors_are_similar(), full_body, hint)
        return game.scene.draw_title(image, list(game.players.values()), "Raise both hands when you are ready")
