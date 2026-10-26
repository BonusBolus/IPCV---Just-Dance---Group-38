"""Registration: the game starts here. The tracker finds 2 players, and both raise their hands when ready."""
import config
from game_logic.states.common import BODY_PART_WORDS, body_visible_down_to, lowest_needed_part, hands_up, HoldGesture, PLAYER_IDS, State
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
        # "Step back" when the camera does not see the lowest body part that the grading needs
        part = lowest_needed_part(game.pose_grader)
        full_body = {i: body_visible_down_to(game.players[i], part) for i in PLAYER_IDS}
        hint = f"Step back - show your {BODY_PART_WORDS[part]}" if part else None
        image = draw_registration(image, game.players, game.heads, self.ready, progress,
                                  game.tracker.colors_are_similar(), full_body, hint)
        return game.scene.draw_title(image, list(game.players.values()), "Raise both hands when you are ready")
