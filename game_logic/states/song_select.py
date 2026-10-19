"""Song selection: the game starts here, and comes back here after the results.

Left hand up (on the left of the screen) = previous song, right hand up = next song.
Both hands up for HOLD_TIME = choose this song, then the players register.
The selected song plays a music preview. Anyone in front of the camera can choose.
"""
import config
from game_logic.states.common import HoldGesture, State, raised_hands
from scene.song_select_screen import draw_song_select


class SongSelectState(State):
    def enter(self, game):
        self.choose = HoldGesture(config.HOLD_TIME)
        self.browse = {"left": HoldGesture(config.BROWSE_HOLD_TIME), "right": HoldGesture(config.BROWSE_HOLD_TIME)}
        self.wait_for_hands_down = False   # after one step: lower the hand before the next step
        game.song.stop()
        game.song.preview()

    def update(self, game):
        hands = {raised_hands(player) for player in game.players.values() if player["visible"]} - {None}

        if self.choose.update("both" in hands, game.now):
            return "registration"

        if not hands:
            self.wait_for_hands_down = False
        for side, step in (("left", -1), ("right", 1)):
            active = side in hands and "both" not in hands and not self.wait_for_hands_down
            if self.browse[side].update(active, game.now):
                self.select(game, (game.song_index + step) % len(game.songs))
                self.wait_for_hands_down = True
                self.browse[side].update(False, game.now)     # the bar starts at 0 again

        if not game.song.music_playing():
            game.song.preview()                               # the preview reached the end: play it again
        return None

    def select(self, game, index):
        game.song_index = index
        game.song = game.songs[index]
        game.song.preview()

    def exit(self, game):
        game.song.stop()                                      # stop the preview; game.song is the chosen song

    def draw(self, game, image):
        browse = {side: gesture.progress for side, gesture in self.browse.items()}
        image = draw_song_select(image, game.song, game.song_index, len(game.songs), browse, self.choose.progress)
        return game.scene.draw_title(image, list(game.players.values()), "Choose a song")
