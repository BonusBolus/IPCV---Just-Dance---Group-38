# The scene

The `scene/` folder draws the screens of the game. The rule: **the states decide what to show, the scene decides how
it looks.** Every part of the screen has its own file, so you only need to read the file you want to change. The
stage behind the players is in `background/`, see [background_and_star.md](background_and_star.md).

## The order in one frame

`Game.step()` builds every image in the same order, in every state:

1. the stage instead of the room (`background/`);
2. the big heads, the crown and the player labels (`face_tracking/game_effects.py`, see [face_tracking.md](face_tracking.md));
3. the glow and sparkles in the player colours at the screen edges (`Scene.draw_edge_effects()`);
4. the screen of the current state (`state.draw()`, which calls the files below);
5. with `--debug` (and on the results screen of `--record`): the skeletons, the boxes, the FPS and the target pose.

## The files

| File | What is in it | Used by |
|---|---|---|
| `scene.py` | The `Scene` class: the title ("Just" and "Dance" in the player colours), the score bar, the ratings with a "pop", the edge glow, and the debug drawings (`draw_player()`, `draw_target_pose_overlay()`) | all states |
| `functions.py` | Colours (BGR), text with a black outline, big text, progress bars, warning banners, the duet bonus popup | all files below |
| `pose_figure.py` | A pose as a stick figure, and the in-between figure of an animated move | cards, timeline |
| `timeline.py` | The coming moves slide from the right to a hit line at the bottom; the bar behind a move is the hold time | Countdown, Playing |
| `move_cards.py` | The NOW card at the right edge; between moves it already shows the NEXT move (with `SHOW_TIMELINE = False`: a separate NEXT card) | Countdown, Playing |
| `star.py` | The bonus star and the "+100" burst | Playing |
| `song_select_screen.py` | The song card, the arrows, "Raise both hands to choose" | Song selection |
| `registration_screen.py` | The progress bar / "READY" above each head, "Waiting for Player 2...", the colour warning | Registration |
| `results_screen.py` | The scores, the winner, "Raise both hands to play again" | Results |

## Where do I change ...?

| I want to change ... | File |
|---|---|
| a colour (gold, green, the rating colours) | `functions.py`, at the top |
| the text outline, the progress bars, the warning banner | `functions.py` |
| how a pose looks (lines, head, thickness) | `pose_figure.py` |
| the timeline: track size, hit line, how many seconds ahead (`LOOKAHEAD`) | `timeline.py` |
| the size or place of the NOW card, the pulse, the flash | `move_cards.py` |
| the "pop" of the ratings (`POP_TIME`, `POP_SIZE`, `FADE_TIME`) | `scene.py`, `pop()` |
| the title, the score bar, where "Perfect" / "Good" is shown | `scene.py` |
| the glow and sparkles at the screen edges | `scene.py`, `draw_edge_effects()` |
| the star and its burst | `star.py` |
| the song card and the arrows | `song_select_screen.py` |
| the bars above the heads, the "Step back - show your feet" hint | `registration_screen.py` |
| the scores and the winner text | `results_screen.py` |
| the player labels, the crown, the big heads | `face_tracking/game_effects.py` |

## Good to know

- **Colours.** OpenCV uses BGR, the players dict has RGB. Always use `player_color(player)` from `functions.py`.
- **Readable on every background.** `draw_text()` draws the text twice: first thick and black, then in colour.
  The song selection, the results and the move cards first make the image behind them darker.
- **The score bar follows the players** (`Scene.update_sides()`): the player who stands on the left has the left
  part of the bar, and their rating is next to it. The sides only swap when the players clearly stand the other way
  round, so the bar does not flip back and forth.
- **Sizes.** The song selection scales with the image height (`u = height / 720`); the other screens are made for
  1280 x 720 (`GAME_SIZE`).

## How the states call the scene

| State | Calls |
|---|---|
| Song selection | `draw_song_select()`, `scene.draw_title()` |
| Registration | `draw_registration()`, `scene.draw_title()` |
| Countdown | `scene.render()`, `draw_timeline()`, `draw_move_cards()`, `draw_big_text()` |
| Playing | `scene.render()`, `draw_timeline()`, `draw_move_cards()`, `scene.draw_ratings()`, `draw_star()`, `draw_star_burst()`, `draw_duet_bonus()`, `draw_banner()` (lost player) |
| Results | `draw_results()`, `draw_duet_bonus()` |
