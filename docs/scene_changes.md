# Scene changes

The game has 4 screens: Registration, Countdown, Playing and Results (see [game_structure.md](game_structure.md)).
The `scene/` folder draws them. The rule: **the states decide what to show, the scene decides how it looks.**

Every part of the screen has **its own file** now, so you only need to read the file you want to change.

**Nothing old was removed or changed.** `scene.py` is the original, with one new method. `functions.py` is the original, with new helpers at the end.
We tested it: everything draws **the same pixels** as before.

## The files

| File | What is in it | Used by |
|---|---|---|
| `scene.py` | **The original** `Scene` class: title, score bar, current pose, `draw_player()`. **New:** `draw_ratings()`, `draw_title()`, `draw_edge_effects()`, the score boxes for 1 or 3-5 players | all states |
| `functions.py` | **The original** text helpers. **New at the end:** colours and small building blocks | all files below |
| `pose_figure.py` | **New.** Draws a pose as a stick figure | `move_cards.py` |
| `move_cards.py` | **New.** The NOW and NEXT cards (with the timeline on: only NOW, at the right edge) | Countdown, Playing |
| `timeline.py` | **New.** The coming moves slide from the right to a hit line at the bottom; the bar is the hold time. `SHOW_TIMELINE` in `config.py` | Countdown, Playing |
| `song_select_screen.py` | **New.** Everything on the song selection screen | Song selection |
| `registration_screen.py` | **New.** Everything on the Registration screen | Registration |
| `results_screen.py` | **New.** Everything on the Results screen | Results |

How the files use each other:

```
song_select_screen.py ──┐
registration_screen.py ─┤
results_screen.py ──────┤
move_cards.py ──────────┼──> functions.py   (colours, text, bars, banners)
scene.py ───────────────┘
move_cards.py ─────────────> pose_figure.py (stick figure)
```

## Where do I change ...?

| I want to change ... | File |
|---|---|
| a colour (gold, green, the rating colours) | `functions.py`, at the top of the new part |
| the text outline, the progress bars, the red warning banner | `functions.py` |
| how a pose looks (lines, head, thickness) | `pose_figure.py` |
| the size or place of the NOW / NEXT cards, the pulse, the flash | `move_cards.py` |
| the timeline: track size, hit line, how many seconds ahead (`LOOKAHEAD`) | `timeline.py` |
| the "pop" of the ratings (`POP_TIME`, `POP_SIZE`, `FADE_TIME`) | `scene.py`, `pop()` |
| the "Step back - show your feet" hint | `registration_screen.py` |
| the progress bar / "READY" above the head, "Waiting for Player 2..." | `registration_screen.py` |
| the "Player 1" / "Player 2" labels on the crown, the crown, the big heads | `face_tracking/game_effects.py`, see [face_tracking.md](face_tracking.md) |
| the glow and sparkles at the screen edges | `scene.py`, `draw_edge_effects()` |
| the song card, the arrows, "Raise both hands to choose" | `song_select_screen.py` |
| the scores, the winner text, "Raise both hands to play again" | `results_screen.py` |
| the title, the score bar, where "Perfect" / "Good" is shown | `scene.py` |

---

## `scene.py`: one new method

Only this was added (and `RATING_COLORS, draw_text` in the import line):

`draw_ratings(frame, ratings, ages=None, total_time=1.0)` draws "Perfect", "Good", ... at the player's score.
`ratings` is `{player_id: "Perfect"}`, `ages` is `{player_id: seconds since the rating came}` (a new rating pops, see `pop()`).
`render()` must be called first in the same frame.


It is in `scene.py` because the ratings are placed under the score bar (`self.score_bottom`).
The score bar follows where the players stand (`update_sides()`): the player on the left has the left part
and the left number, and their rating is right next to it. The sides only swap when the players clearly
stand the other way round, so the bar does not flip back and forth.
`render(frame, players)` without a pose is used for the title and score bar during Countdown and Playing.

**1 to 5 players.** `render()` only shows the registered players:
- exactly 2 players: the score bar, as above;
- 1 or 3-5 players: a **row of score boxes** at the top (`_add_badges()`), in the order the players stand. Each box has
  the player colour, the player number, the score and a small bar that shows the score compared to the best score.
  The rating is shown under the player's box. Small images get smaller boxes ("P1" instead of "Player 1").
- `update_sides()` works for any number of players: two neighbours only swap when they clearly stand the other way
  round, and a player who is out of view keeps their place.

Two more methods were added later:

| Method | What it does | Used by |
|---|---|---|
| `draw_title(frame, players, prompt=None)` | "Just Dance" at the top: "Just" in the colour of player 1, "Dance" in that of player 2. `prompt` is an optional smaller line under it. `render()` uses it too. | Registration (with the prompt), `render()` |
| `draw_edge_effects(frame, players, t)` | A soft glow in the player colours along the left and right edge that pulses slowly, with sparkles rising along the edges. `t` (seconds) drives the animation. | `Game.step()`, every state |

## `functions.py`: building blocks

The old functions (`mix_colors`, `put_text_left`, `put_text_right`, `put_text_center`, `scale_for_height`) are the same. New at the end:

| Name | What it does |
|---|---|
| `WHITE`, `GREY`, `GREEN`, `GOLD`, `RATING_COLORS` | Colours in OpenCV order (**BGR**) |
| `player_color(player)` | The colour of a player in BGR. The players dict has **RGB**, so always use this. |
| `draw_text(frame, text, center, size, color)` | Centred text with a black outline, so it is readable on every camera image. It draws the text twice: first thick and black, then in colour. |
| `draw_big_text(frame, text)` | Big text in the middle: "3", "2", "1", "Dance!" |
| `draw_bar(frame, top_left, size, fraction, color)` | A progress bar. `fraction` goes from 0 to 1. |
| `draw_banner(frame, text, color, row)` | A coloured bar with white text, for warnings. `row=1` puts it under the first one. |

`functions.py` now also imports `config`, for `config.PLAYER_COLOR` in `player_color()`.

## `pose_figure.py`: the stick figure

```python
draw_pose(frame, pose, center_x, nose_y, scale, color, thickness=1)
```

- `pose`: a pose from `poses.json`, like `{"nose": (0, 0), "left_wrist": (-1.3, -0.3), ...}`
- `center_x`, `nose_y`: where the nose goes on the screen (pixels)
- `scale`: pixels per pose unit
- `BONES` is the list of lines between the points. It is the same as the `combinations` list in `scene._add_current_pose()`.

## `move_cards.py`: NOW and NEXT

```python
draw_move_cards(frame, now_move, now_progress, next_move, next_progress, beat_phase, flash)
```

- **NOW** (big card): the move that counts now. The bar gets empty while the player holds the pose. A short flash shows that a new move started. Between moves: "Get ready".
- **NEXT** (small card): the coming move, visible from the countdown on. The bar fills up until the move starts. The figure is 8% bigger on the beat: that is the pulse.
- **Gold moves**: gold border and the label "GOLD".
- All numbers (`now_progress`, `next_progress`, `beat_phase`) come from `song.py`. This file only draws them.
- `draw_card()` draws one card. The card sizes and places are at the top of `draw_move_cards()`.

## `registration_screen.py`

```python
draw_registration(frame, players, heads, ready, progress, colors_similar)
```

- `heads` is `game.heads`: the face of each player, or `None` (see [face_tracking.md](face_tracking.md)).
- The label "Player 1" is **not** drawn here: `Game.step()` writes it on the forehead (or on the crown) in every state.
- Just above the head of each player (where the tips of the crown would be): the progress bar of the hands-up gesture, as wide as the crown. When the player is ready: "READY".
- When no face is found for a player, the label and the bar are drawn at the top of the player's box instead.
- A player that is not found yet: "Waiting for Player 2..." at the bottom.
- At the top: "Raise both hands when you are ready".
- `colors_similar=True`: the banner "Please wear different colours".
- `player["box"]` is `(x, y, w, h)` from the IdentityTracker.

## `results_screen.py`

```python
draw_results(frame, ranking, progress, show_hint)
```

- The image gets darker, and the ranking and the winner are shown in the middle.
- `ranking` is `[(place, player dict), ...]` from the highest score down; players with the same score share a place.
  The Results **state** makes it (`rank_players()`); this file only shows it. 1 player: "Well done!", a shared 1st place: "Draw!".
- `show_hint=True` (after 3 s): "Raise both hands to play again" and the progress bar.
- `draw_lines()` draws the lines of text. Each line is `(text, color, size)`.

---

## How the states call the scene

| State | Calls |
|---|---|
| Song selection | `draw_song_select()`, `game.scene.draw_title()` |
| Registration | `draw_registration()`, `game.scene.draw_title()` |
| Countdown | `game.scene.render()`, `draw_move_cards()`, `draw_big_text()` |
| Playing | `game.scene.render()`, `draw_move_cards()`, `game.scene.draw_ratings()`, `draw_banner()`, `draw_big_text()` |
| Results | `draw_results()` |

`Game.step()` draws the head effects (big heads, crown, player label) and `draw_edge_effects()` **before** the state draws, in every state.

Want to change how something looks? Only change the file in `scene/` (or `face_tracking/game_effects.py` for the head effects). The game rules in `game_logic/states/` stay the same.
