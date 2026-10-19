from face_tracking.game_effects import crown_top
from scene.functions import GREEN, draw_banner, draw_bar, draw_text, player_color

# Scene.draw_title() uses the top of the screen: the title until y = 80, the prompt around y = 115.
# Labels and bars stay below this line, so they never cover the title or the prompt.
TOP_MARGIN = 140


HINT = (0, 165, 255)   # orange (BGR)


def draw_registration(frame, players, heads, ready, progress, colors_similar, full_body=None,
                      step_back_text="Step back - show your feet"):
    """players:     dict from the IdentityTracker, {1: {...}, 2: {...}}
    heads:          {1: face or None, ...}: the face of each player (game_effects.faces_for_players)
    ready:          {1: True, 2: False}: did the player finish the hands-up gesture?
    progress:       {1: 0.4, 2: 0.0}: how far the hands-up gesture is (0..1)
    colors_similar: True if the shirts look the same (then the tracker can mix up the players)
    full_body:      {1: True, 2: False}: does the camera see what the grading needs? If not: step_back_text
    step_back_text: for example "Step back - show your knees" (the Registration state makes it from the grader)"""
    height, width = frame.shape[:2]
    away = [player for player in players.values() if player["registered"] and not player["visible"]]
    for index, player in enumerate(away):                   # registered, but out of view right now
        spot_x = int(width * (index + 1) / (len(away) + 1))
        draw_text(frame, f"Waiting for Player {player['id']}...", (spot_x, height - 60), 0.8, player_color(player))
    if not any(player["registered"] for player in players.values()):
        draw_text(frame, f"Step in to play (up to {len(players)} players)", (width // 2, height - 60), 0.9)
    for player_id, player in players.items():
        color = player_color(player)
        if not player["visible"] or player["box"] is None:
            continue                                        # an empty slot, or away (see above)
        face = heads.get(player_id)
        if face is not None:
            # just above the tips of the crown, as wide as the crown
            top_x, top_y = crown_top(face)
            bar_width = max(80, int(face["size"][0] * 1.18))
            center_x, bar_y = int(top_x), int(top_y) - 22
            if bar_y < TOP_MARGIN:
                # no room above the crown (the player stands close to the camera): under the chin
                chin_y = face["location"][1] + face["size"][1] // 2
                bar_y = max(int(chin_y) + 12, TOP_MARGIN)
        else:
            # no face found: fall back to the top of the player's box (with the label there too)
            x, y, w, h = player["box"]
            center_x, label_y = x + w // 2, max(y - 50, TOP_MARGIN + 15)
            draw_text(frame, f"Player {player_id}", (center_x, label_y), 1.0, color, 3)
            bar_width, bar_y = 140, label_y + 20
        if ready[player_id]:
            draw_text(frame, "READY", (center_x, bar_y + 7), 0.9, GREEN)
        else:
            draw_bar(frame, (center_x - bar_width // 2, bar_y), (bar_width, 14), progress[player_id], color)
        if full_body is not None and step_back_text and not full_body.get(player_id, True):
            hint_x = min(max(center_x, 150), width - 150)      # keep the hint on the screen
            draw_text(frame, step_back_text, (hint_x, bar_y + 42), 0.7, HINT)

    if colors_similar:
        draw_banner(frame, "Please wear different colours", (0, 0, 160))
    return frame
