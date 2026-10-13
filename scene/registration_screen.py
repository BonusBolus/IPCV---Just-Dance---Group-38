"""The Registration screen: the camera image with a label above each player."""
from scene.functions import GREEN, draw_banner, draw_bar, draw_text, player_color


def draw_registration(frame, players, ready, progress, colors_similar):
    """players:        dict from the IdentityTracker, {1: {...}, 2: {...}}
    ready:          {1: True, 2: False}: did the player finish the hands-up gesture?
    progress:       {1: 0.4, 2: 0.0}: how far the hands-up gesture is (0..1)
    colors_similar: True if the shirts look the same (then the tracker can mix up the players)
    """
    height, width = frame.shape[:2]
    for player_id, player in players.items():
        color = player_color(player)
        if not player["visible"] or player["box"] is None:
            spot_x = width // 4 if player_id == 1 else width * 3 // 4
            draw_text(frame, f"Waiting for Player {player_id}...", (spot_x, height - 60), 0.9, color)
            continue
        x, y, w, h = player["box"]
        center_x, label_y = x + w // 2, max(y - 50, 100)
        draw_text(frame, f"Player {player_id}", (center_x, label_y), 1.0, color, 3)
        if ready[player_id]:
            draw_text(frame, "READY", (center_x, label_y + 35), 0.9, GREEN)
        else:
            draw_bar(frame, (center_x - 70, label_y + 20), (140, 14), progress[player_id], color)

    if colors_similar:
        draw_banner(frame, "Please wear different colours", (0, 0, 160))
    return frame
