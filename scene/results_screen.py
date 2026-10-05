"""The Results screen: darker image, the scores and the winner in the middle."""
from scene.functions import GREEN, WHITE, draw_bar, draw_text, player_color


def draw_results(frame, players, winner, progress, show_hint):
    """winner:    the player dict of the winner, or None for a draw
    progress:  how far the hands-up gesture is (0..1)
    show_hint: show "Raise both hands to play again" and the progress bar
    """
    player_1, player_2 = players[1], players[2]
    if winner is None:
        winner_line = ("Draw!", WHITE, 1.5)
    else:
        winner_line = (f"Player {winner['id']} wins!", player_color(winner), 1.5)

    lines = [("Results", WHITE, 2.0),
             (f"Player 1: {player_1['score']}", player_color(player_1), 1.2),
             (f"Player 2: {player_2['score']}", player_color(player_2), 1.2),
             winner_line]
    if show_hint:
        lines.append(("Raise both hands to play again", WHITE, 0.9))
    return draw_lines(frame, lines, progress if show_hint else None)


def draw_lines(frame, lines, progress=None):
    """Darker image with lines of text in the middle. lines: [(text, color, size), ...]"""
    height, width = frame.shape[:2]
    frame[:] = (frame * 0.4).astype(frame.dtype)
    steps = [int(30 * size) + 30 for _, _, size in lines]
    y = (height - sum(steps)) // 2
    for (text, color, size), step in zip(lines, steps):
        draw_text(frame, text, (width // 2, y + step // 2), size, color, 2 if size < 2 else 4)
        y += step
    if progress is not None:
        draw_bar(frame, (width // 2 - 150, y + 20), (300, 16), progress, GREEN)
    return frame
