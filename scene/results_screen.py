from scene.functions import GREEN, WHITE, draw_bar, draw_text, player_color


PLACE_WORDS = {1: "1st", 2: "2nd", 3: "3rd"}


def draw_results(frame, ranking, progress, show_hint):
    """ranking:   [(place, player dict), ...] from the highest score down (see rank_players)
    progress:  how far the hands-up gesture is (0..1)
    show_hint: show "Raise both hands to play again" and the progress bar"""
    lines = [("Results", WHITE, 2.0)]
    for place, player in ranking:
        place_word = PLACE_WORDS.get(place, f"{place}th")
        text = f"Player {player['id']}: {player['score']}" if len(ranking) == 1 else \
            f"{place_word}  Player {player['id']}: {player['score']}"
        lines.append((text, player_color(player), 1.2))
    if len(ranking) == 1:
        lines.append(("Well done!", WHITE, 1.5))
    elif len(ranking) > 1 and ranking[1][0] == 1:
        lines.append(("Draw!", WHITE, 1.5))                  # the best score is shared
    elif ranking:
        lines.append((f"Player {ranking[0][1]['id']} wins!", player_color(ranking[0][1]), 1.5))
    if show_hint:
        lines.append(("Raise both hands to play again", WHITE, 0.9))
    return draw_lines(frame, lines, progress if show_hint else None)


def draw_lines(frame, lines, progress=None):
    """Darker image with lines of text in the middle. lines: [(text, color, size), ...]"""
    height, width = frame.shape[:2]
    frame[:] = (frame * 0.4).astype(frame.dtype)
    steps = [int(30 * size) + 30 for _, _, size in lines]
    fit = min(1.0, (height - 60) / sum(steps))                 # many lines (5 players) on a small image: smaller
    lines = [(text, color, size * fit) for text, color, size in lines]
    steps = [int(step * fit) for step in steps]
    y = (height - sum(steps)) // 2
    for (text, color, size), step in zip(lines, steps):
        draw_text(frame, text, (width // 2, y + step // 2), size, color, 2 if size < 2 else 4)
        y += step
    if progress is not None:
        draw_bar(frame, (width // 2 - 150, y + 20), (300, 16), progress, GREEN)
    return frame
