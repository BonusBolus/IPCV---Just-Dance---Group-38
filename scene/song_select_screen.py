"""The song selection screen: one song in a card in the middle, arrows for the hands, and a bar to choose.

  <   [ Song 2 / 5            ]   >
      [ Rasputin              ]
      [ 3:41  |  127 bpm      ]
      [ 58 moves | 10 duets   ]
  left hand              right hand

          Raise both hands to choose
          [=========         ]
"""
import cv2
import numpy as np

from scene.functions import GOLD, GREEN, GREY, WHITE, draw_bar, draw_text

TOP_MARGIN = 150   # Scene.draw_title() uses the top of the screen (title + prompt), see registration_screen.py


def draw_song_select(frame, song, index, count, browse, choose_progress):
    """song:            the selected Song
    index, count:    "Song 2 / 5"
    browse:          {"left": 0..1, "right": 0..1}: how far the one-hand gesture is
    choose_progress: 0..1: how far the both-hands gesture is"""
    height, width = frame.shape[:2]
    u = height / 720                                   # sizes are made for 720 px high
    frame[:] = (frame * 0.45).astype(frame.dtype)      # darker, so the card is easy to read

    card_w, card_h = int(width * 0.46), int(height * 0.40)
    left, top = (width - card_w) // 2, max(TOP_MARGIN, int(height * 0.25))
    center_x = width // 2
    region = frame[top:top + card_h, left:left + card_w]
    region[:] = (region * 0.5).astype(frame.dtype)
    cv2.rectangle(frame, (left, top), (left + card_w, top + card_h), WHITE, 2)

    draw_text(frame, f"Song {index + 1} / {count}", (center_x, top + int(card_h * 0.13)), 0.7 * u, GREY)
    draw_text(frame, song.title, (center_x, top + int(card_h * 0.38)), title_size(song.title, card_w - 40, u), WHITE, 3)
    minutes, seconds = divmod(int(round(song.play_length)), 60)
    draw_text(frame, f"{minutes}:{seconds:02d}   |   {song.bpm:g} bpm", (center_x, top + int(card_h * 0.62)), 0.9 * u, WHITE)
    details = f"{len(song.played_moves)} moves   |   {song.duet_moves} duets   |   {song.gold_moves} gold"
    draw_text(frame, details, (center_x, top + int(card_h * 0.82)), 0.75 * u, GOLD if song.gold_moves else GREY)

    # the arrows: left hand = previous song, right hand = next song
    arrow_y, arrow_size = top + card_h // 2, int(40 * u)
    for side, direction, x in (("left", -1, left - int(70 * u)), ("right", 1, left + card_w + int(70 * u))):
        progress = browse[side]
        color = GREEN if progress > 0 else WHITE
        tip = (x + direction * arrow_size, arrow_y)
        back_top, back_bottom = (x - direction * arrow_size, arrow_y - arrow_size), (x - direction * arrow_size, arrow_y + arrow_size)
        cv2.fillPoly(frame, [np.array([tip, back_top, back_bottom], np.int32)], color, cv2.LINE_AA)
        draw_text(frame, f"{side} hand", (x, arrow_y + arrow_size + int(28 * u)), 0.75 * u, color)
        bar_w = int(90 * u)
        draw_bar(frame, (x - bar_w // 2, arrow_y + arrow_size + int(45 * u)), (bar_w, max(6, int(10 * u))), progress, GREEN)

    choose_y = min(height - int(70 * u), top + card_h + int(110 * u))
    draw_text(frame, "Raise both hands to choose", (center_x, choose_y), 0.9 * u, WHITE)
    bar_w = int(300 * u)
    draw_bar(frame, (center_x - bar_w // 2, choose_y + int(22 * u)), (bar_w, max(8, int(16 * u))), choose_progress, GREEN)
    return frame


def title_size(title, max_width, u):
    """Font size for the song title: big, but it must fit in the card."""
    size = 1.8 * u
    while size > 0.6 * u and cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, size, 3)[0][0] > max_width:
        size -= 0.1 * u
    return size
