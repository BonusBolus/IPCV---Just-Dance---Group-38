import cv2
import config


def put_text_left(frame, text, bottom_left, font, scale, color, thickness=1):
    x = bottom_left[0]
    y = bottom_left[1]
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame

def put_text_right(frame, text, bottom_right, font, scale, color, thickness=1):
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = bottom_right[0] - text_w
    y = bottom_right[1]
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame

def put_text_center(frame, text, center, font, scale, color, thickness=1):
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - text_w // 2
    y = center[1] + text_h // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame

def scale_for_height(target_height, font, thickness=1, text="Ag"):
    (_, h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    return target_height / h


# Colours in OpenCV order (BGR)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 220, 80)
GOLD = (0, 215, 255)
DUET = (255, 220, 80)
FLOW = (220, 80, 255)    # animated moves (magenta)
RATING_COLORS = {"Perfect": GOLD, "Good": GREEN, "OK": (255, 200, 80), "Miss": (60, 60, 230)}


def player_color(player):
    """Display colour of a player in BGR (the players dict has RGB)."""
    return tuple(int(c) for c in player[config.PLAYER_COLOR][::-1])


def draw_text(frame, text, center, size=1.0, color=WHITE, thickness=2):
    """Centred text with a black outline, so it is readable on every background."""
    put_text_center(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 3)
    put_text_center(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return frame


def draw_big_text(frame, text):
    """Countdown numbers and "Dance!" in the middle of the screen."""
    height, width = frame.shape[:2]
    return draw_text(frame, text, (width // 2, height // 2), 4.0, WHITE, 8)


def draw_duet_bonus(frame, progress, center_y=None):
    """A gold popup below the ratings, rising slightly and fading before it expires."""
    height, width = frame.shape[:2]
    center = (width // 2, (min(175, height // 3) if center_y is None else center_y) - int(12 * progress))
    size = min(1.2, width / 420)
    text = "DUET BONUS"
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 2)
    overlay = frame.copy()
    x, y = center
    cv2.rectangle(overlay, (x - text_w // 2 - 18, y - text_h // 2 - 12),
                  (x + text_w // 2 + 18, y + text_h // 2 + baseline + 12), (20, 20, 20), -1)
    draw_text(overlay, text, center, size, GOLD)
    opacity = min(1.0, max(0.0, (1.0 - progress) / 0.25))
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, dst=frame)
    return frame


def draw_bar(frame, top_left, size, fraction, color):
    """Progress bar, `fraction` 0..1."""
    (x, y), (w, h) = top_left, size
    fill = int(w * min(max(fraction, 0.0), 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), WHITE, 1)
    return frame


def draw_banner(frame, text, color, row=0):
    """Coloured bar with white text, for warnings ("Player 2 lost - step back in!").
    `row` puts a second banner under the first one."""
    height, width = frame.shape[:2]
    y = int(height * 0.3) + row * 60
    cv2.rectangle(frame, (width // 5, y), (width * 4 // 5, y + 50), color, -1)
    return draw_text(frame, text, (width // 2, y + 25), 0.9, WHITE)