import cv2
import mixbox

def mix_colors(rgb1, rgb2, ratio):
    r, g, b = mixbox.lerp(rgb1, rgb2, ratio)
    return (b, g, r)

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