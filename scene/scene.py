import cv2

def createScene(frame, score, colors, current_pose):
    title_top = 0
    title_bottom = 50
    score_top = title_bottom + 20
    score_bottom = score_top + 16
    pose_top = score_bottom + 20
    pose_bottom = pose_top + 20

    output = addTitle(frame, title_bottom)
    output = addScore(output, score, score_top, score_bottom, colors)
    output = addCurrentPose(output, pose_top, pose_bottom)
    return output

def addTitle(frame, title_height):
    """Add title to the frame."""
    width = frame.shape[1]
    font_size = scale_for_height(title_height, cv2.FONT_HERSHEY_TRIPLEX, 2)
    putTextCenter(frame, "Just Dance", (width // 2, title_height // 2), cv2.FONT_HERSHEY_TRIPLEX, font_size, (255, 255, 255), 2)
    return frame

def addScore(frame, score, score_top, score_bottom, colors):
    """Add score to the frame."""
    frame_width = frame.shape[1]
    center = frame_width // 2
    bar_width = frame_width // 4

    bar_left = center - bar_width // 2
    bar_right = center + bar_width // 2

    bar_y = score_top
    bar_height = score_bottom - bar_y
    margin_x = 4
    margin_y = 2

    total = score[0] + score[1]
    rel_score_0 = score[0] / total if total > 0 else 0.5
    rel_score_1 = 1 - rel_score_0

    inner_left = bar_left + margin_x // 2
    inner_right = bar_right - margin_x // 2
    inner_width = inner_right - inner_left
    inner_top = bar_y + margin_y
    inner_bottom = bar_y + bar_height - margin_y

    # Background bar
    cv2.rectangle(frame, (bar_left, bar_y), (bar_right, bar_y + bar_height), (255, 255, 255), -1)

    # Player 0 grows from the left, player 1 from the right
    p0_end = int(inner_left + inner_width * rel_score_0 - margin_x // 2)
    p1_start = int(inner_right - inner_width * rel_score_1 + margin_x // 2)

    cv2.rectangle(frame, (inner_left, inner_top), (p0_end, inner_bottom), colors[0], -1)
    cv2.rectangle(frame, (p1_start, inner_top), (inner_right, inner_bottom), colors[1], -1)

    putTextRight(frame, str(score[0]), (bar_left - 10, bar_y + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 1)
    cv2.putText(frame, str(score[1]), (bar_right + 10, bar_y + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 1)

    return frame

def addCurrentPose(frame, pose_top, pose_bottom):
    """Add current pose to the frame."""
    width = frame.shape[1]
    font_size = scale_for_height(pose_bottom - pose_top, cv2.FONT_HERSHEY_SIMPLEX, 1)
    putTextCenter(frame, "Current Pose: T-Pose", (width // 2, (pose_top + pose_bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    return frame

def putTextRight(frame, text, bottom_right, font, scale, color, thickness=1):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = bottom_right[0] - text_w
    y = bottom_right[1]
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame

def putTextCenter(frame, text, center, font, scale, color, thickness=1):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = center[0] - text_w // 2
    y = center[1] + text_h // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    return frame

def scale_for_height(target_height, font, thickness=1, text="Ag"):
    (_, h), _ = cv2.getTextSize(text, font, 1.0, thickness)
    return target_height / h
