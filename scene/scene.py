import cv2
import mixbox

def createScene(frame, players, current_pose):
    title_top = 0
    title_bottom = 50
    score_top = title_bottom + 20
    score_bottom = score_top + 16
    pose_top = score_bottom + 20
    pose_bottom = pose_top + 20

    output = frame.copy()
    colors = []
    score = []
    for player in players:
        colors.append(player["color"])
        score.append(player["score"])
        output = drawPlayer(output, player)

    output = addTitle(output, title_bottom, colors)
    output = addScore(output, score, score_top, score_bottom, colors)
    output = addCurrentPose(output, pose_top, pose_bottom, current_pose)
    return output

def drawPlayer(frame, player):
    keypoints = player["keypoints"]
    for keypoint in keypoints:
        if keypoints[keypoint] is not None and keypoint not in ["left_eye","right_eye","left_ear","right_ear","nose"]:
            x, y = keypoints[keypoint]
            cv2.circle(frame, (x, y), 5, player["color"][::-1], -1)
    head_width = keypoints["right_ear"][0] - keypoints["left_ear"][0]
    head_center = ((keypoints["right_eye"][0] + keypoints["left_eye"][0]) // 2,keypoints["right_eye"][1])
    cv2.circle(frame,head_center,head_width // 2,player["color"][::-1], -1)
    return frame

def addTitle(frame, title_height, colors):
    """Add title to the frame."""
    width = frame.shape[1]
    font_size = scale_for_height(title_height, cv2.FONT_HERSHEY_TRIPLEX, 2)
    colors_mixed = mixColors(colors[0], colors[1], 0.5)
    putTextCenter(frame, "Just Dance", (width // 2, title_height // 2), cv2.FONT_HERSHEY_TRIPLEX, font_size, colors_mixed[::-1], 2) # BGR
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

    cv2.rectangle(frame, (inner_left, inner_top), (p0_end, inner_bottom), colors[0][::-1], -1) # BGR
    cv2.rectangle(frame, (p1_start, inner_top), (inner_right, inner_bottom), colors[1][::-1], -1) # BGR

    putTextRight(frame, str(score[0]), (bar_left - 10, bar_y + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0][::-1], 1) # BGR
    putTextLeft(frame, str(score[1]), (bar_right + 10, bar_y + bar_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1][::-1], 1) # BGR

    return frame

def addCurrentPose(frame, pose_top, pose_bottom, current_pose):
    """Add current pose to the frame."""
    width = frame.shape[1]
    font_size = scale_for_height(pose_bottom - pose_top, cv2.FONT_HERSHEY_SIMPLEX, 1)
    putTextCenter(frame, f"Current Pose: {current_pose['name']}", (width // 2, (pose_top + pose_bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

    pose_scaling = 50
    pose_points = current_pose['keypoints']
    pose_points = {**pose_points, "neck": (0, 0.3)}
    for keypoint in pose_points:
        if pose_points[keypoint] is not None:
            x, y = pose_points[keypoint]
            x = int(width // 2 + x * pose_scaling)
            y = int((pose_top + pose_bottom) // 2 + (y + 2) * pose_scaling)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
        if keypoint == "nose":
            x, y = pose_points[keypoint]
            x = int(width // 2 + x * pose_scaling)
            y = int((pose_top + pose_bottom) // 2 + (y + 2) * pose_scaling)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)

    combinations = [("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),("left_shoulder", "neck"), ("right_shoulder", "neck"), ("nose", "neck")]    
    for start, end in combinations:
        start_pos = pose_points.get(start)
        end_pos = pose_points.get(end)
        if start_pos is not None and end_pos is not None:
            x1, y1 = start_pos
            x2, y2 = end_pos
            x1 = int(width // 2 + x1 * pose_scaling)
            y1 = int((pose_top + pose_bottom) // 2 + (y1+2) * pose_scaling)
            x2 = int(width // 2 + x2 * pose_scaling)
            y2 = int((pose_top + pose_bottom) // 2 + (y2+2) * pose_scaling)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame

def mixColors(rgb1, rgb2, ratio):
    r, g, b = mixbox.lerp(rgb1, rgb2, ratio)
    return (b, g, r)

def putTextLeft(frame, text, bottom_left, font, scale, color, thickness=1):
    x = bottom_left[0]
    y = bottom_left[1]
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
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
