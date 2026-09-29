import cv2

from functions import mix_colors, put_text_center, put_text_left, put_text_right, scale_for_height

class Scene:
    def __init__(self):
        self.title_top = 0
        self.title_bottom = 50
        self.score_top = self.title_bottom + 20
        self.score_bottom = self.score_top + 16
        self.pose_top = self.score_bottom + 20
        self.pose_bottom = self.pose_top + 20

    def render(self, frame, players, current_pose=None):
        output = frame.copy()
        colors = []
        scores = []

        for player in players:
            colors.append(player["color"])
            scores.append(player["score"])
            output = self._draw_player(output, player)

        output = self._add_title(output, self.title_bottom, colors)
        output = self._add_score(output, scores, self.score_top, self.score_bottom, colors)

        if current_pose is not None:
            output = self._add_current_pose(output, self.pose_top, self.pose_bottom, current_pose)

        return output

    def _draw_player(self, frame, player):
        if not player["visible"]:
            return frame
        keypoints = player["keypoints"]

        for keypoint in keypoints:
            if keypoints[keypoint] is not None and keypoint not in [
                "left_eye", "right_eye", "left_ear", "right_ear", "nose"
            ]:
                x, y = keypoints[keypoint]
                cv2.circle(frame, (x, y), 5, player["color"][::-1], -1)

        # the head circle needs both ears and both eyes; skip it when one is not visible
        if any(keypoints[name] is None for name in ("left_ear", "right_ear", "left_eye", "right_eye")):
            return frame
        # abs: facing the camera, the right ear is on the left side of the image
        head_width = abs(keypoints["right_ear"][0] - keypoints["left_ear"][0])
        head_center = (
            (keypoints["right_eye"][0] + keypoints["left_eye"][0]) // 2,
            keypoints["right_eye"][1],
        )
        cv2.circle(frame, head_center, head_width // 2, player["color"][::-1], -1)
        return frame

    def _add_title(self, frame, title_height, colors):
        width = frame.shape[1]
        font_size = scale_for_height(title_height, cv2.FONT_HERSHEY_TRIPLEX, 2)
        colors_mixed = mix_colors(colors[0], colors[1], 0.5)
        put_text_center(
            frame, "Just Dance", (width // 2, title_height // 2),
            cv2.FONT_HERSHEY_TRIPLEX, font_size, colors_mixed[::-1], 2,
        )
        return frame

    def _add_score(self, frame, score, score_top, score_bottom, colors):
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

        cv2.rectangle(frame, (bar_left, bar_y), (bar_right, bar_y + bar_height), (255, 255, 255), -1)

        p0_end = int(inner_left + inner_width * rel_score_0 - margin_x // 2)
        p1_start = int(inner_right - inner_width * rel_score_1 + margin_x // 2)

        cv2.rectangle(frame, (inner_left, inner_top), (p0_end, inner_bottom), colors[0][::-1], -1)
        cv2.rectangle(frame, (p1_start, inner_top), (inner_right, inner_bottom), colors[1][::-1], -1)

        put_text_right(
            frame, str(score[0]), (bar_left - 10, bar_y + bar_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0][::-1], 1
        )
        put_text_left(
            frame, str(score[1]), (bar_right + 10, bar_y + bar_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1][::-1], 1
        )

        return frame

    def _add_current_pose(self, frame, pose_top, pose_bottom, current_pose):
        width = frame.shape[1]
        font_size = scale_for_height(pose_bottom - pose_top, cv2.FONT_HERSHEY_SIMPLEX, 1)
        put_text_center(
            frame, f"Current Pose: {current_pose['name']}",
            (width // 2, (pose_top + pose_bottom) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2,
        )

        pose_scaling = 50
        pose_points = current_pose["keypoints"]
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

        combinations = [
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
            ("left_shoulder", "neck"), ("right_shoulder", "neck"), ("nose", "neck")
        ]

        for start, end in combinations:
            start_pos = pose_points.get(start)
            end_pos = pose_points.get(end)
            if start_pos is not None and end_pos is not None:
                x1, y1 = start_pos
                x2, y2 = end_pos
                x1 = int(width // 2 + x1 * pose_scaling)
                y1 = int((pose_top + pose_bottom) // 2 + (y1 + 2) * pose_scaling)
                x2 = int(width // 2 + x2 * pose_scaling)
                y2 = int((pose_top + pose_bottom) // 2 + (y2 + 2) * pose_scaling)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        return frame
