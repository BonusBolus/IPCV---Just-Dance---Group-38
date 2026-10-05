import math
import cv2

from scene.functions import RATING_COLORS, draw_text, mix_colors, put_text_center, put_text_left, put_text_right, scale_for_height
from scene.pose_figure import BONES

class Scene:
    def __init__(self, color_key="player_color"):
        self.color_key = color_key
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
            colors.append(player[self.color_key])
            scores.append(player["score"])

        output = self._add_title(output, self.title_bottom, colors)
        output = self._add_score(output, scores, self.score_top, self.score_bottom, colors)

        if current_pose is not None:
            output = self._add_current_pose(output, self.pose_top, self.pose_bottom, current_pose)

        return output

    def draw_ratings(self, frame, ratings):
        """"Perfect", "Good", ... under the score of each player. ratings: {player_id: "Perfect"}"""
        width = frame.shape[1]
        for player_id, rating in ratings.items():
            x = width // 2 + (-1 if player_id == 1 else 1) * width // 8
            draw_text(frame, rating, (x, self.score_bottom + 35), 1.0, RATING_COLORS[rating])
        return frame

    def draw_player(self, frame, players):
        """Draws keypoints and skeleton bones in the player color, for debugging."""
        for player in players.values():
            if not player["visible"]:
                continue
            keypoints = player["keypoints"]
            color = player[self.color_key][::-1]  # RGB to BGR

            # Derive neck keypoint if shoulders exist
            ls = keypoints.get("left_shoulder")
            rs = keypoints.get("right_shoulder")
            neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2) if (ls is not None and rs is not None) else None

            pts = {**keypoints, "neck": neck}

            # Draw bones
            for start, end in BONES:
                p1 = pts.get(start)
                p2 = pts.get(end)
                if p1 is not None and p2 is not None:
                    cv2.line(frame, p1, p2, color, 3)

            # Draw keypoint dots
            for keypoint, pos in pts.items():
                if pos is not None and keypoint not in ("left_eye", "right_eye", "left_ear", "right_ear"):
                    cv2.circle(frame, pos, 5, color, -1)
        return frame

    def draw_target_pose_overlay(self, frame, players, pose):
        """Overlay the target reference pose stick figure onto each visible player, for debugging."""
        if pose is None:
            return frame

        pose_points = {**pose, "neck": (0, 0.3)}
        overlay_color = (0, 255, 255)  # Bright yellow (BGR) overlay for target pose comparison

        for player in players.values():
            if not player["visible"]:
                continue

            kp = player["keypoints"]
            ls = kp.get("left_shoulder")
            rs = kp.get("right_shoulder")

            if ls is not None and rs is not None:
                center_x = (ls[0] + rs[0]) / 2.0
                center_y = (ls[1] + rs[1]) / 2.0
                shoulder_dist = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
                scale = max(shoulder_dist, 30.0)
            elif player["box"] is not None:
                bx, by, bw, bh = player["box"]
                center_x = bx + bw / 2.0
                center_y = by + bh * 0.3
                scale = bw * 0.6
            else:
                continue

            target_pixel_pts = {}
            for k, pt in pose_points.items():
                if pt is not None:
                    px, py = pt
                    px_pixel = int(center_x + px * scale)
                    py_pixel = int(center_y + (py - 0.5) * scale)
                    target_pixel_pts[k] = (px_pixel, py_pixel)

            for start, end in BONES:
                p1 = target_pixel_pts.get(start)
                p2 = target_pixel_pts.get(end)
                if p1 is not None and p2 is not None:
                    cv2.line(frame, p1, p2, overlay_color, 2)

            for k, pt in target_pixel_pts.items():
                cv2.circle(frame, pt, 4, overlay_color, -1)

            if "nose" in target_pixel_pts:
                cv2.circle(frame, target_pixel_pts["nose"], int(scale * 0.2), overlay_color, 2)

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
