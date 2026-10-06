import math

import cv2
import numpy as np

from scene.functions import RATING_COLORS, draw_text, put_text_center, put_text_left, put_text_right, scale_for_height

class Scene:
    def __init__(self, color_key="player_color"):
        self.color_key = color_key
        self.title_top = 0
        self.title_height = 60
        self.title_bottom = 20 + self.title_height + 10
        self.glow = None    # edge glow fade, made on first use (see draw_edge_effects)
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

        output = self.draw_title(output, players, text_height=self.title_height)
        output = self._add_score(output, scores, self.score_top, self.score_bottom, colors)

        if current_pose is not None:
            output = self._add_current_pose(output, self.pose_top, self.pose_bottom, current_pose)

        return output

    def draw_title(self, frame, players, prompt=None, text_height=60):
        """"Just Dance" at the top in the middle: "Just" in the colour of player 1, "Dance" in that of player 2.
        players: list of player dicts, like render()
        prompt:  optional smaller line under the title, e.g. "Raise both hands when you are ready" """
        width = frame.shape[1]
        font, thickness = cv2.FONT_HERSHEY_TRIPLEX, 3
        font_size = scale_for_height(text_height, font, thickness)
        words = ["Just ", "Dance"]
        widths = [cv2.getTextSize(word, font, font_size, thickness)[0][0] for word in words]
        x = (width - sum(widths)) // 2
        y = 20 + text_height
        for word, word_width, player in zip(words, widths, players):
            color = tuple(int(c) for c in player[self.color_key][::-1])     # RGB -> BGR
            cv2.putText(frame, word, (x, y), font, font_size, (0, 0, 0), thickness + 4, cv2.LINE_AA)  # outline
            cv2.putText(frame, word, (x, y), font, font_size, color, thickness, cv2.LINE_AA)
            x += word_width
        if prompt:
            draw_text(frame, prompt, (width // 2, y + 35), 0.8)
        return frame

    def draw_edge_effects(self, frame, players, t, edge=0.12, sparkles=14):
        """A soft glow along the left and right edge (player 1 colour left, player 2 colour right)
        that slowly pulses, with small sparkles rising along the edges. Changes `frame` in place.

        players: list of player dicts, like render()
        t:       time in seconds, drives the animation
        edge:    width of the glow, as a fraction of the screen width
        """
        height, width = frame.shape[:2]
        strip = max(1, int(width * edge))
        colors = [tuple(int(c) for c in player[self.color_key][::-1]) for player in players[:2]]  # RGB -> BGR

        # glow: mix the player colour into the image, strong at the edge and fading to the inside
        if self.glow is None or self.glow.shape != (height, strip):
            fade = np.linspace(1, 0, strip, dtype=np.float32) ** 2
            self.glow = np.ascontiguousarray(np.tile(fade, (height, 1)))               # (height, strip)
        pulse = 0.55 + 0.2 * math.sin(2 * math.pi * 0.5 * t)     # 0.35..0.75, one pulse every 2 seconds
        alpha = self.glow * pulse
        for side, weight, color in ((frame[:, :strip], alpha, colors[0]),
                                    (frame[:, width - strip:], np.ascontiguousarray(alpha[:, ::-1]), colors[1])):
            color_layer = np.full((height, strip, 3), color, np.uint8)
            side[:] = cv2.blendLinear(color_layer, np.ascontiguousarray(side), weight, 1 - weight)

        # sparkles: rise from the bottom and twinkle; the position follows from the time, so no state is needed
        for i in range(sparkles * 2):
            on_left = i % 2 == 0
            seed = (i * 0.618034) % 1.0                     # spreads the sparkles evenly
            speed = 0.08 + 0.07 * ((i * 0.381966) % 1.0)    # screen heights per second
            y = int((1.0 - (seed + t * speed) % 1.0) * height)
            offset = int(strip * (0.15 + 0.6 * ((i * 0.7548) % 1.0)))
            x = offset if on_left else width - 1 - offset
            twinkle = 0.5 + 0.5 * math.sin(2 * math.pi * (1.5 * t + seed))
            size = 2 + int(4 * twinkle)
            base = colors[0] if on_left else colors[1]
            color = tuple(int(c + (255 - c) * twinkle) for c in base)   # from the player colour to white
            cv2.line(frame, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)
        return frame

    def draw_ratings(self, frame, ratings):
        """"Perfect", "Good", ... under the score of each player. ratings: {player_id: "Perfect"}"""
        width = frame.shape[1]
        for player_id, rating in ratings.items():
            x = width // 2 + (-1 if player_id == 1 else 1) * width // 8
            draw_text(frame, rating, (x, self.score_bottom + 35), 1.0, RATING_COLORS[rating])
        return frame

    def draw_player(self, frame, players):
        """Draws the keypoints and a head circle in the player color, for debugging. players: dict from IdentityTracker.update()."""
        for player in players.values():
            if not player["visible"]:
                continue
            keypoints = player["keypoints"]
            color = player[self.color_key][::-1]

            for keypoint in keypoints:
                if keypoints[keypoint] is not None and keypoint not in [
                    "left_eye", "right_eye", "left_ear", "right_ear", "nose"
                ]:
                    x, y = keypoints[keypoint]
                    cv2.circle(frame, (x, y), 5, color, -1)

            # if any(keypoints[name] is None for name in ("left_ear", "right_ear", "left_eye", "right_eye")):
            #     continue
            # head_width = abs(keypoints["right_ear"][0] - keypoints["left_ear"][0])
            # head_center = (
            #     (keypoints["right_eye"][0] + keypoints["left_eye"][0]) // 2,
            #     keypoints["right_eye"][1],
            # )
            # cv2.circle(frame, head_center, head_width // 2, color, -1)
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
