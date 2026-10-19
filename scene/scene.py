import math
import cv2
import numpy as np
from scene.functions import RATING_COLORS, draw_text, put_text_center, put_text_left, put_text_right, scale_for_height
from scene.pose_figure import BONES

POP_TIME = 0.18    # seconds: a new rating starts big and shrinks to its normal size ("pop")
POP_SIZE = 1.6     # how big a rating starts (1.6 = 60% bigger)
FADE_TIME = 0.3    # seconds at the end: the rating fades out
SIDE_SWAP_MARGIN = 40   # pixels: the score sides only swap when the players clearly stand the other way round
RATING_GAP = 70    # pixels between the score bar and a rating (room for the score number)


def pop(age, total_time):
    """Size factor and opacity of a rating that is `age` seconds old and stays `total_time` seconds."""
    size = 1.0
    if age < POP_TIME:
        rest = 1.0 - age / POP_TIME
        size = 1.0 + (POP_SIZE - 1.0) * rest * rest          # fast at the start, slow at the end
    alpha = min(1.0, max(0.0, (total_time - age) / FADE_TIME))
    return size, alpha


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
        self.sides = None   # player ids from left to right on the screen (score bar and ratings)

    def update_sides(self, players):
        """Which player stands left and which right, for the score bar and the ratings: player ids
        from left to right. Uses the middle of each player's box. A player that is not visible keeps
        their side, and the sides only swap when the players clearly stand the other way round
        (SIDE_SWAP_MARGIN), so the bar does not flip back and forth. At the start: Player 1 left."""
        ids = [player["id"] for player in players]
        if self.sides is None or sorted(self.sides) != sorted(ids):
            self.sides = ids
        xs = {player["id"]: player["box"][0] + player["box"][2] / 2
              for player in players if player["visible"] and player["box"] is not None}
        if len(xs) == 2:
            left, right = self.sides
            if xs[left] - xs[right] > SIDE_SWAP_MARGIN:
                self.sides = [right, left]
        return self.sides

    def render(self, frame, players, current_pose=None):
        output = frame.copy()
        colors = []
        scores = []

        # the score bar follows where the players stand: the player on the left has the left part
        by_id = {player["id"]: player for player in players}
        for player_id in self.update_sides(players):
            colors.append(by_id[player_id][self.color_key])
            scores.append(by_id[player_id]["score"])

        output = self.draw_title(output, players, text_height=self.title_height)
        output = self._add_score(output, scores, self.score_top, self.score_bottom, colors)

        if current_pose is not None:
            output = self._add_current_pose(output, self.pose_top, self.pose_bottom, current_pose)

        return output

    def draw_title(self, frame, players, prompt=None, text_height=60):
        """"Just Dance" at the top in the middle.
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
        edge:    width of the glow, as a fraction of the screen width"""
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

    def draw_ratings(self, frame, ratings, ages=None, total_time=1.0):
        """"Perfect", "Good", ... right next to the score bar, on the side of that player's score:
        the left player's rating left of the bar, the right player's rating right of it (see
        update_sides(); render() must be called first in the same frame).
        ratings: {player_id: "Perfect"}
        ages:    {player_id: seconds since the rating came}: a new rating pops (see pop())."""
        width = frame.shape[1]
        bar_width = width // 4
        bar_left, bar_right = width // 2 - bar_width // 2, width // 2 + bar_width // 2
        y = (self.score_top + self.score_bottom) // 2
        left_id = (self.sides or sorted(ratings))[0]
        for player_id, rating in ratings.items():
            size, alpha = pop(ages[player_id], total_time) if ages and player_id in ages else (1.0, 1.0)
            text_w = cv2.getTextSize(rating, cv2.FONT_HERSHEY_SIMPLEX, size, 5)[0][0]
            if player_id == left_id:
                x = max(bar_left - RATING_GAP - text_w // 2, text_w // 2 + 5)
            else:
                x = min(bar_right + RATING_GAP + text_w // 2, width - text_w // 2 - 5)
            self._draw_fading_text(frame, rating, (x, y), size, RATING_COLORS[rating], alpha)
        return frame

    def _draw_fading_text(self, frame, text, center, size, color, alpha):
        """draw_text(), but see-through when alpha < 1 (only the part around the text is blended)."""
        if alpha >= 1.0:
            draw_text(frame, text, center, size, color)
            return
        if alpha <= 0.0:
            return
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 5)
        height, width = frame.shape[:2]
        x0, y0 = max(0, center[0] - text_w // 2 - 8), max(0, center[1] - text_h - 8)
        x1, y1 = min(width, center[0] + text_w // 2 + 8), min(height, center[1] + text_h + 8)
        region = frame[y0:y1, x0:x1]
        layer = region.copy()
        draw_text(layer, text, (center[0] - x0, center[1] - y0), size, color)
        region[:] = cv2.addWeighted(layer, alpha, region, 1.0 - alpha, 0)

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
