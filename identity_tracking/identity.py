"""Task 3: identity tracking with shirt colour and position.

Idea: every person wears different clothes and moves only a little between two frames.
Remember the shirt colour and last position of each player, and in every new frame give each
detected person the player that fits best.

Pipeline per frame:
    image -> PoseEstimator.process()   -> keypoints (num_people, 17, 3), no identity yet
                                          (pose_tracking/pose_main.py, task 2)
          -> people_from_pose()        -> list of Person (box + keypoints)
          -> torso_box(person)         -> the shirt part: between the shoulders and hips
          -> average_color()           -> one colour per person
             color_histogram()         -> colour distribution, for players with similar colours
          -> IdentityTracker.update()  -> cost for every person-player pair, best pairing
          -> draw_labels()             -> show "Player 1" / "Player 2" on screen
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# COCO keypoint order used by YOLO-pose (same as PoseEstimator.KEYPOINTS)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


@dataclass
class Person:
    box: tuple              # (x, y, w, h) in pixels, around the visible keypoints
    keypoints: np.ndarray   # (17, 3) x, y, confidence: one row of the PoseEstimator output


@dataclass
class Player:
    id: int                 # player ID placeholder
    color: np.ndarray       # remembered average shirt colour
    histogram: np.ndarray   # remembered colour distribution of the shirt
    box: tuple              # last box where this player was seen


def people_from_pose(keypoints, min_confidence=0.1, margin=0.08):
    """Turn the PoseEstimator output (num_people, 17, 3) into a list of Person.

    The box of a person is the smallest box around their keypoints that the model is sure
    about, made `margin` (8%) larger on every side because the keypoints sit inside the body.
    People with fewer than 2 such keypoints are skipped: no box can be made for them.
    """
    people = []
    for person_keypoints in keypoints:
        visible = person_keypoints[person_keypoints[:, 2] >= min_confidence, :2]
        if len(visible) < 2:
            continue
        x1, y1 = visible.min(axis=0)
        x2, y2 = visible.max(axis=0)
        dx, dy = margin * (x2 - x1), margin * (y2 - y1)
        box = (int(x1 - dx), int(y1 - dy), int(x2 - x1 + 2 * dx), int(y2 - y1 + 2 * dy))
        people.append(Person(box, person_keypoints))
    return people


def torso_box(person, min_confidence=0.5, torso_ratio=1.3):
    """Return the part of the person (x, y, w, h) that contains the shirt.

    The full box also contains background, the head, arms and legs, which spoil the average
    colour. With the pose keypoints we know where the shirt is: between the shoulders and the
    hips. We take the middle 90% of that width, from the shoulder line down to 90% of that
    height, so the arms and the trousers stay out.

    If the hips are not visible (e.g. standing close to the camera, hips below the image), the
    hip height is estimated from the shoulders: for most people the distance from shoulders to
    hips is about 1.3 x the shoulder width (`torso_ratio`). Whatever falls below the image is
    cut off by cut_region.

    If a shoulder is not visible either, fall back to a fixed part of the person box: the middle
    50% of the width, and 25% to 55% of the height.
    """
    kp = person.keypoints
    shoulders = [LEFT_SHOULDER, RIGHT_SHOULDER]
    hips = [i for i in (LEFT_HIP, RIGHT_HIP) if kp[i, 2] >= min_confidence]
    if all(kp[i, 2] >= min_confidence for i in shoulders):
        x1, x2 = kp[shoulders + hips, 0].min(), kp[shoulders + hips, 0].max()
        top = kp[shoulders, 1].mean()
        if hips:
            bottom = kp[hips, 1].mean()
        else:
            bottom = top + torso_ratio * abs(kp[LEFT_SHOULDER, 0] - kp[RIGHT_SHOULDER, 0])
        w, h = x2 - x1, bottom - top
        if w > 4 and h > 4:  # a person standing sideways has almost no torso width
            return (int(x1 + 0.05 * w), int(top), int(0.9 * w), int(0.9 * h))
    x, y, w, h = person.box
    return (x + w // 4, y + h // 4, w // 2, int(h * 0.3))


def cut_region(image, box):
    """The pixels inside box. max(0, ...) so the box cannot start outside the image
    (a negative index would count from the other side of the image)."""
    x, y, w, h = box
    return image[max(0, y):y + h, max(0, x):x + w]


def average_color(image, box):
    """Return the average colour (B, G, R) of the pixels inside box, as a numpy array."""
    region = cut_region(image, box)
    if region.size == 0:
        return np.zeros(3)
    # the region has shape (height, width, 3): averaging over axis 0 and 1 (all pixels)
    # leaves one value per colour channel
    return region.mean(axis=(0, 1))


def color_histogram(image, box, bins=8):
    """Return the colour distribution of the pixels inside box.

    The average colour loses information: a black-and-white striped shirt and a grey shirt
    have the same average. A histogram counts how many pixels fall in each colour range:
    every channel is split into `bins` ranges, giving 8 x 8 x 8 = 512 colour "boxes".
    It is normalized (sums to 1), so the size of the region does not matter.
    """
    region = cut_region(image, box)
    if region.size == 0:
        return np.zeros(bins ** 3, np.float32)
    hist = cv2.calcHist([region], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, None, 1.0, 0.0, cv2.NORM_L1).flatten()


def color_distance(color_a, color_b):
    """Return how different two colours are (0 = identical).

    A colour is a point in 3D space (B, G, R), so we use the Euclidean distance:
    sqrt((B1 - B2)^2 + (G1 - G2)^2 + (R1 - R2)^2).
    """
    return float(np.linalg.norm(color_a - color_b))


def histogram_distance(hist_a, hist_b):
    """Return how different two histograms are: 0 = identical, 1 = no overlap at all
    (Bhattacharyya distance)."""
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def position_distance(box_a, box_b):
    """Distance in pixels between the centres of two boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return float(np.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2)))


class IdentityTracker:
    def __init__(self, max_players=2, max_distance=60.0, position_weight=0.1,
                 similar_colors=30.0, histogram_weight=100.0):
        self.max_players = max_players
        self.max_distance = max_distance          # colours further apart than this are "someone else"
        self.position_weight = position_weight    # 0.1: moving 100 px costs as much as 10 colour units
        self.similar_colors = similar_colors      # players closer in colour than this: also use histograms
        self.histogram_weight = histogram_weight  # completely different histograms cost 100 colour units
        self.players = []

    def colors_are_similar(self):
        """True if two players have almost the same average shirt colour. Then the average
        alone cannot tell them apart, and the histogram is added to the cost."""
        for i, a in enumerate(self.players):
            for b in self.players[i + 1:]:
                if color_distance(a.color, b.color) < self.similar_colors:
                    return True
        return False

    def cost(self, player, color, histogram, box, use_histogram):
        """How badly a person fits a player: lower is better.

        colour difference + weight * how far the person is from where the player was last seen,
        + weight * histogram difference when the players' colours are similar.
        """
        cost = color_distance(player.color, color)
        cost += self.position_weight * position_distance(player.box, box)
        if use_histogram:
            cost += self.histogram_weight * histogram_distance(player.histogram, histogram)
        return cost

    def update(self, image, people):
        """Give every detected person a player. Returns a list of (player or None, person).

        `people` is the list from people_from_pose().

        Steps:
          1. compute the shirt colour and histogram of every person
          2. compute the cost of every person-player pair (a table: people x players)
          3. find the pairing with the lowest total cost, where every player is given to at
             most one person (Hungarian algorithm, scipy's linear_sum_assignment)
          4. a pair is only accepted if the colours are close enough (max_distance);
             otherwise the person is someone else ("?")
          5. people without a player become new players while there is room
          6. matched players are updated: new position, colour moved 10% towards the new one
        """
        # 1. shirt colour and histogram of every person
        colors = [average_color(image, torso_box(p)) for p in people]
        histograms = [color_histogram(image, torso_box(p)) for p in people]

        assigned = [None] * len(people)
        if self.players and people:
            # 2. cost table: row = person, column = player
            use_histogram = self.colors_are_similar()
            costs = np.zeros((len(people), len(self.players)))
            for i, person in enumerate(people):
                for j, player in enumerate(self.players):
                    costs[i, j] = self.cost(player, colors[i], histograms[i], person.box, use_histogram)

            # 3. best pairing overall: each person at most one player, each player at most one person
            rows, cols = linear_sum_assignment(costs)

            # 4. only accept pairs whose colour is close enough
            for i, j in zip(rows, cols):
                if color_distance(self.players[j].color, colors[i]) < self.max_distance:
                    assigned[i] = self.players[j]

        tracked = {}

        for i, person in enumerate(people):
            player = assigned[i]

            if player is None and len(self.players) < self.max_players:
                player = Player(
                    id=len(self.players) + 1,
                    color=colors[i],
                    histogram=histograms[i],
                    box=person.box
                )
                self.players.append(player)

            elif player is not None:
                player.color = 0.9 * player.color + 0.1 * colors[i]
                player.histogram = 0.9 * player.histogram + 0.1 * histograms[i]
                player.box = person.box

            if player is not None:
                tracked[player.id] = person.keypoints

        return tracked

    def reset(self):
        self.players = []


def draw_labels(image, tracked):
    """Draw the box, the shirt region and the label of every tracked person."""
    for player, person in tracked:
        x, y, w, h = person.box
        tx, ty, tw, th = torso_box(person)
        if player is None:
            label, color = "?", (128, 128, 128)
        else:
            label, color = f"Player {player.id}", tuple(int(c) for c in player.color)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(image, (tx, ty), (tx + tw, ty + th), (255, 255, 255), 1)
        cv2.rectangle(image, (x, y - 30), (x + 30, y), color, -1)  # colour swatch
        cv2.putText(image, label, (x + 36, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
