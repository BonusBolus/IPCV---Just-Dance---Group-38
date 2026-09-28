"""Task 3: identity tracking, first simple version.

Idea: every person wears different clothes. Measure the average colour of each person's
shirt, remember it per player, and in every new frame give each detected person the label
of the player with the most similar colour.

Pipeline per frame:
    image -> PeopleDetector.detect()   -> list of Person (box + keypoints), no identity yet
                                          (detection/people_detector.py, not part of this task)
          -> torso_box(person.box)     -> the shirt part of each box
          -> average_color()           -> one colour per person
          -> IdentityTracker.update()  -> match colours to known players -> labels
          -> draw_labels()             -> show "Player 1" / "Player 2" on screen

Functions marked TODO are yours to write. They return placeholder values for now, so the
demo already runs (every person shows up as "?").
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Player:
    id: int            # player ID placeholder
    color: np.ndarray  # remembered average shirt colour
    box: tuple         # last box where this player was seen


def torso_box(box):
    """Return the part of the person box (x, y, w, h) that contains the shirt.

    The full box also contains background, the head, arms and legs, which spoil the average
    colour. So we take a smaller box: the middle 50% of the width, and 25% to 55% of the height.
    """
    x, y, w, h = box
    return (x + w // 4, y + h // 4, w // 2, int(h * 0.3))


def average_color(image, box):
    """Return the average colour (B, G, R) of the pixels inside box, as a numpy array."""
    x, y, w, h = box
    # max(0, ...) so the box cannot start outside the image
    # (a negative index would count from the other side of the image)
    region = image[max(0, y):y + h, max(0, x):x + w]
    if region.size == 0:
        return np.zeros(3)
    # the region has shape (height, width, 3): averaging over axis 0 and 1 (all pixels)
    # leaves one value per colour channel
    return region.mean(axis=(0, 1))


def color_distance(color_a, color_b):
    """Return how different two colours are (0 = identical).

    A colour is a point in 3D space (B, G, R), so we use the Euclidean distance:
    sqrt((B1 - B2)^2 + (G1 - G2)^2 + (R1 - R2)^2).
    """
    return float(np.linalg.norm(color_a - color_b))


class IdentityTracker:
    def __init__(self, max_players=2, max_distance=80.0):
        self.max_players = max_players
        self.max_distance = max_distance  # colours further apart than this are "someone else"
        self.players = []

    def update(self, image, people):
        """Give every detected person a player. Returns a list of (player or None, person).

        `people` is the list from PeopleDetector.detect().

        Steps for every person:
          1. compute the shirt colour
          2. while there are fewer than max_players players, the person becomes a new player
          3. otherwise, the person gets the player with the most similar colour,
             but only if the colour is close enough (else None = unknown)
          4. the stored colour is updated slowly, so it follows lighting changes

        Known limitation: two people can get the same player (see docs/identity_tracking.md).
        """
        tracked = []
        for person in people:
            # 1. shirt colour of this person
            color = average_color(image, torso_box(person.box))

            # 2. not all players known yet: register a new player
            if len(self.players) < self.max_players:
                player = Player(id=len(self.players) + 1, color=color, box=person.box)
                self.players.append(player)
                tracked.append((player, person))
                continue

            # 3. find the player with the most similar colour
            best_player = None
            best_distance = self.max_distance
            for player in self.players:
                distance = color_distance(player.color, color)
                if distance < best_distance:
                    best_player = player
                    best_distance = distance

            # 4. update that player: remember where it is, and move its colour 10% towards the new one
            if best_player is not None:
                best_player.color = 0.9 * best_player.color + 0.1 * color
                best_player.box = person.box
            tracked.append((best_player, person))
        return tracked

    def reset(self):
        self.players = []


def draw_labels(image, tracked):
    """Draw the box, the shirt region and the label of every tracked person."""
    for player, person in tracked:
        x, y, w, h = person.box
        tx, ty, tw, th = torso_box(person.box)
        if player is None:
            label, color = "?", (128, 128, 128)
        else:
            label, color = f"Player {player.id}", tuple(int(c) for c in player.color)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(image, (tx, ty), (tx + tw, ty + th), (255, 255, 255), 1)
        cv2.rectangle(image, (x, y - 30), (x + 30, y), color, -1)  # colour swatch
        cv2.putText(image, label, (x + 36, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
