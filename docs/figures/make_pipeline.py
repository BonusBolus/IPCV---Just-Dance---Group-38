"""Draws the system pipeline figure for the report: docs/figures/pipeline.pdf (and a .png preview).

Run:  .venv/bin/python docs/figures/make_pipeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "pipeline"
W, H = 7.2, 6.4                     # inches (about 18 x 16 cm)

# black and white: every box white with a black edge (the names below only tell the groups apart in the code)
GREY = BLUE = ORANGE = GREEN = RED = "white"
EDGE = {"white": "#222222"}
TEXT = "#222222"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(cx, cy, w, h, color, title=None, lines=(), title_size=8.5, radius=0.06, z=1):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=color, edgecolor=EDGE.get(color, "#888888"), linewidth=1.0, zorder=z))
    text = ([title] if title else []) + list(lines)
    if not text:
        return
    if title:
        offset = 0.075 * len(lines)
        ax.text(cx, cy + offset, title, ha="center", va="center", fontsize=title_size, fontweight="bold",
                color=TEXT, zorder=z + 1)
        if lines:
            ax.text(cx, cy + offset - 0.13 - 0.075 * (len(lines) - 1), "\n".join(lines), ha="center",
                    va="center", fontsize=7, color="#444444", linespacing=1.25, zorder=z + 1)
    else:
        ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=7, color=TEXT, linespacing=1.15,
                zorder=z + 1)


def arrow(x1, y1, x2, y2, label=None, label_side="right", connection="arc3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                                 color="#555555", connectionstyle=connection, shrinkA=0, shrinkB=0, zorder=3))
    if label:
        dx = 0.06 if label_side == "right" else -0.06
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=6.8, color="#333333", style="italic", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


# --- row A: camera ---------------------------------------------------------------------------
box(1.25, 6.0, 2.05, 0.5, GREY, "Webcam", ["1920×1080, MJPEG"])
box(3.85, 6.0, 2.75, 0.5, GREY, "Capture thread", ["mirror · 1280×720 copy · newest frame only"])
arrow(2.275, 6.0, 2.475, 6.0)

# --- row B: three detectors, at the same time -------------------------------------------------
xs = (1.25, 3.6, 5.95)
box(xs[0], 4.85, 2.05, 0.72, BLUE, "T1  Face detection", ["YuNet · 640 px copy", "own thread"])
box(xs[1], 4.85, 2.05, 0.72, BLUE, "T2  Pose estimation", ["YOLO26n-pose · OpenVINO", "main thread"])
box(xs[2], 4.85, 2.05, 0.72, BLUE, "Person segmentation", ["YOLO26n-seg · 480 px copy", "own thread"])
# one line from the capture thread that splits to the three detectors
bus_y = 5.47
ax.plot([3.85, 3.85], [5.75, bus_y], color="#555555", linewidth=1.0, zorder=3)
ax.plot([xs[0], xs[2]], [bus_y, bus_y], color="#555555", linewidth=1.0, zorder=3)
for x in xs:
    arrow(x, bus_y, x, 5.21)
ax.text(3.91, 5.62, "frame", fontsize=6.8, style="italic", color="#333333", ha="left", va="center")

# --- row C: identity ----------------------------------------------------------------------------
box(3.6, 3.68, 6.6, 0.66, ORANGE, "T3  Identity tracking",
    ["shirt colour + position  →  Hungarian matching  →  Player 1 / Player 2",
     "faces and masks are given to the player they belong to (bystanders are removed)"])
arrow(xs[0], 4.49, xs[0], 4.01, "faces")
arrow(xs[1], 4.49, xs[1], 4.01, "keypoints  N×17×3")
arrow(xs[2], 4.49, xs[2], 4.01, "masks")

# --- row D: interaction logic with the state machine --------------------------------------------
t4x, t4y, t4w, t4h = 3.3, 2.3, 4.25, 1.38
box(t4x, t4y, t4w, t4h, GREEN)
ax.text(t4x, t4y + 0.52, "T4  Interaction logic", ha="center", va="center", fontsize=8.5, fontweight="bold",
        color=TEXT, zorder=5)
states = ["Song\nselect", "Regis-\ntration", "Count-\ndown", "Playing", "Results"]
chip_w, gap = 0.7, 0.13
start = t4x - (5 * chip_w + 4 * gap) / 2 + chip_w / 2
chip_y = t4y + 0.08
for k, name in enumerate(states):
    cx = start + k * (chip_w + gap)
    box(cx, chip_y, chip_w, 0.42, "white", lines=[name], radius=0.05, z=5)
    if k < 4:
        arrow(cx + chip_w / 2, chip_y, cx + chip_w / 2 + gap, chip_y)
arrow(start + 4 * (chip_w + gap), chip_y - 0.21, start, chip_y - 0.21,
      connection="arc3,rad=-0.18")
ax.text(t4x, t4y - 0.5, "grading per move · duets · bonus star · hands-up gestures", ha="center", va="center",
        fontsize=7, color="#444444", zorder=5)

box(6.25, 2.3, 1.3, 0.62, GREY, "Song file", ["moves on beats", "+ music"])
arrow(5.6, 2.3, t4x + t4w / 2, 2.3)

arrow(t4x, 3.35, t4x, t4y + t4h / 2,
      "players: id, visible, box,\n17 named keypoints, colours, score")

# --- row E: rendering -------------------------------------------------------------------------------
box(3.6, 0.96, 6.6, 0.84, RED)
ax.text(0.45, 1.24, "T5  Scene rendering", ha="left", va="center", fontsize=8.5, fontweight="bold", color=TEXT,
        zorder=5)
layers = ["Stage\n(room replaced)", "Players\n(mask)", "Big heads\n(from 1080p)", "Crown +\nlabels",
          "Move cards, timeline,\nscores, ratings"]
widths = [1.1, 0.95, 1.1, 0.95, 1.55]
x = 0.45
for k, (name, w) in enumerate(zip(layers, widths)):
    box(x + w / 2, 0.86, w, 0.44, "white", lines=[name], radius=0.05, z=5)
    if k < 4:
        arrow(x + w, 0.86, x + w + 0.16, 0.86)
    x += w + 0.16

arrow(0.62, 3.35, 0.62, 1.38)
ax.text(0.68, 3.17, "faces, player mask", fontsize=6.8, style="italic", color="#333333", ha="left", va="center",
        zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
arrow(t4x, t4y - t4h / 2, t4x, 1.38, "state, ratings, points, star")

# --- row F: output ----------------------------------------------------------------------------------
box(3.6, 0.16, 2.2, 0.28, GREY, lines=["Screen + speakers"])
arrow(3.6, 0.54, 3.6, 0.30)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT.with_suffix(".pdf"))
