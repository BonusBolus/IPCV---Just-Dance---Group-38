"""PNG of the pipeline overview (the same diagram as pipeline_overview.tex), without LaTeX.

Same boxes, sizes (cm), arrows and labels as the TikZ code; Computer Modern, the LaTeX font.
Transparent: no background and the boxes are not filled (only black lines and text).
Run:  .venv/bin/python docs/figures/make_pipeline_overview_png.py   -> docs/figures/pipeline_overview.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"              # bold parts as $\mathbf{...}$ in Computer Modern bold
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "pipeline_overview.png"
CM = 1 / 2.54
REGULAR, BOLD, ITALIC = "cmr10", "cmb10", "cmti10"   # Computer Modern (comes with matplotlib)
TEXT_PT, LABEL_PT = 8, 7                             # \footnotesize and \scriptsize in a 10 pt document

# the TikZ sizes: text width 3.2 cm + inner sep 2 x 4 pt; node distance 9 mm (vertical) and 6 mm (horizontal)
BOX_W, WIDE_W, GAP_X, GAP_Y = 3.48, 11.6, 0.6, 0.9
TWO_LINES, ONE_LINE, WIDE_H = 1.1, 1.0, 0.8


def layout():
    """Box name -> (centre x, centre y, width, height) in cm, built like TikZ's 'below=of' / 'left=of'."""
    boxes = {"pose": (0.0, 0.0, BOX_W, TWO_LINES)}
    boxes["face"] = (-(BOX_W + GAP_X), 0.0, BOX_W, TWO_LINES)
    boxes["seg"] = (BOX_W + GAP_X, 0.0, BOX_W, TWO_LINES)
    boxes["cam"] = (0.0, TWO_LINES / 2 + GAP_Y + WIDE_H / 2, WIDE_W, WIDE_H)
    y = -TWO_LINES / 2 - GAP_Y - WIDE_H / 2
    boxes["id"] = (0.0, y, WIDE_W, WIDE_H)
    bottom = y - WIDE_H / 2
    for name in ("logic", "render", "screen"):
        y = bottom - GAP_Y - ONE_LINE / 2
        boxes[name] = (0.0, y, BOX_W, ONE_LINE)
        bottom = y - ONE_LINE / 2
    return boxes


TEXTS = {
    "pose": [("T2 Pose estimation", BOLD), ("YOLO26n-pose", REGULAR)],
    "face": [("T1 Face detection", BOLD), ("YuNet", REGULAR)],
    "seg": [("Segmentation", BOLD), ("YOLO26n-seg", REGULAR)],
    "cam": [[("Webcam", BOLD), (": one frame", REGULAR)]],
    "id": [[("T3 Identity tracking", BOLD), (": Player 1 and Player 2", REGULAR)]],
    "logic": [("T4 Interaction logic", BOLD)],
    "render": [("T5 Scene rendering", BOLD)],
    "screen": [("Screen", REGULAR)],
}
ARROWS = [("cam", "face", None), ("cam", "pose", None), ("cam", "seg", None),
          ("face", "id", "faces"), ("pose", "id", "keypoints"), ("seg", "id", "masks"),
          ("id", "logic", "players"), ("logic", "render", "players, events"), ("render", "screen", None)]


def draw():
    boxes = layout()
    left = min(x - w / 2 for x, y, w, h in boxes.values()) - 0.3
    right = max(x + w / 2 for x, y, w, h in boxes.values()) + 0.3
    top = max(y + h / 2 for x, y, w, h in boxes.values()) + 0.3
    low = min(y - h / 2 for x, y, w, h in boxes.values()) - 0.3
    fig = plt.figure(figsize=((right - left) * CM, (top - low) * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(left, right)
    ax.set_ylim(low, top)
    ax.set_aspect("equal")
    ax.axis("off")

    for name, (x, y, w, h) in boxes.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor="none", edgecolor="black", linewidth=0.6))
        lines = TEXTS[name]
        line_h = 0.36
        for k, line in enumerate(lines):
            ty = y + (len(lines) - 1) * line_h / 2 - k * line_h
            parts = line if isinstance(line, list) else [line]
            # one text per line: a bold part as a bold formula ($\mathbf{...}$, Computer Modern bold) and the
            # rest as normal text, so matplotlib spaces "Webcam: one frame" exactly like LaTeX
            text = "".join("$\\mathbf{" + part.replace(" ", "\\ ") + "}$" if font == BOLD else part
                           for part, font in parts)
            ax.text(x, ty, text, fontfamily=REGULAR, fontsize=TEXT_PT, va="center", ha="center")

    for a, b, label in ARROWS:
        ax_, ay, aw, ah = boxes[a]
        bx, by, bw, bh = boxes[b]
        x = bx if a == "cam" else ax_          # from the webcam bar straight down to each detector
        y1, y2 = ay - ah / 2, by + bh / 2
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                     mutation_scale=6, linewidth=0.6, color="black", shrinkA=0, shrinkB=0))
        if label:
            ax.text(x + 0.12, (y1 + y2) / 2, label, fontfamily=ITALIC, fontsize=LABEL_PT, va="center", ha="left")
    fig.savefig(OUT, dpi=400, transparent=True)          # no background: only lines, arrows and text
    print("saved", OUT)


if __name__ == "__main__":
    draw()
