# -*- coding: utf-8 -*-

"""
Publication layout for the already validated
Top-3 3D U-Net over-segmentation cases.

This script does NOT change:
- model
- checkpoint
- selected volumes
- selected max-FP slices
- predictions
- metrics

It only regenerates the combined figure with a cleaner layout.
"""

from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. RUN THE ALREADY-VALIDATED V3 SCRIPT
# ============================================================

RUN_DIR = Path.cwd().resolve()

V3_SCRIPT = (
    RUN_DIR
    / "make_top3_overseg_maxfp_v3.py"
)

if not V3_SCRIPT.exists():
    raise FileNotFoundError(
        f"Cannot find V3 script:\n{V3_SCRIPT}"
    )

print("[INFO] Loading validated V3 analysis...")

v3 = runpy.run_path(
    str(V3_SCRIPT)
)

if "visual_cases" not in v3:
    raise RuntimeError(
        "V3 script finished, but visual_cases was not found."
    )

visual_cases = v3["visual_cases"]

if len(visual_cases) != 3:
    raise RuntimeError(
        f"Expected 3 validated cases, found {len(visual_cases)}."
    )

OUT_DIR = (
    RUN_DIR
    / "top3_overseg_maxfp_v3"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("[OK] Three validated cases loaded.")


# ============================================================
# 2. CLEAN DRAWING FUNCTION
# ============================================================

def draw_panel(
    axes,
    case
):

    mri = case["mri"]
    gt = case["gt"]
    pred = case["pred"]

    # MRI
    axes[0].imshow(
        mri,
        cmap="gray"
    )

    # Ground truth
    axes[1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    # Prediction
    axes[2].imshow(
        pred,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    # Overlay
    axes[3].imshow(
        mri,
        cmap="gray"
    )

    if gt.any():
        axes[3].contour(
            gt,
            levels=[0.5],
            colors="lime",
            linewidths=2.0
        )

    if pred.any():
        axes[3].contour(
            pred,
            levels=[0.5],
            colors="red",
            linewidths=1.6
        )

    for ax in axes:
        ax.axis("off")


# ============================================================
# 3. BUILD PUBLICATION-STYLE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(13.6, 10.6)
)

column_titles = [
    "MRI",
    "Ground Truth",
    "Prediction",
    "Overlay",
]


# ------------------------------------------------------------
# Draw the 12 image panels
# ------------------------------------------------------------

for row_idx, case in enumerate(
    visual_cases
):

    draw_panel(
        axes[row_idx],
        case
    )

    # Column titles only on first row
    if row_idx == 0:

        for col_idx, title in enumerate(
            column_titles
        ):

            axes[
                row_idx,
                col_idx
            ].set_title(
                title,
                fontsize=13,
                pad=8
            )


# ============================================================
# 4. COMPACT SPACING
# ============================================================

# Crucial:
# almost no blank area on the left.
fig.subplots_adjust(
    left=0.025,
    right=0.985,
    top=0.925,
    bottom=0.085,

    # four columns closer together
    wspace=0.055,

    # leave space below each row
    hspace=0.34
)


# ============================================================
# 5. CASE DESCRIPTION BELOW EACH ROW
# ============================================================

# We calculate the position after subplots_adjust,
# so every description is centered under all four images.

for row_idx, case in enumerate(
    visual_cases
):

    left_box = axes[
        row_idx,
        0
    ].get_position()

    right_box = axes[
        row_idx,
        3
    ].get_position()

    row_center = (
        left_box.x0
        + right_box.x1
    ) / 2

    text_y = (
        left_box.y0
        - 0.018
    )

    description = (
        f"Case {case['rank']}   |   "
        f"{case['patient']}   |   "
        f"index = {case['index']}   |   "
        f"max-FP slice = {case['slice_idx']}"
    )

    fig.text(
        row_center,
        text_y,
        description,
        ha="center",
        va="top",
        fontsize=10.5
    )


# ============================================================
# 6. MAIN TITLE
# ============================================================

fig.suptitle(
    "Representative Over-segmentation Cases of the Reproduced 3D U-Net",
    fontsize=15,
    y=0.982
)


# ============================================================
# 7. GLOBAL LEGEND
# ============================================================

legend_handles = [

    Line2D(
        [0],
        [0],
        color="lime",
        linewidth=2.0,
        label="Ground Truth"
    ),

    Line2D(
        [0],
        [0],
        color="red",
        linewidth=1.6,
        label="Prediction"
    ),
]


fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(
        0.5,
        0.012
    )
)


# ============================================================
# 8. SAVE
# ============================================================

save_path = (
    OUT_DIR
    / "top3_overseg_PUBLICATION_LAYOUT.png"
)

fig.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close(
    fig
)


print("\n===================================")
print("SUCCESS")
print("Publication figure saved to:")
print(save_path)
print("===================================")
