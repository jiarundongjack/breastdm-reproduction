# -*- coding: utf-8 -*-

"""
Select the representative over-segmentation slice
inside the already-selected 3D volume.

Selected volume:
    index   = 55
    patient = BreaDM-Ma-1821

Volume-level selection has already been completed.

Slice-level logic
=================
Eligibility:
    GTArea_s > 0
    TP_s > 0
    PredArea_s > GTArea_s

Among eligible slices:
    select the slice with maximum FP_s

Tie-breakers:
    1. larger PredArea - GTArea
    2. larger PredArea / GTArea
    3. larger TP

Important:
Slice-level analysis is ONLY used for visualization.
It does NOT redefine the formal 3D DSC, mIoU, or PPV.
"""
from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

RUN_DIR = RUN3

TARGET_INDEX = 55
TARGET_PATIENT = "BreaDM-Ma-1821"

V3_SCRIPT = RUNTIME / "make_top3_overseg_maxfp_v3.py"
CSV_PATH = RUN_DIR / "test_predictions.csv"

OUT_DIR = OUTPUT / "volume55_slice_overseg_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


if not V3_SCRIPT.exists():
    raise FileNotFoundError(
        f"Cannot find validated V3 script:\n{V3_SCRIPT}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find test_predictions.csv:\n{CSV_PATH}"
    )


# ============================================================
# 2. VERIFY THE SELECTED VOLUME
# ============================================================

df = pd.read_csv(CSV_PATH)

row = df[
    df["index"].astype(int) == TARGET_INDEX
]

if len(row) != 1:
    raise RuntimeError(
        f"Expected exactly one row for index={TARGET_INDEX}, "
        f"but found {len(row)}."
    )

row = row.iloc[0]

patient = str(row["patient"])

if patient != TARGET_PATIENT:
    raise RuntimeError(
        f"Patient mismatch.\n"
        f"Expected: {TARGET_PATIENT}\n"
        f"Found   : {patient}"
    )

print("\n======================================================")
print("SELECTED VOLUME")
print("======================================================")
print(f"index   : {TARGET_INDEX}")
print(f"patient : {patient}")
print(f"GT      : {int(row['gt_positive_pixels'])}")
print(f"Pred    : {int(row['pred_positive_pixels'])}")
print(f"PPV     : {float(row['ppv']):.6f}")
print("======================================================\n")


# ============================================================
# 3. LOAD THE ALREADY-VALIDATED V3 ENVIRONMENT
# ============================================================

print("[INFO] Loading validated V3 model and dataset...")

v3 = runpy.run_path(
    str(V3_SCRIPT)
)


# ============================================================
# 4. FIND MODEL
# ============================================================

model = v3.get("model", None)

if not isinstance(model, torch.nn.Module):

    model = None

    for name, obj in v3.items():

        if isinstance(obj, torch.nn.Module):

            model = obj

            print(
                f"[INFO] Model detected from V3 namespace: {name}"
            )

            break


if model is None:
    raise RuntimeError(
        "Could not find the loaded PyTorch model "
        "inside the validated V3 script."
    )


# ============================================================
# 5. FIND TEST DATASET
# ============================================================

test_dataset = v3.get(
    "test_dataset",
    None
)

if test_dataset is None:

    for name, obj in v3.items():

        try:

            if (
                hasattr(obj, "__getitem__")
                and hasattr(obj, "__len__")
                and len(obj) == len(df)
                and not isinstance(
                    obj,
                    (
                        pd.DataFrame,
                        pd.Series,
                        list,
                        tuple,
                        dict,
                        str,
                    ),
                )
            ):

                test_dataset = obj

                print(
                    f"[INFO] Test dataset detected "
                    f"from V3 namespace: {name}"
                )

                break

        except Exception:
            pass


if test_dataset is None:
    raise RuntimeError(
        "Could not find test_dataset inside the "
        "validated V3 environment."
    )


print(
    f"[OK] Test dataset contains "
    f"{len(test_dataset)} volumes."
)


# ============================================================
# 6. DEVICE
# ============================================================

try:
    device = next(
        model.parameters()
    ).device

except StopIteration:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


model.eval()

print(
    f"[INFO] Model device: {device}"
)


# ============================================================
# 7. LOAD VOLUME index=55
# ============================================================

sample = test_dataset[
    TARGET_INDEX
]


if isinstance(
    sample,
    (tuple, list)
):

    if len(sample) < 2:
        raise RuntimeError(
            "Dataset sample does not contain both "
            "image and target."
        )

    image = sample[0]
    target = sample[1]


elif isinstance(
    sample,
    dict
):

    image = sample.get(
        "image",
        None
    )

    target = sample.get(
        "target",
        sample.get(
            "mask",
            None
        )
    )


else:

    raise RuntimeError(
        "Unsupported dataset sample format."
    )


if image is None or target is None:
    raise RuntimeError(
        "Could not extract image and target "
        "from dataset sample."
    )


if not torch.is_tensor(image):
    image = torch.as_tensor(image)

if not torch.is_tensor(target):
    target = torch.as_tensor(target)


image_cpu = image.detach().cpu()
target_cpu = target.detach().cpu()


# Remove possible singleton target channel
while (
    target_cpu.ndim > 3
    and target_cpu.shape[0] == 1
):
    target_cpu = target_cpu.squeeze(0)


# ============================================================
# 8. RE-INFER PREDICTION USING FORMAL BEST MODEL
# ============================================================

if image.ndim == 4:

    # Expected:
    # [C, D, H, W]
    model_input = image.unsqueeze(0)

elif image.ndim == 3:

    # Fallback:
    # [D, H, W] -> [1, 1, D, H, W]
    model_input = (
        image
        .unsqueeze(0)
        .unsqueeze(0)
    )

else:

    raise RuntimeError(
        f"Unexpected image shape: {tuple(image.shape)}"
    )


model_input = model_input.to(
    device
)


with torch.no_grad():

    output = model(
        model_input
    )

    if isinstance(
        output,
        dict
    ):

        output = output["out"]

    pred = (
        output
        .argmax(dim=1)[0]
        .detach()
        .cpu()
    )


if pred.ndim != 3:
    raise RuntimeError(
        f"Unexpected prediction shape: "
        f"{tuple(pred.shape)}"
    )


if target_cpu.ndim != 3:
    raise RuntimeError(
        f"Unexpected target shape: "
        f"{tuple(target_cpu.shape)}"
    )


if pred.shape != target_cpu.shape:
    raise RuntimeError(
        f"Prediction-target shape mismatch:\n"
        f"pred   = {tuple(pred.shape)}\n"
        f"target = {tuple(target_cpu.shape)}"
    )


# ============================================================
# 9. VERIFY VOLUME-LEVEL COUNTS
# ============================================================

valid = (
    target_cpu != 255
)

gt_volume = (
    (target_cpu == 1)
    & valid
)

pred_volume = (
    (pred == 1)
    & valid
)


volume_tp = int(
    (
        pred_volume
        & gt_volume
    ).sum().item()
)

volume_fp = int(
    (
        pred_volume
        & (~gt_volume)
    ).sum().item()
)

volume_fn = int(
    (
        (~pred_volume)
        & gt_volume
    ).sum().item()
)

volume_gt = int(
    gt_volume.sum().item()
)

volume_pred = int(
    pred_volume.sum().item()
)

volume_coverage = (
    volume_tp / volume_gt
    if volume_gt > 0
    else 0.0
)

volume_ppv = (
    volume_tp / volume_pred
    if volume_pred > 0
    else 0.0
)


print("\n======================================================")
print("RE-INFERRED VOLUME CHECK")
print("======================================================")

print(
    f"GT             : {volume_gt}"
)

print(
    f"Pred           : {volume_pred}"
)

print(
    f"TP             : {volume_tp}"
)

print(
    f"FP             : {volume_fp}"
)

print(
    f"FN             : {volume_fn}"
)

print(
    f"TP / GT        : {volume_coverage:.6f}"
)

print(
    f"PPV            : {volume_ppv:.6f}"
)

print(
    f"Pred / GT      : "
    f"{volume_pred / volume_gt:.6f}"
)

print("======================================================\n")


# Strict consistency checks with CSV
csv_gt = int(
    row["gt_positive_pixels"]
)

csv_pred = int(
    row["pred_positive_pixels"]
)


if volume_gt != csv_gt:
    raise RuntimeError(
        f"GT count mismatch: "
        f"re-inference={volume_gt}, CSV={csv_gt}"
    )

if volume_pred != csv_pred:
    raise RuntimeError(
        f"Pred count mismatch: "
        f"re-inference={volume_pred}, CSV={csv_pred}"
    )


print(
    "[OK] Re-inference matches the saved "
    "formal volume-level counts."
)


# ============================================================
# 10. SLICE-LEVEL AUDIT
# ============================================================

records = []

depth = target_cpu.shape[0]


for s in range(depth):

    gt_s = (
        (target_cpu[s] == 1)
        & (target_cpu[s] != 255)
    )

    pred_s = (
        (pred[s] == 1)
        & (target_cpu[s] != 255)
    )

    gt_area = int(
        gt_s.sum().item()
    )

    pred_area = int(
        pred_s.sum().item()
    )

    tp = int(
        (
            pred_s
            & gt_s
        ).sum().item()
    )

    fp = int(
        (
            pred_s
            & (~gt_s)
        ).sum().item()
    )

    fn = int(
        (
            (~pred_s)
            & gt_s
        ).sum().item()
    )

    coverage = (
        tp / gt_area
        if gt_area > 0
        else np.nan
    )

    ppv = (
        tp / pred_area
        if pred_area > 0
        else np.nan
    )

    pred_gt = (
        pred_area / gt_area
        if gt_area > 0
        else np.nan
    )

    extra_area = (
        pred_area - gt_area
    )

    eligible = (
        gt_area > 0
        and tp > 0
        and pred_area > gt_area
    )

    records.append(
        {
            "slice": s,
            "GTArea": gt_area,
            "PredArea": pred_area,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "GT_Coverage": coverage,
            "PPV": ppv,
            "Pred_GT": pred_gt,
            "ExtraArea": extra_area,
            "Eligible": eligible,
        }
    )


slice_df = pd.DataFrame(
    records
)


slice_df.to_csv(
    OUT_DIR
    / "all_slices_volume55.csv",
    index=False
)


# ============================================================
# 11. APPLY THE AGREED SLICE CONDITIONS
# ============================================================

eligible_df = slice_df[
    slice_df["Eligible"]
].copy()


if len(eligible_df) == 0:

    raise RuntimeError(
        "No slice satisfies:\n"
        "GTArea > 0, TP > 0, PredArea > GTArea."
    )


# ============================================================
# 12. SELECT THE REPRESENTATIVE SLICE
# ============================================================

# Main criterion:
#     maximum FP
#
# Tie-breakers:
#     larger excess area
#     larger Pred/GT
#     larger TP

eligible_df = eligible_df.sort_values(
    by=[
        "FP",
        "ExtraArea",
        "Pred_GT",
        "TP",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ],
).reset_index(
    drop=True
)


eligible_df.insert(
    0,
    "SliceRank",
    range(
        1,
        len(eligible_df) + 1
    )
)


eligible_df.to_csv(
    OUT_DIR
    / "eligible_slices_ranked.csv",
    index=False
)


best = eligible_df.iloc[0]

slice_idx = int(
    best["slice"]
)


pd.DataFrame(
    [best]
).to_csv(
    OUT_DIR
    / "SELECTED_slice.csv",
    index=False
)


# ============================================================
# 13. PRINT SELECTED SLICE
# ============================================================

print("\n======================================================")
print("SELECTED REPRESENTATIVE SLICE")
print("======================================================")

print(
    f"Volume index      : {TARGET_INDEX}"
)

print(
    f"Patient           : {TARGET_PATIENT}"
)

print(
    f"Selected slice    : {slice_idx}"
)

print(
    f"GT area           : "
    f"{int(best['GTArea'])}"
)

print(
    f"Pred area         : "
    f"{int(best['PredArea'])}"
)

print(
    f"TP                : "
    f"{int(best['TP'])}"
)

print(
    f"FP                : "
    f"{int(best['FP'])}"
)

print(
    f"FN                : "
    f"{int(best['FN'])}"
)

print(
    f"GT coverage       : "
    f"{best['GT_Coverage']:.6f}"
)

print(
    f"PPV               : "
    f"{best['PPV']:.6f}"
)

print(
    f"Pred / GT         : "
    f"{best['Pred_GT']:.6f}"
)

print("======================================================\n")


# ============================================================
# 14. EXTRACT SELECTED SLICE FOR VISUALIZATION
# ============================================================

if image_cpu.ndim == 4:

    # [C, D, H, W]
    mri = (
        image_cpu[
            0,
            slice_idx
        ]
        .numpy()
    )

elif image_cpu.ndim == 3:

    mri = (
        image_cpu[
            slice_idx
        ]
        .numpy()
    )

else:

    raise RuntimeError(
        f"Unexpected image shape for display: "
        f"{tuple(image_cpu.shape)}"
    )


gt_show = (
    target_cpu[
        slice_idx
    ].numpy()
    == 1
)

pred_show = (
    pred[
        slice_idx
    ].numpy()
    == 1
)


# ============================================================
# 15. CREATE FOUR-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(
        13.6,
        3.5
    )
)


axes[0].imshow(
    mri,
    cmap="gray"
)

axes[0].set_title(
    "MRI"
)


axes[1].imshow(
    gt_show,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Ground Truth"
)


axes[2].imshow(
    pred_show,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Prediction"
)


axes[3].imshow(
    mri,
    cmap="gray"
)

if gt_show.any():

    axes[3].contour(
        gt_show,
        levels=[0.5],
        colors="lime",
        linewidths=2.0
    )


if pred_show.any():

    axes[3].contour(
        pred_show,
        levels=[0.5],
        colors="red",
        linewidths=1.6
    )


axes[3].set_title(
    "Overlay"
)


for ax in axes:
    ax.axis(
        "off"
    )


fig.suptitle(
    (
        f"Representative Over-segmentation Slice | "
        f"{TARGET_PATIENT} | "
        f"volume index={TARGET_INDEX} | "
        f"slice={slice_idx}"
    ),
    fontsize=13
)


fig.subplots_adjust(
    left=0.02,
    right=0.99,
    top=0.82,
    bottom=0.03,
    wspace=0.06
)


FIG_PATH = (
    OUT_DIR
    / "representative_overseg_slice_volume55.png"
)


fig.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close(
    fig
)


# ============================================================
# 16. FINISH
# ============================================================

print(
    "[SUCCESS] Slice-level analysis completed."
)

print(
    "\nOutput folder:"
)

print(
    OUT_DIR
)

print(
    "\nMain figure:"
)

print(
    FIG_PATH
)

print(
    "\nSelected-slice table:"
)

print(
    OUT_DIR
    / "SELECTED_slice.csv"
)