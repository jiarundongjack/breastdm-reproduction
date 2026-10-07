# -*- coding: utf-8 -*-

"""
BreastDM 3D U-Net
Top-3 over-segmentation visualization using the FORMAL seed42 run.

IMPORTANT
---------
- Does NOT modify train3d.py.
- Does NOT retrain the model.
- Does NOT import the whole train3d.py training entry.
- Uses the retained formal:
    code_snapshot/src/unet3d.py
    code_snapshot/my_dataset3d.py
    best_model.pth
    test_predictions.csv
- Locked volumes: 101, 8, 38
- Within each volume, selects the MAX-FP slice among slices satisfying:
      GT area > 0
      TP > 0
      Predicted area > GT area
  so the visualization targets over-segmentation rather than complete
  localization failure.
"""

from pathlib import Path
import os
import sys
import inspect

import numpy as np
import pandas as pd

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 0. LOCKED CONFIGURATION
# ============================================================

RUN_DIR = Path.cwd().resolve()

SNAPSHOT_DIR = RUN_DIR / "code_snapshot"
BEST_MODEL = RUN_DIR / "best_model.pth"
TEST_CSV = RUN_DIR / "test_predictions.csv"

UNET_FILE = SNAPSHOT_DIR / "src" / "unet3d.py"
DATASET_FILE = SNAPSHOT_DIR / "my_dataset3d.py"

# Three volume-level cases already selected
LOCKED_INDICES = [101, 8, 38]

# Formal run configuration
FORMAL_DATA_ROOT = Path(
    r"D:\OneDrive\Desktop\BreaDM\seg3D_clean"
)
FORMAL_DEPTH = 8
FORMAL_INPUT_SIZE = 224

OUT_DIR = RUN_DIR / "top3_overseg_maxfp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_TOL = 1e-6


# ============================================================
# 1. FILE CHECK
# ============================================================

required_files = [
    UNET_FILE,
    DATASET_FILE,
    BEST_MODEL,
    TEST_CSV,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(
            f"\nRequired formal-run file not found:\n{p}"
        )

if not FORMAL_DATA_ROOT.exists():
    raise FileNotFoundError(
        "\nFormal 3D data root does not exist:\n"
        f"{FORMAL_DATA_ROOT}"
    )

print("\n[OK] Required formal-run files found.")


# ============================================================
# 2. IMPORT ONLY THE REQUIRED FORMAL MODULES
# ============================================================

# Put the retained snapshot at the front of sys.path.
# This allows:
#     from src.unet3d import UNet
#     from my_dataset3d import DriveDataset
# without importing train3d.py / train_utils.
sys.path.insert(0, str(SNAPSHOT_DIR))

from src.unet3d import UNet
from my_dataset3d import DriveDataset

print("[OK] Imported UNet and DriveDataset from formal code_snapshot.")


# ============================================================
# 3. EVALUATION TRANSFORM
# ============================================================

class FormalEvalIdentity:
    """
    Formal 3D Validation/Test SegmentationPresetEval is an identity
    transform: it returns image and target directly.

    Fixed depth processing, H/W resize, intensity clipping and z-score
    remain inside the retained DriveDataset implementation.
    """
    def __call__(self, image, target):
        return image, target


eval_transform = FormalEvalIdentity()


# ============================================================
# 4. BUILD FORMAL TEST DATASET ROBUSTLY
# ============================================================

def build_test_dataset():
    sig = inspect.signature(DriveDataset)
    params = sig.parameters

    kwargs = {}

    # Root aliases
    root_key = None
    for name in [
        "root",
        "data_root",
        "data_path",
        "root_dir",
        "dataset_root",
    ]:
        if name in params:
            root_key = name
            kwargs[name] = str(FORMAL_DATA_ROOT)
            break

    if "split" in params:
        kwargs["split"] = "test"

    if "transforms" in params:
        kwargs["transforms"] = eval_transform
    elif "transform" in params:
        kwargs["transform"] = eval_transform

    if "depth" in params:
        kwargs["depth"] = FORMAL_DEPTH

    if "input_size" in params:
        kwargs["input_size"] = FORMAL_INPUT_SIZE

    if "image_size" in params:
        kwargs["image_size"] = FORMAL_INPUT_SIZE

    if "size" in params:
        p = params["size"]
        if p.default is inspect._empty:
            kwargs["size"] = FORMAL_INPUT_SIZE

    # Some datasets use train=False instead of split="test"
    if "train" in params and "split" not in params:
        kwargs["train"] = False

    # Preferred: keyword construction
    if root_key is not None:
        try:
            dataset = DriveDataset(**kwargs)
            return dataset
        except TypeError as e1:
            keyword_error = e1
    else:
        keyword_error = None

    # Fallback: positional root
    fallback_kwargs = {
        k: v for k, v in kwargs.items()
        if k != root_key
    }

    try:
        dataset = DriveDataset(
            str(FORMAL_DATA_ROOT),
            **fallback_kwargs
        )
        return dataset

    except TypeError as e2:
        raise RuntimeError(
            "\nCould not reconstruct the formal DriveDataset.\n"
            f"DriveDataset signature:\n{sig}\n\n"
            f"Keyword attempt error:\n{keyword_error}\n\n"
            f"Positional-root attempt error:\n{e2}"
        )


test_dataset = build_test_dataset()

print("[OK] Formal test dataset constructed.")
print("     Test dataset length:", len(test_dataset))

if len(test_dataset) != 141:
    raise RuntimeError(
        f"\nFormal test dataset should contain 141 volumes, "
        f"but current dataset contains {len(test_dataset)}.\n"
        "STOPPED to avoid generating inconsistent figures."
    )


# ============================================================
# 5. LOAD FORMAL CHECKPOINT
# ============================================================

try:
    checkpoint = torch.load(
        BEST_MODEL,
        map_location="cpu",
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        BEST_MODEL,
        map_location="cpu"
    )


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError(
            "best_model.pth is not a dictionary checkpoint."
        )

    # Common formal checkpoint layouts
    for key in [
        "model",
        "state_dict",
        "model_state_dict"
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        # Raw state_dict fallback
        if len(ckpt) > 0 and all(
            torch.is_tensor(v)
            for v in ckpt.values()
        ):
            state = ckpt
        else:
            raise RuntimeError(
                "Could not identify model state_dict "
                "inside best_model.pth."
            )

    cleaned = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


state_dict = extract_state_dict(checkpoint)


# ============================================================
# 6. READ SAVED CHECKPOINT ARGS WHEN AVAILABLE
# ============================================================

saved_args = None

if isinstance(checkpoint, dict):
    saved_args = checkpoint.get("args", None)


def saved_arg(*names, default=None):
    if saved_args is None:
        return default

    for name in names:
        if isinstance(saved_args, dict):
            if name in saved_args:
                return saved_args[name]
        else:
            if hasattr(saved_args, name):
                return getattr(saved_args, name)

    return default


# ============================================================
# 7. INFER BASE CHANNELS FROM CHECKPOINT IF NECESSARY
# ============================================================

def infer_base_channels(state):
    candidates = []

    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue

        # Conv3D weights are usually:
        # [out_channels, in_channels, D, H, W]
        if tensor.ndim != 5:
            continue

        # Prefer the first encoder/input convolution
        if tensor.shape[1] == 1:
            priority = 0 if "in_conv" in key.lower() else 1
            candidates.append(
                (
                    priority,
                    len(key),
                    key,
                    int(tensor.shape[0])
                )
            )

    if not candidates:
        return None

    candidates.sort()

    chosen = candidates[0]

    print(
        "[INFO] Inferred base channel candidate:",
        chosen[2],
        "->",
        chosen[3]
    )

    return chosen[3]


inferred_base_c = infer_base_channels(state_dict)


# ============================================================
# 8. BUILD FORMAL UNET
# ============================================================

def build_formal_model():
    sig = inspect.signature(UNet)
    params = sig.parameters

    kwargs = {}

    for name, param in params.items():

        # Skip *args / **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        ):
            continue

        # Input channels
        if name in {
            "in_channels",
            "n_channels",
            "input_channels",
            "in_ch"
        }:
            value = saved_arg(
                name,
                "in_channels",
                default=1
            )
            kwargs[name] = int(value)
            continue

        # Number of classes
        if name in {
            "num_classes",
            "n_classes",
            "classes",
            "out_channels"
        }:
            value = saved_arg(
                name,
                "num_classes",
                default=2
            )
            kwargs[name] = int(value)
            continue

        # Base feature channels
        if name in {
            "base_c",
            "base_channels",
            "init_features",
            "features"
        }:
            value = saved_arg(
                name,
                "base_c",
                "base_channels",
                default=None
            )

            if value is None:
                value = inferred_base_c

            if value is not None:
                kwargs[name] = int(value)
                continue

        # Bilinear flag, only if constructor requires or saved args exist
        if name == "bilinear":
            value = saved_arg(
                "bilinear",
                default=None
            )

            if value is not None:
                kwargs[name] = bool(value)
                continue

        # If constructor already has a default, leave it untouched.
        if param.default is not inspect._empty:
            continue

        # Unknown mandatory constructor argument = stop safely
        raise RuntimeError(
            "\nUNet contains an unknown required constructor argument:\n"
            f"    {name}\n"
            f"Full UNet signature:\n    {sig}\n\n"
            "STOPPED rather than guessing."
        )

    print("[INFO] UNet constructor kwargs:", kwargs)

    model = UNet(**kwargs)

    # strict=True: any architecture mismatch must stop the script
    model.load_state_dict(
        state_dict,
        strict=True
    )

    return model


model = build_formal_model()

print("[OK] Formal best_model.pth loaded with strict=True.")


# ============================================================
# 9. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)
model.eval()

print("[INFO] Device:", device)


# ============================================================
# 10. LOAD RETAINED PER-VOLUME DIAGNOSTIC CSV
# ============================================================

df = pd.read_csv(TEST_CSV)

required_columns = [
    "index",
    "image_path",
    "mask_path",
    "patient",
    "sequence",
    "filename",
    "dice",
    "iou",
    "ppv",
    "gt_positive_pixels",
    "pred_positive_pixels",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "\ntest_predictions.csv is missing columns:\n"
        + "\n".join(missing)
    )

df["index"] = pd.to_numeric(
    df["index"],
    errors="raise"
).astype(int)

if len(df) != 141:
    raise RuntimeError(
        f"\ntest_predictions.csv should have 141 records, "
        f"but found {len(df)}."
    )

print("[OK] test_predictions.csv loaded.")


# ============================================================
# 11. HELPER: PATH NORMALIZATION
# ============================================================

def norm_path(path):
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(
                str(path)
            )
        )
    )


# ============================================================
# 12. HELPER: EXACT PER-VOLUME DIAGNOSTIC METRICS
# ============================================================

def volume_metrics(target_np, pred_np):
    """
    Mirrors retained save_test_predictions() case-level logic.

    NOTE:
    These are per-volume diagnostic values.
    They are NOT the final global test-set PPV aggregation.
    """

    valid = target_np != 255

    gt_pos = (
        (target_np == 1)
        & valid
    )

    pred_pos = (
        (pred_np == 1)
        & valid
    )

    tp = int(
        (gt_pos & pred_pos).sum()
    )

    fp = int(
        ((~gt_pos) & pred_pos & valid).sum()
    )

    fn = int(
        (gt_pos & (~pred_pos) & valid).sum()
    )

    dice_den = (
        2 * tp + fp + fn
    )

    iou_den = (
        tp + fp + fn
    )

    ppv_den = (
        tp + fp
    )

    dice = (
        2.0 * tp / dice_den
        if dice_den > 0
        else 1.0
    )

    iou = (
        tp / iou_den
        if iou_den > 0
        else 1.0
    )

    ppv = (
        tp / ppv_den
        if ppv_den > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
        "ppv": float(ppv),
        "gt_positive_pixels": int(
            gt_pos.sum()
        ),
        "pred_positive_pixels": int(
            pred_pos.sum()
        ),
    }


# ============================================================
# 13. HELPER: VALIDATE RE-INFERENCE AGAINST RETAINED CSV
# ============================================================

def validate_case(idx, row, m):

    failed = []

    for key in [
        "dice",
        "iou",
        "ppv"
    ]:
        old = float(row[key])
        new = float(m[key])

        if abs(old - new) > METRIC_TOL:
            failed.append(
                f"{key}: rerun={new:.8f}, csv={old:.8f}"
            )

    old_gt = int(
        row["gt_positive_pixels"]
    )
    old_pred = int(
        row["pred_positive_pixels"]
    )

    if m["gt_positive_pixels"] != old_gt:
        failed.append(
            "gt_positive_pixels: "
            f"rerun={m['gt_positive_pixels']}, "
            f"csv={old_gt}"
        )

    if m["pred_positive_pixels"] != old_pred:
        failed.append(
            "pred_positive_pixels: "
            f"rerun={m['pred_positive_pixels']}, "
            f"csv={old_pred}"
        )

    if failed:
        raise RuntimeError(
            f"\nINDEX {idx} failed formal re-inference validation.\n"
            "Figure generation STOPPED.\n\n"
            + "\n".join(failed)
        )

    print(
        f"[PASS] index {idx}: "
        "re-inference matches retained CSV."
    )


# ============================================================
# 14. HELPER: VERIFY DATASET INDEX MATCHES CSV PATH
# ============================================================

def verify_dataset_path(idx, row):

    if not hasattr(
        test_dataset,
        "images"
    ):
        print(
            "[WARN] test_dataset has no .images attribute; "
            "path-level verification skipped."
        )
        return

    dataset_image_path = (
        test_dataset.images[idx]
    )

    csv_image_path = (
        row["image_path"]
    )

    if norm_path(
        dataset_image_path
    ) != norm_path(
        csv_image_path
    ):
        raise RuntimeError(
            f"\nDataset/CSV index mismatch at index={idx}\n"
            f"Dataset: {dataset_image_path}\n"
            f"CSV    : {csv_image_path}\n"
            "STOPPED."
        )

    print(
        f"[PASS] index {idx}: "
        "dataset image path matches CSV."
    )


# ============================================================
# 15. HELPER: SLICE-LEVEL STATISTICS
# ============================================================

def calculate_slice_stats(
    target_np,
    pred_np
):
    """
    Slice-level statistics are ONLY for visualization selection.
    They do not redefine the formal 3D metrics.
    """

    valid = (
        target_np != 255
    )

    depth = target_np.shape[0]

    rows = []

    for s in range(depth):

        valid_s = valid[s]

        gt_s = (
            (target_np[s] == 1)
            & valid_s
        )

        pred_s = (
            (pred_np[s] == 1)
            & valid_s
        )

        tp = int(
            (gt_s & pred_s).sum()
        )

        fp = int(
            (
                (~gt_s)
                & pred_s
                & valid_s
            ).sum()
        )

        fn = int(
            (
                gt_s
                & (~pred_s)
                & valid_s
            ).sum()
        )

        gt_area = int(
            gt_s.sum()
        )

        pred_area = int(
            pred_s.sum()
        )

        dice_den = (
            2 * tp + fp + fn
        )

        iou_den = (
            tp + fp + fn
        )

        ppv_den = (
            tp + fp
        )

        dice = (
            2.0 * tp / dice_den
            if dice_den > 0
            else 1.0
        )

        iou = (
            tp / iou_den
            if iou_den > 0
            else 1.0
        )

        ppv = (
            tp / ppv_den
            if ppv_den > 0
            else 0.0
        )

        ratio = (
            pred_area / gt_area
            if gt_area > 0
            else np.nan
        )

        rows.append({
            "slice_idx": int(s),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "extra_area": (
                pred_area - gt_area
            ),
            "pred_gt_ratio": (
                float(ratio)
                if np.isfinite(ratio)
                else np.nan
            ),
            "dice": float(dice),
            "iou": float(iou),
            "ppv": float(ppv),
        })

    return rows


# ============================================================
# 16. HELPER: SELECT TRUE OVER-SEGMENTATION SLICE
# ============================================================

def choose_max_fp_overseg_slice(
    slice_rows
):
    """
    Eligible slice MUST satisfy:
        GT area > 0
        TP > 0
        Prediction area > GT area

    Then select:
        MAX FP

    Tie-break:
        1. larger extra predicted area
        2. larger Pred/GT ratio
        3. larger TP

    We intentionally DO NOT fall back to a pure localization-error
    slice. If no eligible slice exists, the program stops for that
    volume instead of mislabeling it as over-segmentation.
    """

    eligible = [
        r for r in slice_rows
        if (
            r["gt_area"] > 0
            and r["tp"] > 0
            and r["pred_area"] > r["gt_area"]
        )
    ]

    if not eligible:
        raise RuntimeError(
            "\nThis locked volume contains no slice satisfying:\n"
            "GT>0, TP>0, PredArea>GTArea.\n"
            "Therefore it cannot be used as a clean "
            "over-segmentation visualization."
        )

    def safe_ratio(r):
        value = r["pred_gt_ratio"]
        return (
            value
            if np.isfinite(value)
            else -1.0
        )

    best = max(
        eligible,
        key=lambda r: (
            r["fp"],
            r["extra_area"],
            safe_ratio(r),
            r["tp"],
        )
    )

    return best


# ============================================================
# 17. HELPER: DISPLAY NORMALIZATION
# ============================================================

def normalize_mri(slice_img):

    x = np.asarray(
        slice_img,
        dtype=np.float32
    )

    lo = float(x.min())
    hi = float(x.max())

    return (
        x - lo
    ) / (
        hi - lo + 1e-8
    )


# ============================================================
# 18. HELPER: DRAW FOUR PANELS
# ============================================================

def draw_four_panel(
    axes,
    mri,
    gt,
    pred
):

    titles = [
        "MRI",
        "Ground Truth",
        "Prediction",
        "Overlay",
    ]

    axes[0].imshow(
        mri,
        cmap="gray"
    )

    axes[1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].imshow(
        pred,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].imshow(
        mri,
        cmap="gray"
    )

    if np.any(gt):
        axes[3].contour(
            gt,
            levels=[0.5],
            colors="lime",
            linewidths=2.0
        )

    if np.any(pred):
        axes[3].contour(
            pred,
            levels=[0.5],
            colors="red",
            linewidths=1.5
        )

    for ax, title in zip(
        axes,
        titles
    ):
        ax.set_title(
            title,
            fontsize=13
        )
        ax.axis("off")


# ============================================================
# 19. RUN LOCKED CASES
# ============================================================

summary_rows = []
visual_cases = []

print("\n==============================================")
print("3D U-Net Over-segmentation Visualization")
print("----------------------------------------------")
print("Run dir       :", RUN_DIR)
print("Data root     :", FORMAL_DATA_ROOT)
print("Depth         :", FORMAL_DEPTH)
print("Input size    :", FORMAL_INPUT_SIZE)
print("Device        :", device)
print("Locked volumes:", LOCKED_INDICES)
print("==============================================\n")


for rank, idx in enumerate(
    LOCKED_INDICES,
    start=1
):

    matches = df[
        df["index"] == idx
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"\nExpected exactly one CSV record "
            f"for index {idx}; found {len(matches)}."
        )

    row = matches.iloc[0]

    print(
        f"\n========== CASE {rank}: INDEX {idx} =========="
    )

    verify_dataset_path(
        idx,
        row
    )

    # --------------------------------------------------------
    # Formal dataset item
    # --------------------------------------------------------

    image, target = (
        test_dataset[idx]
    )

    if image.ndim != 4:
        raise RuntimeError(
            f"Expected image [C,D,H,W], "
            f"got {tuple(image.shape)}"
        )

    if target.ndim != 3:
        raise RuntimeError(
            f"Expected target [D,H,W], "
            f"got {tuple(target.shape)}"
        )

    image_input = (
        image
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # Formal best-model inference
    # --------------------------------------------------------

    with torch.no_grad():

        output_dict = model(
            image_input
        )

        if not isinstance(
            output_dict,
            dict
        ) or "out" not in output_dict:
            raise RuntimeError(
                "Formal UNet output does not contain ['out']."
            )

        logits = (
            output_dict["out"]
        )

        pred = (
            logits
            .argmax(dim=1)[0]
            .cpu()
            .numpy()
        )

    target_np = (
        target
        .cpu()
        .numpy()
        .copy()
    )

    # --------------------------------------------------------
    # Re-inference validation
    # --------------------------------------------------------

    vm = volume_metrics(
        target_np,
        pred
    )

    validate_case(
        idx,
        row,
        vm
    )

    # --------------------------------------------------------
    # Slice analysis
    # --------------------------------------------------------

    slice_rows = (
        calculate_slice_stats(
            target_np,
            pred
        )
    )

    # Save every slice's evidence
    slice_df = pd.DataFrame(
        slice_rows
    )

    slice_csv = (
        OUT_DIR
        / f"case{rank}_index{idx}_slice_stats.csv"
    )

    slice_df.to_csv(
        slice_csv,
        index=False,
        encoding="utf-8-sig"
    )

    selected = (
        choose_max_fp_overseg_slice(
            slice_rows
        )
    )

    s = selected[
        "slice_idx"
    ]

    print(
        "[SELECTED] max-FP over-segmentation slice:",
        s
    )

    print(
        "           TP / FP / FN =",
        selected["tp"],
        "/",
        selected["fp"],
        "/",
        selected["fn"]
    )

    print(
        "           GT area / Pred area =",
        selected["gt_area"],
        "/",
        selected["pred_area"]
    )

    print(
        "           Slice Pred/GT =",
        f"{selected['pred_gt_ratio']:.4f}"
    )

    print(
        "           Slice PPV =",
        f"{selected['ppv']:.4f}"
    )

    # --------------------------------------------------------
    # Display slice
    # --------------------------------------------------------

    # Same convention as the original representative figure:
    # use image channel 0 at selected depth slice.
    mri = (
        image[0, s]
        .cpu()
        .numpy()
    )

    mri = normalize_mri(
        mri
    )

    valid_s = (
        target_np[s] != 255
    )

    gt_show = (
        (target_np[s] == 1)
        & valid_s
    ).astype(np.uint8)

    pred_show = (
        (pred[s] == 1)
        & valid_s
    ).astype(np.uint8)

    volume_gt = int(
        row["gt_positive_pixels"]
    )

    volume_pred = int(
        row["pred_positive_pixels"]
    )

    volume_ratio = (
        volume_pred / volume_gt
        if volume_gt > 0
        else np.nan
    )

    patient = str(
        row["patient"]
    )

    sequence = str(
        row["sequence"]
    )

    filename = str(
        row["filename"]
    )

    case_ppv = float(
        row["ppv"]
    )

    case_dice = float(
        row["dice"]
    )

    case_iou = float(
        row["iou"]
    )

    # --------------------------------------------------------
    # Detailed individual figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.5)
    )

    draw_four_panel(
        axes,
        mri,
        gt_show,
        pred_show
    )

    legend = [
        Line2D(
            [0],
            [0],
            color="lime",
            linewidth=2,
            label="Ground Truth"
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.5,
            label="Prediction"
        ),
    ]

    axes[3].legend(
        handles=legend,
        loc="lower right",
        fontsize=9,
        frameon=True
    )

    fig.suptitle(
        (
            f"Representative Over-segmentation Case {rank}\n"
            f"{patient} | {sequence} | index={idx}\n"
            f"Case-level DSC={case_dice:.4f} | "
            f"IoU={case_iou:.4f} | "
            f"PPV={case_ppv:.4f} | "
            f"Volume Pred/GT={volume_ratio:.2f}\n"
            f"Displayed max-FP slice={s}: "
            f"TP={selected['tp']} | "
            f"FP={selected['fp']} | "
            f"FN={selected['fn']} | "
            f"Slice Pred/GT={selected['pred_gt_ratio']:.2f}"
        ),
        fontsize=12,
        y=0.995
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.84
        ]
    )

    safe_patient = (
        patient
        .replace("\\", "_")
        .replace("/", "_")
    )

    safe_sequence = (
        sequence
        .replace("+", "_plus_")
        .replace("\\", "_")
        .replace("/", "_")
    )

    individual_path = (
        OUT_DIR
        / (
            f"case{rank}_"
            f"index{idx}_"
            f"{safe_patient}_"
            f"{safe_sequence}_"
            f"maxFP.png"
        )
    )

    fig.savefig(
        individual_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Store for final clean combined figure
    # --------------------------------------------------------

    visual_cases.append({
        "rank": rank,
        "index": idx,
        "patient": patient,
        "sequence": sequence,
        "mri": mri,
        "gt": gt_show,
        "pred": pred_show,
        "slice_idx": s,
        "case_ppv": case_ppv,
        "volume_ratio": volume_ratio,
        "slice_fp": selected["fp"],
    })

    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    summary_rows.append({
        "rank": rank,
        "index": idx,
        "patient": patient,
        "sequence": sequence,
        "filename": filename,

        "case_level_dice": case_dice,
        "case_level_iou": case_iou,
        "case_level_ppv": case_ppv,

        "volume_gt_positive_pixels": volume_gt,
        "volume_pred_positive_pixels": volume_pred,
        "volume_pred_gt_ratio": volume_ratio,

        "selected_slice_idx": s,

        "slice_tp": selected["tp"],
        "slice_fp": selected["fp"],
        "slice_fn": selected["fn"],

        "slice_gt_area": selected["gt_area"],
        "slice_pred_area": selected["pred_area"],
        "slice_extra_area": selected["extra_area"],
        "slice_pred_gt_ratio": selected["pred_gt_ratio"],

        "slice_dice": selected["dice"],
        "slice_iou": selected["iou"],
        "slice_ppv": selected["ppv"],

        "selection_rule":
            "max FP among slices with GT>0, TP>0, PredArea>GTArea",

        "individual_figure": str(
            individual_path
        ),

        "slice_stats_csv": str(
            slice_csv
        ),
    })

    print(
        "[SAVED]",
        individual_path
    )


# ============================================================
# 20. CLEAN 3 x 4 MANUSCRIPT FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(15.5, 11.0)
)

column_titles = [
    "MRI",
    "Ground Truth",
    "Prediction",
    "Overlay",
]

for r, case in enumerate(
    visual_cases
):

    draw_four_panel(
        axes[r],
        case["mri"],
        case["gt"],
        case["pred"]
    )

    # Column titles only once at top
    if r > 0:
        for c in range(4):
            axes[r, c].set_title("")

    # Case label at left
    axes[r, 0].text(
        -0.08,
        0.5,
        (
            f"Case {case['rank']}\n"
            f"index={case['index']}\n"
            f"{case['patient']}\n"
            f"max-FP slice={case['slice_idx']}"
        ),
        transform=axes[r, 0].transAxes,
        ha="right",
        va="center",
        fontsize=10.5
    )

# Restore top-row column titles cleanly
for c, title in enumerate(
    column_titles
):
    axes[0, c].set_title(
        title,
        fontsize=13
    )

global_legend = [
    Line2D(
        [0],
        [0],
        color="lime",
        linewidth=2,
        label="Ground Truth"
    ),
    Line2D(
        [0],
        [0],
        color="red",
        linewidth=1.5,
        label="Prediction"
    ),
]

fig.legend(
    handles=global_legend,
    loc="lower center",
    ncol=2,
    fontsize=10,
    frameon=False
)

fig.suptitle(
    "Representative Over-segmentation Cases of the Reproduced 3D U-Net",
    fontsize=15,
    y=0.985
)

fig.subplots_adjust(
    left=0.19,
    right=0.99,
    top=0.93,
    bottom=0.07,
    wspace=0.05,
    hspace=0.18
)

combined_clean_path = (
    OUT_DIR
    / "top3_overseg_maxFP_combined_clean.png"
)

fig.savefig(
    combined_clean_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 21. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUT_DIR
    / "top3_overseg_maxFP_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 22. FINAL REPORT
# ============================================================

print("\n==============================================")
print("SUCCESS")
print("----------------------------------------------")
print("No training code was modified.")
print("No retraining was performed.")
print("The formal best checkpoint was reused.")
print("The formal test dataset was reused.")
print("All locked cases passed re-inference validation.")
print("")
print("Output folder:")
print(OUT_DIR)
print("")
print("Clean 12-panel manuscript figure:")
print(combined_clean_path)
print("")
print("Evidence summary CSV:")
print(summary_path)
print("==============================================")
