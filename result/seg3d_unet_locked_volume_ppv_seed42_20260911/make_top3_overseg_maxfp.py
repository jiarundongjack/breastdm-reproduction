# -*- coding: utf-8 -*-

"""
Generate publication-quality over-segmentation examples for the formal
BreastDM 3D U-Net reproduction run.

Logic
-----
1. Lock three pre-selected 3D volumes:
       index 101, 8, 38
2. Reconstruct the ORIGINAL formal test pipeline from code_snapshot:
       - original DriveDataset
       - original evaluation transform
       - original UNet
       - formal best_model.pth
       - argmax prediction
3. Before making figures, re-run each volume and verify that:
       dice / iou / ppv / GT-positive / Pred-positive
   match test_predictions.csv.
   If they do not match, STOP instead of silently drawing inconsistent figures.
4. Inside each selected volume:
       choose the slice with MAXIMUM FP
   among slices satisfying:
       GT area > 0
       TP > 0
       Predicted area > GT area
   This deliberately targets OVER-SEGMENTATION rather than complete
   localization failure.
5. Save:
       - 3 individual 4-panel figures
       - 1 combined 3x4 figure
       - 1 CSV containing exact slice-level evidence
"""

from pathlib import Path
import os
import sys
import inspect
import importlib.util

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch


# ============================================================
# 0. LOCKED SETTINGS
# ============================================================

RUN_DIR = Path.cwd().resolve()

SNAPSHOT_DIR = RUN_DIR / "code_snapshot"
TRAIN3D_PATH = SNAPSHOT_DIR / "train3d.py"
CHECKPOINT_PATH = RUN_DIR / "best_model.pth"
PREDICTIONS_CSV = RUN_DIR / "test_predictions.csv"

# Locked top-3 volume-level over-segmentation candidates
SELECTED_INDICES = [101, 8, 38]

OUTPUT_DIR = RUN_DIR / "top3_overseg_maxfp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Formal run fallback values confirmed from the retained run evidence
FORMAL_DATA_ROOT_FALLBACK = Path(
    r"D:\OneDrive\Desktop\BreaDM\seg3D_clean"
)
FORMAL_DEPTH_FALLBACK = 8

# Numerical tolerance when validating re-inference against test_predictions.csv
METRIC_TOL = 1e-5


# ============================================================
# 1. BASIC CHECKS
# ============================================================

required_files = [
    TRAIN3D_PATH,
    CHECKPOINT_PATH,
    PREDICTIONS_CSV,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found:\n{p}")

if not SNAPSHOT_DIR.exists():
    raise FileNotFoundError(f"code_snapshot not found:\n{SNAPSHOT_DIR}")


# ============================================================
# 2. IMPORT THE FORMAL SNAPSHOT WITHOUT MODIFYING IT
# ============================================================

# Make sure snapshot-local imports such as:
#   src.unet3d
#   my_dataset3d
#   train_utils
# resolve to the retained formal snapshot.
sys.path.insert(0, str(SNAPSHOT_DIR))

spec = importlib.util.spec_from_file_location(
    "formal_train3d_snapshot",
    str(TRAIN3D_PATH)
)

if spec is None or spec.loader is None:
    raise RuntimeError("Could not create import spec for code_snapshot/train3d.py")

formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)

for name in ["UNet", "DriveDataset", "get_transform"]:
    if not hasattr(formal, name):
        raise RuntimeError(
            f"Formal snapshot does not expose required object: {name}"
        )

UNet = formal.UNet
DriveDataset = formal.DriveDataset
get_transform = formal.get_transform


# ============================================================
# 3. LOAD FORMAL CHECKPOINT
# ============================================================

try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )
except TypeError:
    # Compatibility fallback for older torch versions
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )


def get_saved_args(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("args", None)
    return None


saved_args = get_saved_args(checkpoint)


def arg_value(*names, default=None):
    """
    Read a saved training argument from either:
      argparse.Namespace / SimpleNamespace / dict
    """
    for name in names:
        if saved_args is None:
            continue

        if isinstance(saved_args, dict):
            if name in saved_args:
                return saved_args[name]
        else:
            if hasattr(saved_args, name):
                return getattr(saved_args, name)

    return default


def extract_state_dict(ckpt):
    """
    Accept common checkpoint layouts without changing the formal weights.
    """
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint is not a dictionary.")

    for key in ["model", "state_dict", "model_state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        # Could itself already be a raw state_dict
        tensor_values = [
            torch.is_tensor(v)
            for v in ckpt.values()
        ]
        if len(tensor_values) > 0 and all(tensor_values):
            state = ckpt
        else:
            raise RuntimeError(
                "Could not identify model state_dict in best_model.pth"
            )

    cleaned = {}

    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    return cleaned


state_dict = extract_state_dict(checkpoint)


# ============================================================
# 4. RESOLVE FORMAL DATA ROOT / DEPTH
# ============================================================

data_root_raw = arg_value(
    "data_path",
    "data_root",
    "root",
    default=str(FORMAL_DATA_ROOT_FALLBACK)
)

DATA_ROOT = Path(str(data_root_raw))

DEPTH = int(
    arg_value(
        "depth",
        default=FORMAL_DEPTH_FALLBACK
    )
)

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        "Formal data root does not exist.\n"
        f"Resolved path: {DATA_ROOT}\n"
        "The retained formal run was audited with:\n"
        f"{FORMAL_DATA_ROOT_FALLBACK}"
    )


# ============================================================
# 5. BUILD THE ORIGINAL EVALUATION TRANSFORM
# ============================================================

def build_eval_transform():
    """
    Call the original snapshot get_transform() using its own signature.
    No new preprocessing is introduced here.
    """
    sig = inspect.signature(get_transform)
    params = sig.parameters

    kwargs = {}

    if "train" in params:
        kwargs["train"] = False

    # Only supply these if the retained function explicitly requests them.
    if "input_size" in params:
        kwargs["input_size"] = int(
            arg_value("input_size", default=224)
        )

    if "base_size" in params:
        kwargs["base_size"] = int(
            arg_value("base_size", default=224)
        )

    if "crop_size" in params:
        kwargs["crop_size"] = int(
            arg_value("crop_size", default=224)
        )

    try:
        return get_transform(**kwargs)

    except TypeError:
        # Most likely formal signature: get_transform(train)
        return get_transform(False)


eval_transform = build_eval_transform()


# ============================================================
# 6. BUILD THE ORIGINAL TEST DATASET
# ============================================================

def build_test_dataset():
    """
    Reconstruct DriveDataset from the retained snapshot.

    First use signature-guided construction.
    If the constructor differs slightly, try a few equivalent
    forms using the SAME formal root / split / transform / depth.
    """

    sig = inspect.signature(DriveDataset)
    params = sig.parameters

    kwargs = {}

    root_name = None

    for candidate in [
        "root",
        "data_root",
        "data_path",
        "path"
    ]:
        if candidate in params:
            root_name = candidate
            kwargs[candidate] = str(DATA_ROOT)
            break

    if "split" in params:
        kwargs["split"] = "test"

    if "transforms" in params:
        kwargs["transforms"] = eval_transform
    elif "transform" in params:
        kwargs["transform"] = eval_transform

    if "depth" in params:
        kwargs["depth"] = DEPTH

    if "train" in params and "split" not in params:
        kwargs["train"] = False

    # First attempt: fully keyword-driven
    if root_name is not None:
        try:
            return DriveDataset(**kwargs)
        except TypeError:
            pass

    # Fallback attempts for a positional root
    common_kwargs = {}

    if "split" in params:
        common_kwargs["split"] = "test"

    if "transforms" in params:
        common_kwargs["transforms"] = eval_transform
    elif "transform" in params:
        common_kwargs["transform"] = eval_transform

    if "depth" in params:
        common_kwargs["depth"] = DEPTH

    if "train" in params and "split" not in params:
        common_kwargs["train"] = False

    try:
        return DriveDataset(
            str(DATA_ROOT),
            **common_kwargs
        )

    except TypeError as e:
        raise RuntimeError(
            "Could not reconstruct formal DriveDataset.\n"
            f"DriveDataset signature: {sig}\n"
            f"Attempted root: {DATA_ROOT}\n"
            f"Original error: {e}"
        )


test_dataset = build_test_dataset()


# ============================================================
# 7. BUILD THE ORIGINAL UNET
# ============================================================

def infer_base_channels_from_state(state):
    """
    Infer the first Conv3D output-channel count only if needed.
    This does NOT alter weights.
    """
    preferred = []

    for k, v in state.items():
        if (
            torch.is_tensor(v)
            and v.ndim == 5
            and v.shape[1] == 1
        ):
            preferred.append((k, int(v.shape[0])))

    if preferred:
        preferred.sort(
            key=lambda x: (
                0 if "in_conv" in x[0].lower() else 1,
                len(x[0])
            )
        )
        return preferred[0][1]

    return None


def build_model():
    sig = inspect.signature(UNet)
    params = sig.parameters

    kwargs = {}

    # Input channels
    for name in [
        "in_channels",
        "n_channels",
        "input_channels",
        "in_ch"
    ]:
        if name in params:
            kwargs[name] = int(
                arg_value(
                    name,
                    "in_channels",
                    default=1
                )
            )
            break

    # Number of segmentation classes
    for name in [
        "num_classes",
        "n_classes",
        "classes"
    ]:
        if name in params:
            kwargs[name] = int(
                arg_value(
                    name,
                    "num_classes",
                    default=2
                )
            )
            break

    # Base feature channels, if present
    for name in [
        "base_c",
        "base_channels"
    ]:
        if name in params:
            value = arg_value(
                name,
                "base_c",
                "base_channels",
                default=None
            )

            if value is None:
                value = infer_base_channels_from_state(
                    state_dict
                )

            if value is not None:
                kwargs[name] = int(value)

    # Use checkpoint value only if the constructor has the option.
    if "bilinear" in params:
        saved = arg_value(
            "bilinear",
            default=None
        )
        if saved is not None:
            kwargs["bilinear"] = bool(saved)

    try:
        model = UNet(**kwargs)

    except Exception as e:
        raise RuntimeError(
            "Could not reconstruct formal UNet.\n"
            f"UNet signature: {sig}\n"
            f"Attempted kwargs: {kwargs}\n"
            f"Original error: {e}"
        )

    load_result = model.load_state_dict(
        state_dict,
        strict=True
    )

    # strict=True should already guarantee exact consistency
    if (
        len(load_result.missing_keys) != 0
        or len(load_result.unexpected_keys) != 0
    ):
        raise RuntimeError(
            "Formal best checkpoint did not load exactly.\n"
            f"Missing: {load_result.missing_keys}\n"
            f"Unexpected: {load_result.unexpected_keys}"
        )

    return model


model = build_model()


# ============================================================
# 8. DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)
model.eval()


# ============================================================
# 9. LOAD ORIGINAL PER-VOLUME DIAGNOSTIC RECORDS
# ============================================================

df = pd.read_csv(PREDICTIONS_CSV)

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

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "test_predictions.csv is missing required columns:\n"
        + "\n".join(missing_columns)
    )

df["index"] = pd.to_numeric(
    df["index"],
    errors="raise"
).astype(int)


# ============================================================
# 10. HELPERS
# ============================================================

def norm_path(p):
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(str(p))
        )
    )


def calculate_volume_metrics(target_np, pred_np):
    """
    Exactly follow save_test_predictions() logic seen in the
    retained formal train3d.py:

      valid = target != 255
      GT positive = target == 1
      Pred positive = pred == 1

      Dice: 2TP/(2TP+FP+FN), empty/empty -> 1
      IoU : TP/(TP+FP+FN), empty/empty -> 1
      PPV : TP/(TP+FP), no predicted positive -> 0

    These are CASE-LEVEL diagnostics saved in test_predictions.csv.
    They are NOT the final test-set global PPV aggregation.
    """

    valid = target_np != 255

    gt_pos = (target_np == 1) & valid
    pred_pos = (pred_np == 1) & valid

    tp = int(
        (gt_pos & pred_pos).sum()
    )

    fp = int(
        ((~gt_pos) & pred_pos & valid).sum()
    )

    fn = int(
        (gt_pos & (~pred_pos) & valid).sum()
    )

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    ppv_den = tp + fp

    dice = (
        (2.0 * tp / dice_den)
        if dice_den > 0
        else 1.0
    )

    iou = (
        (tp / iou_den)
        if iou_den > 0
        else 1.0
    )

    ppv = (
        (tp / ppv_den)
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


def validate_against_csv(
    idx,
    row,
    metrics
):
    """
    Critical safety gate:
    if re-inference is not reproducing the retained CSV case metrics,
    STOP rather than draw a misleading figure.
    """

    checks = {
        "dice": (
            metrics["dice"],
            float(row["dice"])
        ),
        "iou": (
            metrics["iou"],
            float(row["iou"])
        ),
        "ppv": (
            metrics["ppv"],
            float(row["ppv"])
        ),
    }

    failed = []

    for name, (new, old) in checks.items():
        if abs(new - old) > METRIC_TOL:
            failed.append(
                f"{name}: rerun={new:.8f}, csv={old:.8f}"
            )

    old_gt = int(row["gt_positive_pixels"])
    old_pred = int(row["pred_positive_pixels"])

    if metrics["gt_positive_pixels"] != old_gt:
        failed.append(
            "gt_positive_pixels: "
            f"rerun={metrics['gt_positive_pixels']}, "
            f"csv={old_gt}"
        )

    if metrics["pred_positive_pixels"] != old_pred:
        failed.append(
            "pred_positive_pixels: "
            f"rerun={metrics['pred_positive_pixels']}, "
            f"csv={old_pred}"
        )

    if failed:
        raise RuntimeError(
            f"\nINDEX {idx}: formal re-inference does NOT match "
            "test_predictions.csv.\n"
            "Figure generation has been stopped intentionally.\n"
            + "\n".join(failed)
        )

    print(
        f"[PASS] index {idx}: rerun metrics exactly match retained CSV."
    )


def verify_dataset_index(idx, row):
    """
    Confirm that dataset[idx] refers to the same formal test image
    recorded by save_test_predictions().
    """

    if hasattr(test_dataset, "images"):
        dataset_path = test_dataset.images[idx]
        csv_path = row["image_path"]

        if norm_path(dataset_path) != norm_path(csv_path):
            raise RuntimeError(
                f"Dataset index mismatch for index {idx}\n"
                f"Dataset: {dataset_path}\n"
                f"CSV    : {csv_path}"
            )

        print(
            f"[PASS] index {idx}: dataset image path matches CSV."
        )


def get_slice_stats(
    target_np,
    pred_np
):
    """
    Compute per-slice TP/FP/FN strictly for visualization selection.
    This does NOT redefine the formal 3D evaluation metrics.
    """

    valid = target_np != 255
    results = []

    depth = target_np.shape[0]

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
            ((~gt_s) & pred_s & valid_s).sum()
        )

        fn = int(
            (gt_s & (~pred_s) & valid_s).sum()
        )

        gt_area = int(
            gt_s.sum()
        )

        pred_area = int(
            pred_s.sum()
        )

        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        ppv_den = tp + fp

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

        pred_gt_ratio = (
            pred_area / gt_area
            if gt_area > 0
            else np.nan
        )

        results.append({
            "slice_idx": int(s),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "extra_area": int(
                pred_area - gt_area
            ),
            "pred_gt_ratio": float(
                pred_gt_ratio
            ) if np.isfinite(pred_gt_ratio)
            else np.nan,
            "dice": float(dice),
            "iou": float(iou),
            "ppv": float(ppv),
        })

    return results


def select_max_fp_overseg_slice(stats):
    """
    Primary rule:
      among slices where:
          GT > 0
          TP > 0
          Pred area > GT area
      choose the slice with the largest FP.

    Why impose these constraints?
      - GT > 0: must actually contain lesion
      - TP > 0: avoids selecting a pure localization-error slice
      - Pred > GT: makes the displayed slice compatible with
                   the intended over-segmentation interpretation

    Tie-break:
      1. higher FP
      2. larger Pred-GT excess
      3. larger Pred/GT ratio
      4. larger TP
    """

    eligible = [
        x for x in stats
        if (
            x["gt_area"] > 0
            and x["tp"] > 0
            and x["pred_area"] > x["gt_area"]
        )
    ]

    rule = (
        "max FP among slices with GT>0, TP>0, "
        "and PredArea>GTArea"
    )

    if not eligible:
        # Conservative fallback:
        # still require actual lesion overlap.
        eligible = [
            x for x in stats
            if (
                x["gt_area"] > 0
                and x["tp"] > 0
            )
        ]

        rule = (
            "fallback: max FP among slices with GT>0 and TP>0"
        )

    if not eligible:
        raise RuntimeError(
            "No slice with GT>0 and TP>0 was found. "
            "This volume cannot serve as a clean over-segmentation example."
        )

    def ratio_for_sort(x):
        r = x["pred_gt_ratio"]
        return (
            r if np.isfinite(r)
            else -1.0
        )

    selected = max(
        eligible,
        key=lambda x: (
            x["fp"],
            x["extra_area"],
            ratio_for_sort(x),
            x["tp"],
        )
    )

    return selected, rule


def normalize_for_display(image_slice):
    image_slice = np.asarray(
        image_slice,
        dtype=np.float32
    )

    vmin = float(
        image_slice.min()
    )

    vmax = float(
        image_slice.max()
    )

    return (
        image_slice - vmin
    ) / (
        vmax - vmin + 1e-8
    )


def draw_case(
    axes,
    image_show,
    gt_show,
    pred_show,
):
    titles = [
        "MRI",
        "Ground Truth",
        "Prediction",
        "Overlay",
    ]

    axes[0].imshow(
        image_show,
        cmap="gray"
    )

    axes[1].imshow(
        gt_show,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].imshow(
        pred_show,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].imshow(
        image_show,
        cmap="gray"
    )

    if np.any(gt_show):
        axes[3].contour(
            gt_show,
            levels=[0.5],
            colors="lime",
            linewidths=2.0
        )

    if np.any(pred_show):
        axes[3].contour(
            pred_show,
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
# 11. PROCESS THE THREE LOCKED VOLUMES
# ============================================================

all_cases = []
summary_rows = []

print("\n==============================================")
print("Formal 3D over-segmentation visualization")
print("Run directory :", RUN_DIR)
print("Data root     :", DATA_ROOT)
print("Depth         :", DEPTH)
print("Device        :", device)
print("Locked indices:", SELECTED_INDICES)
print("==============================================\n")


for rank, idx in enumerate(
    SELECTED_INDICES,
    start=1
):

    matched = df[
        df["index"] == idx
    ]

    if len(matched) != 1:
        raise RuntimeError(
            f"Expected exactly one CSV row for index {idx}, "
            f"found {len(matched)}."
        )

    row = matched.iloc[0]

    verify_dataset_index(
        idx,
        row
    )

    # ---------- exact original test item ----------
    image, target = test_dataset[idx]

    # image expected [1, D, H, W]
    # target expected [D, H, W]
    image_input = (
        image
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model(
            image_input
        )["out"]

        pred = (
            output
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

    # ---------- validate formal re-inference ----------
    volume_metrics = calculate_volume_metrics(
        target_np,
        pred
    )

    validate_against_csv(
        idx,
        row,
        volume_metrics
    )

    # ---------- choose max-FP lesion-overlap slice ----------
    slice_stats = get_slice_stats(
        target_np,
        pred
    )

    selected_slice, selection_rule = (
        select_max_fp_overseg_slice(
            slice_stats
        )
    )

    s = selected_slice[
        "slice_idx"
    ]

    # Original representative-figure convention:
    # show first MRI channel at the selected depth slice.
    image_show = (
        image[0, s]
        .cpu()
        .numpy()
    )

    image_show = normalize_for_display(
        image_show
    )

    valid_show = (
        target_np[s] != 255
    )

    gt_show = (
        (target_np[s] == 1)
        & valid_show
    ).astype(np.uint8)

    pred_show = (
        (pred[s] == 1)
        & valid_show
    ).astype(np.uint8)

    # Volume Pred/GT ratio from retained case record
    volume_gt = int(
        row["gt_positive_pixels"]
    )

    volume_pred = int(
        row["pred_positive_pixels"]
    )

    volume_pred_gt_ratio = (
        volume_pred / volume_gt
        if volume_gt > 0
        else np.nan
    )

    case_info = {
        "rank": rank,
        "index": idx,
        "patient": str(
            row["patient"]
        ),
        "sequence": str(
            row["sequence"]
        ),
        "filename": str(
            row["filename"]
        ),

        # CASE-LEVEL diagnostic values from retained CSV
        "volume_dice": float(
            row["dice"]
        ),
        "volume_iou": float(
            row["iou"]
        ),
        "volume_ppv_aux": float(
            row["ppv"]
        ),
        "volume_gt_positive": volume_gt,
        "volume_pred_positive": volume_pred,
        "volume_pred_gt_ratio": float(
            volume_pred_gt_ratio
        ),

        "slice_idx": int(s),
        "slice_tp": selected_slice["tp"],
        "slice_fp": selected_slice["fp"],
        "slice_fn": selected_slice["fn"],
        "slice_gt_area": selected_slice["gt_area"],
        "slice_pred_area": selected_slice["pred_area"],
        "slice_extra_area": selected_slice["extra_area"],
        "slice_pred_gt_ratio": selected_slice["pred_gt_ratio"],
        "slice_dice": selected_slice["dice"],
        "slice_iou": selected_slice["iou"],
        "slice_ppv": selected_slice["ppv"],
        "selection_rule": selection_rule,
    }

    # ---------- individual publication-style figure ----------
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.6)
    )

    draw_case(
        axes,
        image_show,
        gt_show,
        pred_show
    )

    fig.suptitle(
        (
            f"Representative Over-segmentation Case {rank} "
            f"(index={idx}, {case_info['patient']}, "
            f"{case_info['sequence']})\n"
            f"Volume-level diagnostic: "
            f"DSC={case_info['volume_dice']:.4f}, "
            f"IoU={case_info['volume_iou']:.4f}, "
            f"PPV={case_info['volume_ppv_aux']:.4f}, "
            f"Pred/GT={case_info['volume_pred_gt_ratio']:.2f} | "
            f"Displayed max-FP slice={s}: "
            f"TP={case_info['slice_tp']}, "
            f"FP={case_info['slice_fp']}, "
            f"FN={case_info['slice_fn']}"
        ),
        fontsize=12.5,
        y=0.99
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color="lime",
            lw=2,
            label="Ground Truth"
        ),
        Line2D(
            [0], [0],
            color="red",
            lw=1.5,
            label="Prediction"
        ),
    ]

    axes[3].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        frameon=True
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    safe_patient = (
        case_info["patient"]
        .replace("/", "_")
        .replace("\\", "_")
    )

    safe_sequence = (
        case_info["sequence"]
        .replace("+", "_plus_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    single_path = (
        OUTPUT_DIR
        / (
            f"case{rank}_index{idx}_"
            f"{safe_patient}_{safe_sequence}_"
            f"maxFP.png"
        )
    )

    fig.savefig(
        single_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    case_info["figure_path"] = str(
        single_path
    )

    summary_rows.append(
        case_info
    )

    all_cases.append({
        "info": case_info,
        "image": image_show,
        "gt": gt_show,
        "pred": pred_show,
    })

    print(
        f"\nCase {rank} | index {idx}"
    )
    print(
        f"  patient            : {case_info['patient']}"
    )
    print(
        f"  volume PPV (aux)   : {case_info['volume_ppv_aux']:.6f}"
    )
    print(
        f"  volume Pred/GT      : {case_info['volume_pred_gt_ratio']:.4f}"
    )
    print(
        f"  selected slice      : {s}"
    )
    print(
        f"  slice TP / FP / FN  : "
        f"{case_info['slice_tp']} / "
        f"{case_info['slice_fp']} / "
        f"{case_info['slice_fn']}"
    )
    print(
        f"  slice Pred/GT       : "
        f"{case_info['slice_pred_gt_ratio']:.4f}"
    )
    print(
        f"  slice PPV           : "
        f"{case_info['slice_ppv']:.6f}"
    )
    print(
        f"  selection rule      : {selection_rule}"
    )
    print(
        f"  saved               : {single_path}"
    )


# ============================================================
# 12. COMBINED 3 x 4 FIGURE = 12 PANELS
# ============================================================

fig, axes = plt.subplots(
    len(all_cases),
    4,
    figsize=(16, 12)
)

if len(all_cases) == 1:
    axes = np.expand_dims(
        axes,
        axis=0
    )

for r, case in enumerate(
    all_cases
):

    draw_case(
        axes[r],
        case["image"],
        case["gt"],
        case["pred"]
    )

    info = case["info"]

    row_text = (
        f"Case {info['rank']}  |  index={info['index']}\n"
        f"{info['patient']}  |  {info['sequence']}\n"
        f"Volume PPV={info['volume_ppv_aux']:.3f}, "
        f"Pred/GT={info['volume_pred_gt_ratio']:.2f}\n"
        f"Max-FP slice={info['slice_idx']}, "
        f"FP={info['slice_fp']}"
    )

    axes[r, 0].text(
        -0.08,
        0.50,
        row_text,
        transform=axes[r, 0].transAxes,
        ha="right",
        va="center",
        fontsize=10.5
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color="lime",
            lw=2,
            label="Ground Truth"
        ),
        Line2D(
            [0], [0],
            color="red",
            lw=1.5,
            label="Prediction"
        ),
    ]

    axes[r, 3].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True
    )


fig.suptitle(
    (
        "Representative Over-segmentation Cases of the "
        "Reproduced 3D U-Net\n"
        "Volumes were selected first; within each volume, "
        "the lesion-overlapping slice with the largest FP count "
        "was displayed."
    ),
    fontsize=14,
    y=0.995
)

fig.subplots_adjust(
    left=0.20,
    right=0.99,
    top=0.92,
    bottom=0.03,
    wspace=0.06,
    hspace=0.32
)

combined_path = (
    OUTPUT_DIR
    / "top3_overseg_maxFP_combined.png"
)

fig.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. SAVE EXACT EVIDENCE TABLE
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "top3_overseg_maxFP_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. FINISH
# ============================================================

print("\n==============================================")
print("DONE")
print("No original training code was modified.")
print("No retraining was performed.")
print("All 3 locked volumes passed CSV re-inference validation.")
print("")
print("Output directory:")
print(OUTPUT_DIR)
print("")
print("Combined 12-panel figure:")
print(combined_path)
print("")
print("Evidence summary:")
print(summary_path)
print("==============================================")
