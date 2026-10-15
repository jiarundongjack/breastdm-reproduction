# -*- coding: utf-8 -*-

"""
BreastDM 3D U-Net
V3 - Top-3 over-segmentation visualization

Purpose
-------
1. Reuse the FORMAL seed42 3D U-Net best checkpoint.
2. Lock three already-selected over-segmentation volumes:
       index 101, 8, 38
3. Re-run inference using the retained formal:
       code_snapshot/src/unet3d.py
       code_snapshot/my_dataset3d.py
       best_model.pth
4. Verify re-inference against test_predictions.csv.
5. Within each locked volume, select the slice with the largest FP
   among slices satisfying:
       GT area > 0
       TP > 0
       Predicted area > GT area
6. Save:
       - 3 individual four-panel figures
       - 1 combined 3x4 = 12-panel figure
       - slice-level audit CSVs
       - one summary CSV

This script does NOT:
    - modify train3d.py
    - retrain
    - alter the checkpoint
    - alter the formal test metrics
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
# 0. LOCKED FORMAL RUN SETTINGS
# ============================================================

RUN_DIR = Path.cwd().resolve()

SNAPSHOT_DIR = RUN_DIR / "code_snapshot"
BEST_MODEL = RUN_DIR / "best_model.pth"
TEST_CSV = RUN_DIR / "test_predictions.csv"

UNET_FILE = SNAPSHOT_DIR / "src" / "unet3d.py"
DATASET_FILE = SNAPSHOT_DIR / "my_dataset3d.py"

# Locked top-3 volume-level over-segmentation candidates
LOCKED_INDICES = [101, 8, 38]

# Formal run data configuration
FORMAL_DATA_ROOT = Path(
    r"D:\OneDrive\Desktop\BreaDM\seg3D_clean"
)

FORMAL_DEPTH = 8
FORMAL_INPUT_SIZE = 224

OUT_DIR = RUN_DIR / "top3_overseg_maxfp_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# test_predictions.csv was produced from the same inference outputs.
# Tight tolerance is therefore appropriate.
METRIC_TOL = 1e-6


# ============================================================
# 1. REQUIRED FILE CHECK
# ============================================================

required_files = [
    UNET_FILE,
    DATASET_FILE,
    BEST_MODEL,
    TEST_CSV,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired formal-run file not found:\n{path}"
        )

if not FORMAL_DATA_ROOT.exists():
    raise FileNotFoundError(
        "\nFormal 3D data root does not exist:\n"
        f"{FORMAL_DATA_ROOT}"
    )

print("\n[OK] Required formal-run files found.")


# ============================================================
# 2. IMPORT ONLY FORMAL MODEL + DATASET
# ============================================================

# Important:
# do NOT import train3d.py, because doing so imports training-only
# modules such as train_utils.
sys.path.insert(
    0,
    str(SNAPSHOT_DIR)
)

from src.unet3d import UNet
from my_dataset3d import DriveDataset

print(
    "[OK] Imported UNet and DriveDataset "
    "from formal code_snapshot."
)


# ============================================================
# 3. FORMAL EVALUATION TRANSFORM
# ============================================================

class FormalEvalIdentity:
    """
    Formal 3D validation/test SegmentationPresetEval directly
    returns image and target.

    Fixed depth handling, spatial resize, clipping and z-score
    remain inside the retained DriveDataset implementation.
    """

    def __call__(self, image, target):
        return image, target


eval_transform = FormalEvalIdentity()


# ============================================================
# 4. BUILD FORMAL TEST DATASET
# ============================================================

def build_test_dataset():

    sig = inspect.signature(
        DriveDataset
    )

    params = sig.parameters
    kwargs = {}

    root_key = None

    for candidate in [
        "root",
        "data_root",
        "data_path",
        "root_dir",
        "dataset_root",
    ]:
        if candidate in params:
            root_key = candidate
            kwargs[candidate] = str(
                FORMAL_DATA_ROOT
            )
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

    if (
        "train" in params
        and "split" not in params
    ):
        kwargs["train"] = False

    # Preferred keyword construction
    if root_key is not None:
        try:
            return DriveDataset(
                **kwargs
            )

        except TypeError as e_keyword:
            keyword_error = e_keyword

    else:
        keyword_error = None

    # Fallback: root as positional argument
    fallback_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k != root_key
    }

    try:
        return DriveDataset(
            str(FORMAL_DATA_ROOT),
            **fallback_kwargs
        )

    except TypeError as e_positional:
        raise RuntimeError(
            "\nCould not reconstruct formal DriveDataset.\n"
            f"Signature:\n{sig}\n\n"
            f"Keyword error:\n{keyword_error}\n\n"
            f"Positional-root error:\n{e_positional}"
        )


test_dataset = build_test_dataset()

print(
    "[OK] Formal test dataset constructed."
)

print(
    "     Test dataset length:",
    len(test_dataset)
)

if len(test_dataset) != 141:
    raise RuntimeError(
        "\nFormal test dataset mismatch.\n"
        f"Expected 141 volumes, got {len(test_dataset)}.\n"
        "STOPPED."
    )


# ============================================================
# 5. LOAD FORMAL BEST CHECKPOINT
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

    if not isinstance(
        ckpt,
        dict
    ):
        raise TypeError(
            "best_model.pth is not a dictionary checkpoint."
        )

    for key in [
        "model",
        "state_dict",
        "model_state_dict",
    ]:

        if (
            key in ckpt
            and isinstance(
                ckpt[key],
                dict
            )
        ):
            state = ckpt[key]
            break

    else:

        if (
            len(ckpt) > 0
            and all(
                torch.is_tensor(v)
                for v in ckpt.values()
            )
        ):
            state = ckpt

        else:
            raise RuntimeError(
                "Could not identify model state_dict "
                "inside best_model.pth."
            )

    cleaned = {}

    for key, value in state.items():

        if key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


state_dict = extract_state_dict(
    checkpoint
)

print(
    "[OK] Formal checkpoint state_dict extracted."
)


# ============================================================
# 6. RECOVER TRUE UNET ARCHITECTURE FROM CHECKPOINT ITSELF
# ============================================================

def find_first_existing_key(
    state,
    candidates
):

    for key in candidates:
        if key in state:
            return key

    return None


# ---------- input/base convolution ----------

input_conv_key = find_first_existing_key(
    state_dict,
    [
        "in_conv.0.weight",
        "in_conv.double_conv.0.weight",
        "inc.double_conv.0.weight",
    ]
)

if input_conv_key is None:

    # Conservative fallback:
    # find first Conv3D weight whose input channel = 1
    candidate_keys = []

    for key, tensor in state_dict.items():

        if (
            torch.is_tensor(tensor)
            and tensor.ndim == 5
            and tensor.shape[1] == 1
        ):

            candidate_keys.append(
                key
            )

    if not candidate_keys:
        raise RuntimeError(
            "\nCould not infer the first 3D convolution "
            "from the formal checkpoint."
        )

    candidate_keys.sort()

    input_conv_key = candidate_keys[0]


input_weight = state_dict[
    input_conv_key
]

if input_weight.ndim != 5:
    raise RuntimeError(
        f"\nUnexpected input conv shape:\n"
        f"{input_conv_key}: {tuple(input_weight.shape)}"
    )

IN_CHANNELS = int(
    input_weight.shape[1]
)

BASE_C = int(
    input_weight.shape[0]
)


# ---------- output convolution ----------

output_conv_key = find_first_existing_key(
    state_dict,
    [
        "out_conv.0.weight",
        "out_conv.weight",
        "outc.conv.weight",
    ]
)

if output_conv_key is None:

    # Search likely final 1x1x1 Conv3D layers
    output_candidates = []

    for key, tensor in state_dict.items():

        if (
            torch.is_tensor(tensor)
            and tensor.ndim == 5
            and tuple(
                tensor.shape[2:]
            ) == (1, 1, 1)
        ):

            if (
                "out" in key.lower()
                or "final" in key.lower()
            ):

                output_candidates.append(
                    key
                )

    if not output_candidates:
        raise RuntimeError(
            "\nCould not infer the final segmentation "
            "output convolution from checkpoint."
        )

    output_candidates.sort()

    output_conv_key = (
        output_candidates[-1]
    )


output_weight = state_dict[
    output_conv_key
]

NUM_CLASSES = int(
    output_weight.shape[0]
)


print("\n[CHECKPOINT ARCHITECTURE]")
print(
    "Input conv key :",
    input_conv_key
)
print(
    "Input weight   :",
    tuple(input_weight.shape)
)
print(
    "Output conv key:",
    output_conv_key
)
print(
    "Output weight  :",
    tuple(output_weight.shape)
)

print(
    "Recovered architecture:"
)
print(
    "  in_channels =",
    IN_CHANNELS
)
print(
    "  num_classes =",
    NUM_CLASSES
)
print(
    "  base_c      =",
    BASE_C
)


# These assertions protect against the exact V2 failure.
if IN_CHANNELS != 1:
    raise RuntimeError(
        f"\nUnexpected formal in_channels={IN_CHANNELS}; "
        "expected 1."
    )

if NUM_CLASSES != 2:
    raise RuntimeError(
        f"\nUnexpected formal num_classes={NUM_CLASSES}; "
        "checkpoint should represent "
        "background + tumor = 2 classes."
    )

if BASE_C != 32:
    raise RuntimeError(
        f"\nUnexpected formal base_c={BASE_C}; "
        "expected 32 from retained checkpoint."
    )


# ============================================================
# 7. BUILD EXACT FORMAL UNET
# ============================================================

model_sig = inspect.signature(
    UNet
)

print(
    "\nUNet constructor signature:",
    model_sig
)


model_kwargs = {}

if "in_channels" in model_sig.parameters:
    model_kwargs[
        "in_channels"
    ] = IN_CHANNELS

elif "n_channels" in model_sig.parameters:
    model_kwargs[
        "n_channels"
    ] = IN_CHANNELS

else:
    raise RuntimeError(
        "\nUNet constructor has no recognized "
        "input-channel argument."
    )


if "num_classes" in model_sig.parameters:
    model_kwargs[
        "num_classes"
    ] = NUM_CLASSES

elif "n_classes" in model_sig.parameters:
    model_kwargs[
        "n_classes"
    ] = NUM_CLASSES

else:
    raise RuntimeError(
        "\nUNet constructor has no recognized "
        "class-count argument."
    )


if "base_c" in model_sig.parameters:
    model_kwargs[
        "base_c"
    ] = BASE_C

elif "base_channels" in model_sig.parameters:
    model_kwargs[
        "base_channels"
    ] = BASE_C


print(
    "[INFO] UNet constructor kwargs:",
    model_kwargs
)


model = UNet(
    **model_kwargs
)

# Strict loading is intentional:
# any mismatch must stop the visualization.
model.load_state_dict(
    state_dict,
    strict=True
)

print(
    "[OK] Formal best checkpoint loaded "
    "with strict=True."
)


# ============================================================
# 8. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(
    device
)

model.eval()

print(
    "[INFO] Device:",
    device
)


# ============================================================
# 9. LOAD RETAINED TEST-PREDICTION TABLE
# ============================================================

df = pd.read_csv(
    TEST_CSV
)

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
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "\ntest_predictions.csv missing columns:\n"
        + "\n".join(
            missing_columns
        )
    )


df["index"] = pd.to_numeric(
    df["index"],
    errors="raise"
).astype(int)


if len(df) != 141:
    raise RuntimeError(
        f"\nExpected 141 rows in test_predictions.csv; "
        f"found {len(df)}."
    )

print(
    "[OK] test_predictions.csv loaded."
)


# ============================================================
# 10. PATH NORMALIZATION
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
# 11. PER-VOLUME DIAGNOSTIC METRICS
# ============================================================

def calculate_volume_metrics(
    target_np,
    pred_np
):

    valid = (
        target_np != 255
    )

    gt_pos = (
        (target_np == 1)
        & valid
    )

    pred_pos = (
        (pred_np == 1)
        & valid
    )

    tp = int(
        (
            gt_pos
            & pred_pos
        ).sum()
    )

    fp = int(
        (
            (~gt_pos)
            & pred_pos
            & valid
        ).sum()
    )

    fn = int(
        (
            gt_pos
            & (~pred_pos)
            & valid
        ).sum()
    )

    dice_den = (
        2 * tp
        + fp
        + fn
    )

    iou_den = (
        tp
        + fp
        + fn
    )

    ppv_den = (
        tp
        + fp
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

        "dice": float(
            dice
        ),

        "iou": float(
            iou
        ),

        "ppv": float(
            ppv
        ),

        "gt_positive_pixels": int(
            gt_pos.sum()
        ),

        "pred_positive_pixels": int(
            pred_pos.sum()
        ),
    }


# ============================================================
# 12. VERIFY DATASET INDEX
# ============================================================

def verify_dataset_index(
    idx,
    row
):

    if not hasattr(
        test_dataset,
        "images"
    ):

        print(
            "[WARN] test_dataset has no .images attribute; "
            "path-level verification skipped."
        )

        return

    dataset_path = (
        test_dataset.images[idx]
    )

    csv_path = (
        row["image_path"]
    )

    if norm_path(
        dataset_path
    ) != norm_path(
        csv_path
    ):

        raise RuntimeError(
            f"\nDataset index mismatch for index {idx}\n"
            f"Dataset: {dataset_path}\n"
            f"CSV    : {csv_path}\n"
            "STOPPED."
        )

    print(
        f"[PASS] index {idx}: "
        "dataset image path matches CSV."
    )


# ============================================================
# 13. VERIFY RE-INFERENCE
# ============================================================

def validate_reinference(
    idx,
    row,
    metrics
):

    failures = []

    for name in [
        "dice",
        "iou",
        "ppv"
    ]:

        original = float(
            row[name]
        )

        rerun = float(
            metrics[name]
        )

        if abs(
            original - rerun
        ) > METRIC_TOL:

            failures.append(
                f"{name}: "
                f"rerun={rerun:.8f}, "
                f"csv={original:.8f}"
            )


    original_gt = int(
        row[
            "gt_positive_pixels"
        ]
    )

    original_pred = int(
        row[
            "pred_positive_pixels"
        ]
    )


    if (
        metrics[
            "gt_positive_pixels"
        ]
        != original_gt
    ):

        failures.append(
            "gt_positive_pixels: "
            f"rerun="
            f"{metrics['gt_positive_pixels']}, "
            f"csv={original_gt}"
        )


    if (
        metrics[
            "pred_positive_pixels"
        ]
        != original_pred
    ):

        failures.append(
            "pred_positive_pixels: "
            f"rerun="
            f"{metrics['pred_positive_pixels']}, "
            f"csv={original_pred}"
        )


    if failures:

        raise RuntimeError(
            f"\nINDEX {idx}: formal re-inference "
            "does NOT match retained CSV.\n"
            "Figure generation intentionally stopped.\n\n"
            + "\n".join(
                failures
            )
        )


    print(
        f"[PASS] index {idx}: "
        "re-inference matches retained CSV."
    )


# ============================================================
# 14. SLICE-LEVEL AUDIT
# ============================================================

def calculate_slice_stats(
    target_np,
    pred_np
):

    valid = (
        target_np != 255
    )

    depth = (
        target_np.shape[0]
    )

    stats = []

    for slice_idx in range(
        depth
    ):

        valid_s = (
            valid[slice_idx]
        )

        gt_s = (
            (
                target_np[
                    slice_idx
                ] == 1
            )
            & valid_s
        )

        pred_s = (
            (
                pred_np[
                    slice_idx
                ] == 1
            )
            & valid_s
        )

        tp = int(
            (
                gt_s
                & pred_s
            ).sum()
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
            2 * tp
            + fp
            + fn
        )

        iou_den = (
            tp
            + fp
            + fn
        )

        ppv_den = (
            tp
            + fp
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

        stats.append({
            "slice_idx":
                int(
                    slice_idx
                ),

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "gt_area":
                gt_area,

            "pred_area":
                pred_area,

            "extra_area":
                int(
                    pred_area
                    - gt_area
                ),

            "pred_gt_ratio":
                (
                    float(
                        ratio
                    )
                    if np.isfinite(
                        ratio
                    )
                    else np.nan
                ),

            "dice":
                float(
                    dice
                ),

            "iou":
                float(
                    iou
                ),

            "ppv":
                float(
                    ppv
                ),
        })

    return stats


# ============================================================
# 15. SELECT MAX-FP TRUE OVER-SEGMENTATION SLICE
# ============================================================

def select_overseg_slice(
    stats
):

    eligible = [
        row
        for row in stats
        if (
            row["gt_area"] > 0
            and row["tp"] > 0
            and row["pred_area"]
                > row["gt_area"]
        )
    ]


    if not eligible:

        raise RuntimeError(
            "\nNo slice satisfies all over-segmentation "
            "visualization requirements:\n"
            "  GT area > 0\n"
            "  TP > 0\n"
            "  Predicted area > GT area\n\n"
            "This volume will NOT be mislabeled "
            "as an over-segmentation example."
        )


    def safe_ratio(
        row
    ):

        ratio = (
            row[
                "pred_gt_ratio"
            ]
        )

        if np.isfinite(
            ratio
        ):
            return ratio

        return -1.0


    selected = max(
        eligible,
        key=lambda row: (
            row["fp"],
            row["extra_area"],
            safe_ratio(
                row
            ),
            row["tp"],
        )
    )

    return selected


# ============================================================
# 16. DISPLAY NORMALIZATION
# ============================================================

def normalize_mri(
    image
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image_min = float(
        image.min()
    )

    image_max = float(
        image.max()
    )

    return (
        image - image_min
    ) / (
        image_max
        - image_min
        + 1e-8
    )


# ============================================================
# 17. DRAW FOUR-PANEL CASE
# ============================================================

def draw_case(
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


    if np.any(
        gt
    ):

        axes[3].contour(
            gt,
            levels=[
                0.5
            ],
            colors="lime",
            linewidths=2.0
        )


    if np.any(
        pred
    ):

        axes[3].contour(
            pred,
            levels=[
                0.5
            ],
            colors="red",
            linewidths=1.5
        )


    for axis, title in zip(
        axes,
        titles
    ):

        axis.set_title(
            title,
            fontsize=13
        )

        axis.axis(
            "off"
        )


# ============================================================
# 18. PROCESS LOCKED TOP-3 VOLUMES
# ============================================================

visual_cases = []
summary_rows = []


print("\n==============================================")
print("BreastDM 3D U-Net Over-segmentation V3")
print("----------------------------------------------")
print("Run directory :", RUN_DIR)
print("Data root     :", FORMAL_DATA_ROOT)
print("Depth         :", FORMAL_DEPTH)
print("Input size    :", FORMAL_INPUT_SIZE)
print("Device        :", device)
print("Locked indices:", LOCKED_INDICES)
print("==============================================\n")


for rank, idx in enumerate(
    LOCKED_INDICES,
    start=1
):

    print(
        f"\n========== CASE {rank} | INDEX {idx} =========="
    )


    matched = df[
        df["index"] == idx
    ]


    if len(
        matched
    ) != 1:

        raise RuntimeError(
            f"\nExpected exactly one record "
            f"for index={idx}; "
            f"found {len(matched)}."
        )


    row = (
        matched.iloc[0]
    )


    # --------------------------------------------------------
    # Dataset / CSV identity check
    # --------------------------------------------------------

    verify_dataset_index(
        idx,
        row
    )


    # --------------------------------------------------------
    # Load formal volume
    # --------------------------------------------------------

    image, target = (
        test_dataset[
            idx
        ]
    )


    if image.ndim != 4:

        raise RuntimeError(
            f"\nExpected image shape [C,D,H,W], "
            f"got {tuple(image.shape)}."
        )


    if target.ndim != 3:

        raise RuntimeError(
            f"\nExpected target shape [D,H,W], "
            f"got {tuple(target.shape)}."
        )


    image_input = (
        image
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------------
    # Formal inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_input
        )


        if (
            not isinstance(
                output,
                dict
            )
            or "out" not in output
        ):

            raise RuntimeError(
                "\nFormal UNet output does not "
                "contain dictionary key 'out'."
            )


        logits = (
            output[
                "out"
            ]
        )


        if logits.shape[1] != 2:

            raise RuntimeError(
                f"\nExpected 2 output classes; "
                f"got logits shape "
                f"{tuple(logits.shape)}."
            )


        pred = (
            logits
            .argmax(
                dim=1
            )[0]
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
    # Formal case-level validation
    # --------------------------------------------------------

    vm = (
        calculate_volume_metrics(
            target_np,
            pred
        )
    )


    validate_reinference(
        idx,
        row,
        vm
    )


    # --------------------------------------------------------
    # Slice-level audit
    # --------------------------------------------------------

    slice_stats = (
        calculate_slice_stats(
            target_np,
            pred
        )
    )


    slice_stats_df = (
        pd.DataFrame(
            slice_stats
        )
    )


    slice_stats_path = (
        OUT_DIR
        / (
            f"case{rank}_"
            f"index{idx}_"
            f"slice_stats.csv"
        )
    )


    slice_stats_df.to_csv(
        slice_stats_path,
        index=False,
        encoding="utf-8-sig"
    )


    selected = (
        select_overseg_slice(
            slice_stats
        )
    )


    slice_idx = (
        selected[
            "slice_idx"
        ]
    )


    print(
        "[SELECTED] max-FP over-segmentation slice:",
        slice_idx
    )


    print(
        "           TP / FP / FN:",
        selected["tp"],
        "/",
        selected["fp"],
        "/",
        selected["fn"]
    )


    print(
        "           GT / Pred area:",
        selected["gt_area"],
        "/",
        selected["pred_area"]
    )


    print(
        "           Slice Pred/GT:",
        f"{selected['pred_gt_ratio']:.4f}"
    )


    print(
        "           Slice PPV:",
        f"{selected['ppv']:.4f}"
    )


    # --------------------------------------------------------
    # Construct displayed slice
    # --------------------------------------------------------

    # Keep same representative-figure convention:
    # image channel 0 at selected depth slice.
    mri = (
        image[
            0,
            slice_idx
        ]
        .cpu()
        .numpy()
    )


    mri = (
        normalize_mri(
            mri
        )
    )


    valid_slice = (
        target_np[
            slice_idx
        ] != 255
    )


    gt_show = (
        (
            target_np[
                slice_idx
            ] == 1
        )
        & valid_slice
    ).astype(
        np.uint8
    )


    pred_show = (
        (
            pred[
                slice_idx
            ] == 1
        )
        & valid_slice
    ).astype(
        np.uint8
    )


    patient = str(
        row[
            "patient"
        ]
    )


    sequence = str(
        row[
            "sequence"
        ]
    )


    filename = str(
        row[
            "filename"
        ]
    )


    case_dice = float(
        row[
            "dice"
        ]
    )


    case_iou = float(
        row[
            "iou"
        ]
    )


    case_ppv = float(
        row[
            "ppv"
        ]
    )


    volume_gt = int(
        row[
            "gt_positive_pixels"
        ]
    )


    volume_pred = int(
        row[
            "pred_positive_pixels"
        ]
    )


    volume_ratio = (
        volume_pred
        / volume_gt
        if volume_gt > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Individual 4-panel figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            16,
            4.7
        )
    )


    draw_case(
        axes,
        mri,
        gt_show,
        pred_show
    )


    legend_handles = [
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
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        frameon=True
    )


    fig.suptitle(
        (
            f"Representative Over-segmentation Case {rank}\n"
            f"{patient} | {sequence} | index={idx}\n"
            f"Volume-level diagnostic: "
            f"DSC={case_dice:.4f} | "
            f"IoU={case_iou:.4f} | "
            f"PPV={case_ppv:.4f} | "
            f"Pred/GT={volume_ratio:.2f}\n"
            f"Displayed max-FP slice={slice_idx}: "
            f"TP={selected['tp']} | "
            f"FP={selected['fp']} | "
            f"FN={selected['fn']} | "
            f"Pred/GT={selected['pred_gt_ratio']:.2f}"
        ),
        fontsize=12,
        y=0.995
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.82
        ]
    )


    safe_patient = (
        patient
        .replace(
            "\\",
            "_"
        )
        .replace(
            "/",
            "_"
        )
    )


    safe_sequence = (
        sequence
        .replace(
            "+",
            "_plus_"
        )
        .replace(
            "\\",
            "_"
        )
        .replace(
            "/",
            "_"
        )
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


    plt.close(
        fig
    )


    print(
        "[SAVED]",
        individual_path
    )


    # --------------------------------------------------------
    # Save case for combined figure
    # --------------------------------------------------------

    visual_cases.append({
        "rank":
            rank,

        "index":
            idx,

        "patient":
            patient,

        "sequence":
            sequence,

        "mri":
            mri,

        "gt":
            gt_show,

        "pred":
            pred_show,

        "slice_idx":
            slice_idx,

        "volume_ppv":
            case_ppv,

        "volume_pred_gt_ratio":
            volume_ratio,

        "slice_fp":
            selected["fp"],
    })


    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    summary_rows.append({

        "rank":
            rank,

        "index":
            idx,

        "patient":
            patient,

        "sequence":
            sequence,

        "filename":
            filename,

        "case_level_dice":
            case_dice,

        "case_level_iou":
            case_iou,

        "case_level_ppv":
            case_ppv,

        "volume_gt_positive_pixels":
            volume_gt,

        "volume_pred_positive_pixels":
            volume_pred,

        "volume_pred_gt_ratio":
            volume_ratio,

        "selected_slice_idx":
            slice_idx,

        "slice_tp":
            selected[
                "tp"
            ],

        "slice_fp":
            selected[
                "fp"
            ],

        "slice_fn":
            selected[
                "fn"
            ],

        "slice_gt_area":
            selected[
                "gt_area"
            ],

        "slice_pred_area":
            selected[
                "pred_area"
            ],

        "slice_extra_area":
            selected[
                "extra_area"
            ],

        "slice_pred_gt_ratio":
            selected[
                "pred_gt_ratio"
            ],

        "slice_dice":
            selected[
                "dice"
            ],

        "slice_iou":
            selected[
                "iou"
            ],

        "slice_ppv":
            selected[
                "ppv"
            ],

        "selection_rule":
            (
                "max FP among slices with "
                "GT>0, TP>0, PredArea>GTArea"
            ),

        "individual_figure":
            str(
                individual_path
            ),

        "slice_stats_csv":
            str(
                slice_stats_path
            ),
    })


# ============================================================
# 19. COMBINED 3 x 4 = 12-PANEL MANUSCRIPT FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(
        15.5,
        11.2
    )
)


for row_idx, case in enumerate(
    visual_cases
):

    draw_case(
        axes[
            row_idx
        ],
        case[
            "mri"
        ],
        case[
            "gt"
        ],
        case[
            "pred"
        ]
    )


    # Only top row retains column titles.
    if row_idx > 0:

        for col_idx in range(
            4
        ):

            axes[
                row_idx,
                col_idx
            ].set_title(
                ""
            )


    axes[
        row_idx,
        0
    ].text(

        -0.08,
        0.5,

        (
            f"Case {case['rank']}\n"
            f"index={case['index']}\n"
            f"{case['patient']}\n"
            f"max-FP slice={case['slice_idx']}"
        ),

        transform=axes[
            row_idx,
            0
        ].transAxes,

        ha="right",
        va="center",
        fontsize=10.5
    )


# Explicit clean top-row titles
top_titles = [
    "MRI",
    "Ground Truth",
    "Prediction",
    "Overlay",
]


for col_idx, title in enumerate(
    top_titles
):

    axes[
        0,
        col_idx
    ].set_title(
        title,
        fontsize=13
    )


legend_handles = [

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
    handles=legend_handles,
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


combined_path = (
    OUT_DIR
    / "top3_overseg_maxFP_combined_clean.png"
)


fig.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 20. SAVE SUMMARY CSV
# ============================================================

summary_df = (
    pd.DataFrame(
        summary_rows
    )
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
# 21. FINAL REPORT
# ============================================================

print("\n==============================================")
print("SUCCESS")
print("----------------------------------------------")
print("Formal architecture recovered from checkpoint:")
print(
    f"  in_channels={IN_CHANNELS}, "
    f"num_classes={NUM_CLASSES}, "
    f"base_c={BASE_C}"
)
print("")
print("No training code was modified.")
print("No retraining was performed.")
print("Formal best checkpoint was reused.")
print("Formal test dataset was reused.")
print("All three locked cases passed re-inference validation.")
print("")
print("Output folder:")
print(OUT_DIR)
print("")
print("Combined 12-panel figure:")
print(combined_path)
print("")
print("Evidence summary CSV:")
print(summary_path)
print("==============================================")
