from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
from pathlib import Path
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

PROJECT = Path.cwd()
UNET_DIR = RUNTIME

sys.path.insert(0, str(UNET_DIR))

from src.unet3d import UNet


SEG2D = Path((SEG2_DATA))
SEG3D = Path((SEG3_DATA))

MODEL_PATH = RUN3 / "best_model.pth"

split = "train"
patient = "BreaDM-Ma-1904"
sequence = "SUB2"

VOLUME_PATH = (
    SEG3D / split / "images" / patient / f"{sequence}.npy"
)

MASK_PATH = (
    SEG3D / split / "labels" / patient / f"{sequence}.npy"
)

RGB_SOURCE_PATH = (
    SEG2D
    / split
    / "images"
    / patient
    / sequence
    / "p-038.jpg"
)


# ============================================================
# PREPROCESSING
# identical to our 3D pipeline:
# 1. H,W,D -> D,H,W
# 2. resize H,W to 224x224
# 3. 0.1%-99.9% clipping
# 4. volume-wise z-score using valid non-zero slices
# 5. padding slices remain zero
# ============================================================

def breastdm_zscore_3d(image):

    # image: [1, D, H, W]

    valid_slices = (
        torch.count_nonzero(
            image[0],
            dim=(1, 2)
        ) > 0
    )

    if not torch.any(valid_slices):
        return image

    valid_data = image[0, valid_slices]

    low = torch.quantile(valid_data, 0.001)
    high = torch.quantile(valid_data, 0.999)

    valid_data = torch.clamp(
        valid_data,
        min=low,
        max=high
    )

    mean = valid_data.mean()
    std = valid_data.std()

    if std > 0:
        valid_data = (valid_data - mean) / std

    result = image.clone()

    result[0, valid_slices] = valid_data
    result[0, ~valid_slices] = 0

    return result


def preprocess(volume, mask):

    # [H,W,D] -> [D,H,W]
    volume = np.transpose(volume, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))

    image = torch.from_numpy(
        volume.astype(np.float32)
    ).unsqueeze(0)

    target = torch.from_numpy(
        (mask > 0).astype(np.int64)
    )

    # [1,D,H,W] -> treat D as batch during spatial resizing
    image = F.interpolate(
        image.permute(1, 0, 2, 3),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).permute(1, 0, 2, 3)

    target = F.interpolate(
        target.unsqueeze(1).float(),
        size=(224, 224),
        mode="nearest"
    ).squeeze(1).long()

    image = breastdm_zscore_3d(image)

    return image, target


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, target):

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    iou_bg = (
        tn / (tn + fp + fn)
        if (tn + fp + fn) > 0
        else 0
    )

    iou_tumor = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0
    )

    miou = (iou_bg + iou_tumor) / 2

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "DSC": dice,
        "mIoU": miou,
        "PPV": ppv,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 85)
print("RGB INTENSITY DISCREPANCY — FINAL IMPACT TEST")
print("=" * 85)

original_volume = np.load(VOLUME_PATH).astype(np.float32)
mask = np.load(MASK_PATH)

corrected_volume = original_volume.copy()

rgb_image = Image.open(RGB_SOURCE_PATH)

print(f"\nSource image mode : {rgb_image.mode}")

rgb_array = np.asarray(rgb_image)

if rgb_image.mode == "RGB":

    channel_equal = (
        np.array_equal(rgb_array[:, :, 0], rgb_array[:, :, 1])
        and
        np.array_equal(rgb_array[:, :, 0], rgb_array[:, :, 2])
    )

    print(f"RGB channels identical: {channel_equal}")

gray = np.asarray(
    rgb_image.convert("L"),
    dtype=np.float32
)

# p-038 corresponds to slice index 3
corrected_volume[:, :, 3] = gray


# ============================================================
# RAW INPUT DIFFERENCE
# ============================================================

raw_diff = np.abs(
    original_volume - corrected_volume
)

print("\n" + "=" * 85)
print("RAW 3D VOLUME DIFFERENCE")
print("=" * 85)

print(f"Mean absolute difference : {raw_diff.mean():.8f}")
print(f"Maximum difference       : {raw_diff.max():.8f}")
print(
    f"Changed voxels           : "
    f"{np.count_nonzero(raw_diff)} / {raw_diff.size}"
)


# ============================================================
# PREPROCESS BOTH VERSIONS
# ============================================================

original_input, target = preprocess(
    original_volume,
    mask
)

corrected_input, _ = preprocess(
    corrected_volume,
    mask
)

processed_diff = torch.abs(
    original_input - corrected_input
)

print("\n" + "=" * 85)
print("DIFFERENCE AFTER MODEL PREPROCESSING")
print("=" * 85)

print(
    f"Mean absolute difference : "
    f"{processed_diff.mean().item():.8f}"
)

print(
    f"Maximum difference       : "
    f"{processed_diff.max().item():.8f}"
)

print(
    f"Changed values           : "
    f"{torch.count_nonzero(processed_diff).item()} / "
    f"{processed_diff.numel()}"
)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")
print(f"Model : {MODEL_PATH}")

model = UNet(
    in_channels=1,
    num_classes=2,
    base_c=32
)

import argparse
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

if isinstance(checkpoint, dict):

    if "model" in checkpoint:
        state_dict = checkpoint["model"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# remove "module." prefix if needed
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(clean_state_dict)

model.to(device)
model.eval()


# ============================================================
# MODEL INFERENCE
# ============================================================

original_input = original_input.unsqueeze(0).to(device)
corrected_input = corrected_input.unsqueeze(0).to(device)

target = target.to(device)

with torch.no_grad():

    out_original = model(original_input)
    out_corrected = model(corrected_input)

    if isinstance(out_original, dict):
        out_original = out_original["out"]

    if isinstance(out_corrected, dict):
        out_corrected = out_corrected["out"]

    prob_original = torch.softmax(
        out_original,
        dim=1
    )[:, 1]

    prob_corrected = torch.softmax(
        out_corrected,
        dim=1
    )[:, 1]

    pred_original = torch.argmax(
        out_original,
        dim=1
    )[0]

    pred_corrected = torch.argmax(
        out_corrected,
        dim=1
    )[0]


# ============================================================
# COMPARE PROBABILITIES
# ============================================================

prob_diff = torch.abs(
    prob_original - prob_corrected
)

print("\n" + "=" * 85)
print("MODEL PROBABILITY DIFFERENCE")
print("=" * 85)

print(
    f"Mean tumor-probability difference : "
    f"{prob_diff.mean().item():.10f}"
)

print(
    f"Maximum probability difference    : "
    f"{prob_diff.max().item():.10f}"
)


# ============================================================
# COMPARE HARD SEGMENTATION
# ============================================================

prediction_difference = (
    pred_original != pred_corrected
)

changed_pred_voxels = (
    prediction_difference.sum().item()
)

total_pred_voxels = (
    prediction_difference.numel()
)

print("\n" + "=" * 85)
print("FINAL SEGMENTATION PREDICTION COMPARISON")
print("=" * 85)

print(
    f"Changed prediction voxels : "
    f"{changed_pred_voxels} / {total_pred_voxels}"
)

print(
    f"Prediction difference rate: "
    f"{100 * changed_pred_voxels / total_pred_voxels:.8f}%"
)

print(
    f"Original predicted tumor voxels : "
    f"{(pred_original == 1).sum().item()}"
)

print(
    f"Corrected predicted tumor voxels: "
    f"{(pred_corrected == 1).sum().item()}"
)


# ============================================================
# METRIC COMPARISON
# ============================================================

metrics_original = calculate_metrics(
    pred_original,
    target
)

metrics_corrected = calculate_metrics(
    pred_corrected,
    target
)

print("\n" + "=" * 85)
print("SEGMENTATION METRICS")
print("=" * 85)

print("\nOriginal 3D volume:")
for k, v in metrics_original.items():
    if isinstance(v, float):
        print(f"{k:5s}: {v:.8f}")
    else:
        print(f"{k:5s}: {v}")

print("\nRGB-corrected 3D volume:")
for k, v in metrics_corrected.items():
    if isinstance(v, float):
        print(f"{k:5s}: {v:.8f}")
    else:
        print(f"{k:5s}: {v}")


print("\n" + "=" * 85)
print("FINAL CONCLUSION")
print("=" * 85)

if changed_pred_voxels == 0:

    print(
        "HARD SEGMENTATION PREDICTION: IDENTICAL"
    )

    print(
        "The RGB intensity discrepancy does NOT change "
        "the final segmentation prediction for this volume."
    )

    print(
        "Therefore, it does not affect the final 3D "
        "segmentation metrics."
    )

else:

    print(
        "A small prediction difference was detected."
    )

    print(
        "Use the changed-voxel rate and metric differences "
        "above to quantify its impact."
    )