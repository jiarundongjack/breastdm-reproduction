from pathlib import Path
import numpy as np
import pandas as pd

RUN_DIR = Path(
    r"D:\OneDrive\Desktop\Breast-cancer-dataset-master"
    r"\Segmentation task\unet\results"
    r"\seg3d_unet_locked_volume_ppv_seed42_20260911"
)

CSV_PATH = RUN_DIR / "test_predictions.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Cannot find test_predictions.csv:\n{CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = [
    "index",
    "patient",
    "sequence",
    "filename",
    "dice",
    "iou",
    "ppv",
    "gt_positive_pixels",
    "pred_positive_pixels",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(
        "Missing required columns in test_predictions.csv:\n"
        + ", ".join(missing)
    )

df["GT"] = df["gt_positive_pixels"].astype(int)
df["Pred"] = df["pred_positive_pixels"].astype(int)

# Recover TP in two independent ways.
# Dice = 2TP / (GT + Pred)
df["TP_from_dice"] = np.rint(
    df["dice"].astype(float) * (df["GT"] + df["Pred"]) / 2.0
).astype(int)

# PPV = TP / Pred
df["TP_from_ppv"] = np.rint(
    df["ppv"].astype(float) * df["Pred"]
).astype(int)

df["TP_recovery_diff"] = np.abs(
    df["TP_from_dice"] - df["TP_from_ppv"]
)

# Use Dice-derived TP as primary reconstruction.
df["TP"] = np.minimum(
    df["TP_from_dice"],
    np.minimum(df["GT"], df["Pred"])
)

df["FP"] = df["Pred"] - df["TP"]
df["FN"] = df["GT"] - df["TP"]

# Primary condition: GT coverage
df["GT_Coverage"] = np.where(
    df["GT"] > 0,
    df["TP"] / df["GT"],
    np.nan
)

# Secondary condition: PPV
df["PPV_check"] = np.where(
    df["Pred"] > 0,
    df["TP"] / df["Pred"],
    np.nan
)

# Extra audit quantities
df["Pred_GT"] = np.where(
    df["GT"] > 0,
    df["Pred"] / df["GT"],
    np.nan
)

df["FP_over_GT"] = np.where(
    df["GT"] > 0,
    df["FP"] / df["GT"],
    np.nan
)

# Eligible over-segmentation volumes:
# GT exists, true overlap exists, and prediction is larger than GT.
eligible = df[
    (df["GT"] > 0)
    & (df["TP"] > 0)
    & (df["Pred"] > df["GT"])
].copy()

# Two-level ranking:
# 1) maximize TP / GT
# 2) minimize PPV
# Extra fields are only deterministic tie-breakers.
ranked = eligible.sort_values(
    by=["GT_Coverage", "PPV_check", "FP_over_GT", "index"],
    ascending=[False, True, False, True],
).copy()

ranked.insert(0, "VolumeRank", range(1, len(ranked) + 1))
top3 = ranked.head(3).copy()

OUT_DIR = RUN_DIR / "volume_overseg_two_level_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

save_cols = [
    "VolumeRank",
    "index",
    "patient",
    "sequence",
    "filename",
    "GT",
    "Pred",
    "TP",
    "FP",
    "FN",
    "GT_Coverage",
    "PPV_check",
    "Pred_GT",
    "FP_over_GT",
    "dice",
    "iou",
    "TP_recovery_diff",
]

ranked[save_cols].to_csv(
    OUT_DIR / "all_eligible_volumes_ranked.csv",
    index=False
)

top3[save_cols].to_csv(
    OUT_DIR / "TOP3_volumes_two_level_selection.csv",
    index=False
)

print("\n" + "=" * 84)
print("TOP-3 3D VOLUME SELECTION FOR OVER-SEGMENTATION")
print("=" * 84)
print("\nSelection logic:")
print("Primary   : maximize TP / GT (GT coverage)")
print("Secondary : minimize PPV")
print("Eligibility: GT > 0, TP > 0, Pred > GT")
print(f"\nFormal test volumes : {len(df)}")
print(f"Eligible volumes    : {len(eligible)}")
print(
    "Max TP reconstruction difference: "
    f"{int(df['TP_recovery_diff'].max())} voxel(s)"
)

print("\nTOP 3 VOLUMES:\n")

display_cols = [
    "VolumeRank",
    "index",
    "patient",
    "GT",
    "Pred",
    "TP",
    "FP",
    "FN",
    "GT_Coverage",
    "PPV_check",
    "Pred_GT",
]

print(
    top3[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved to:")
print(OUT_DIR)
print("=" * 84 + "\n")