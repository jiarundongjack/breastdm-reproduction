import os
import pandas as pd
import numpy as np

run = r"D:\OneDrive\Desktop\Breast-cancer-dataset-master\Segmentation task\unet\results\seg3d_unet_locked_volume_ppv_seed42_20260911"
csv_path = os.path.join(run, "test_predictions.csv")

df = pd.read_csv(csv_path)

# 转换为数值
num_cols = [
    "index", "dice", "iou", "ppv",
    "gt_positive_pixels", "pred_positive_pixels"
]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 去掉无有效GT的情况
cand = df[
    (df["gt_positive_pixels"] > 0) &
    (df["pred_positive_pixels"] > 0) &
    (df["dice"] > 0)
].copy()

# Pred / GT ratio
cand["pred_gt_ratio"] = (
    cand["pred_positive_pixels"] /
    cand["gt_positive_pixels"]
)

# 只保留预测区域比GT大的病例
cand = cand[
    cand["pred_gt_ratio"] > 1.0
].copy()

# 为便于理解，根据PPV估算 TP / FP / FN
cand["tp_est"] = cand["ppv"] * cand["pred_positive_pixels"]
cand["fp_est"] = cand["pred_positive_pixels"] - cand["tp_est"]
cand["fn_est"] = cand["gt_positive_pixels"] - cand["tp_est"]

# 优先寻找：
# 1. Pred明显大于GT
# 2. PPV低
# 3. 但Dice不能为0 —— 避免纯定位失败
cand["overseg_score"] = (
    cand["pred_gt_ratio"] *
    (1.0 - cand["ppv"])
)

cand = cand.sort_values(
    ["overseg_score", "pred_gt_ratio"],
    ascending=False
)

show_cols = [
    "index",
    "patient",
    "sequence",
    "filename",
    "dice",
    "iou",
    "ppv",
    "gt_positive_pixels",
    "pred_positive_pixels",
    "pred_gt_ratio",
    "tp_est",
    "fp_est",
    "fn_est",
    "overseg_score"
]

top20 = cand[show_cols].head(20)

print("\nTOP 20 OVER-SEGMENTATION CANDIDATES\n")
print(top20.to_string(index=False))

save_path = os.path.join(
    run,
    "top20_overseg_candidates.csv"
)

top20.to_csv(save_path, index=False)

print("\nSaved:")
print(save_path)
