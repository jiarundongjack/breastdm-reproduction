from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import pandas as pd

root = (RESULT)

seeds = [48271, 59317, 84629]

all_rows = []

for seed in seeds:
    path = os.path.join(
        root,
        f"fusion_exp2_100ep_seed{seed}_final_multiseed",
        "test_predictions_two_class.csv"
    )

    df = pd.read_csv(path)

    # 标记随机种子
    df["Seed"] = seed

    # 确保概率是数值
    df["Malignant Probability"] = pd.to_numeric(
        df["Malignant Probability"], errors="coerce"
    )

    # False Positive:
    # True = Benign (0), Predicted = Malignant (1)
    fp = df[
        (df["True Label"] == 0) &
        (df["Predicted Label"] == 1)
    ].copy()

    all_rows.append(fp)

all_fp = pd.concat(all_rows, ignore_index=True)

# ==========================================================
# 1. 所有单次运行中 malignant probability 最高的 FP Top 10
# ==========================================================

top10_single = (
    all_fp
    .sort_values("Malignant Probability", ascending=False)
    .head(10)
)

single_cols = [
    "Seed",
    "Index",
    "True Label",
    "Predicted Label",
    "Benign Probability",
    "Malignant Probability"
]

print("\n============================================")
print("TOP 10 FALSE POSITIVES - SINGLE RUN")
print("============================================")
print(top10_single[single_cols].to_string(index=False))

top10_single[single_cols].to_csv(
    os.path.join(OUTPUT, "exp2_top10_false_positive_single_run.csv"),
    index=False
)

# ==========================================================
# 2. 三个 seed 都误判为 malignant 的 persistent FP
# ==========================================================

grouped = (
    all_fp
    .groupby("Index")
    .agg(
        FP_Count=("Seed", "count"),
        Mean_Malignant_Probability=("Malignant Probability", "mean"),
        Max_Malignant_Probability=("Malignant Probability", "max"),
        Min_Malignant_Probability=("Malignant Probability", "min")
    )
    .reset_index()
)

# 三个 seed 均为假阳性
persistent = grouped[grouped["FP_Count"] == 3].copy()

persistent = (
    persistent
    .sort_values("Mean_Malignant_Probability", ascending=False)
    .head(10)
)

print("\n============================================")
print("TOP 10 PERSISTENT FALSE POSITIVES")
print("MISCLASSIFIED AS MALIGNANT IN ALL 3 SEEDS")
print("============================================")
print(persistent.to_string(index=False))

persistent.to_csv(
    os.path.join(OUTPUT, "exp2_top10_persistent_false_positive.csv"),
    index=False
)

# ==========================================================
# 3. 把 persistent Top 10 的三个 seed 具体概率全部列出来
# ==========================================================

persistent_ids = persistent["Index"].tolist()

details = all_fp[
    all_fp["Index"].isin(persistent_ids)
].copy()

details = details.sort_values(
    ["Index", "Seed"]
)

detail_cols = [
    "Index",
    "Seed",
    "True Label",
    "Predicted Label",
    "Benign Probability",
    "Malignant Probability"
]

details[detail_cols].to_csv(
    os.path.join(OUTPUT, "exp2_top10_persistent_false_positive_details.csv"),
    index=False
)

print("\n============================================")
print("DETAILS FOR PERSISTENT TOP 10")
print("============================================")
print(details[detail_cols].to_string(index=False))

print("\nSaved:")
print("exp2_top10_false_positive_single_run.csv")
print("exp2_top10_persistent_false_positive.csv")
print("exp2_top10_persistent_false_positive_details.csv")
