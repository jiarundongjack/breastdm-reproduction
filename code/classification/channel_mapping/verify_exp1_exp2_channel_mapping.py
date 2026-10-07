from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# 数据根目录
# =========================================================
base_dir = Path(__file__).resolve().parent

exp1_root = EXP1_DATA / "test"
exp2_root = Path((EXP2_DATA / 'test'))


# =========================================================
# 建立 Relative Path -> File Path 映射
# =========================================================
def collect_files(root):
    mapping = {}

    for file_path in root.rglob("*.npy"):
        relative_path = file_path.relative_to(root)

        mapping[str(relative_path)] = file_path

    return mapping


exp1_files = collect_files(exp1_root)
exp2_files = collect_files(exp2_root)

common_files = sorted(
    set(exp1_files.keys()) &
    set(exp2_files.keys())
)


print("Exp-1 test files:", len(exp1_files))
print("Exp-2 test files:", len(exp2_files))
print("Common files:", len(common_files))


# =========================================================
# 逐文件、逐像素验证
#
# 假设：
# Exp-1 = H × W × 9
# Exp-2 = H × W × 17
#
# 检查：
# Exp-2 channels 8:17
# 是否与
# Exp-1 channels 0:9
# 完全一致
# =========================================================
rows = []

total_matched = 0
total_failed = 0


for relative_path in common_files:

    exp1_path = exp1_files[relative_path]
    exp2_path = exp2_files[relative_path]

    arr1 = np.load(exp1_path)
    arr2 = np.load(exp2_path)

    # Shape 检查
    valid_shape = (
        arr1.ndim == 3
        and arr2.ndim == 3
        and arr1.shape[-1] == 9
        and arr2.shape[-1] == 17
        and arr1.shape[:2] == arr2.shape[:2]
    )

    if not valid_shape:

        rows.append({
            "Relative Path": relative_path,
            "Exp1 Shape": str(arr1.shape),
            "Exp2 Shape": str(arr2.shape),
            "Pixel Exact Match": False,
            "Max Abs Difference": np.nan
        })

        total_failed += 1
        continue


    # Exp-2 后9通道
    exp2_last9 = arr2[:, :, 8:17]

    # 与 Exp-1 全9通道逐像素比较
    exact_match = np.array_equal(
        arr1,
        exp2_last9
    )

    # 避免 uint8 相减溢出，先转成 int16
    difference = np.abs(
        arr1.astype(np.int16)
        -
        exp2_last9.astype(np.int16)
    )

    max_abs_difference = int(
        difference.max()
    )

    if exact_match:
        total_matched += 1
    else:
        total_failed += 1

    rows.append({
        "Relative Path": relative_path,
        "Exp1 Shape": str(arr1.shape),
        "Exp2 Shape": str(arr2.shape),
        "Pixel Exact Match": exact_match,
        "Max Abs Difference": max_abs_difference
    })


# =========================================================
# 保存结果
# =========================================================
result_df = pd.DataFrame(rows)

output_dir = (
    OUTPUT
    / "fp_fn_cases"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir
    / "exp1_exp2_channel_mapping_verification.csv"
)

result_df.to_csv(
    output_path,
    index=False
)


# =========================================================
# 终端总结
# =========================================================
print("\n========== Channel Mapping Verification ==========\n")

print(
    "Hypothesis:"
    "\nExp-2[:, :, 8:17] == Exp-1[:, :, 0:9]"
)

print("\nFiles checked:", len(common_files))
print("Exact matched:", total_matched)
print("Failed:", total_failed)

if len(common_files) > 0:
    match_rate = total_matched / len(common_files)

    print(
        f"Exact match rate: {match_rate:.2%}"
    )


if (
    len(common_files) > 0
    and total_failed == 0
):

    print(
        "\n[CONFIRMED]"
        "\nFor every common test sample:"
        "\nExp-2 channels 8-16 are pixel-wise identical "
        "to Exp-1 channels 0-8."
    )

else:

    print(
        "\n[NOT FULLY CONFIRMED]"
        "\nSome files are different."
    )

    failed_df = result_df[
        result_df["Pixel Exact Match"] == False
    ]

    print(
        failed_df.head(20).to_string(
            index=False
        )
    )


print("\nSaved to:")
print(output_path)