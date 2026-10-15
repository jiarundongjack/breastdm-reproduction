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
# 收集相同测试文件
# =========================================================
def collect_files(root):
    mapping = {}

    for path in root.rglob("*.npy"):
        rel = str(path.relative_to(root))
        mapping[rel] = path

    return mapping


exp1_files = collect_files(exp1_root)
exp2_files = collect_files(exp2_root)

common_files = sorted(
    set(exp1_files.keys()) &
    set(exp2_files.keys())
)

print("Common test files:", len(common_files))


# =========================================================
# 三种 subtraction 计算方式
# =========================================================
def post_minus_pre(post, pre):
    result = post.astype(np.int16) - pre.astype(np.int16)
    return np.clip(result, 0, 255).astype(np.uint8)


def pre_minus_post(pre, post):
    result = pre.astype(np.int16) - post.astype(np.int16)
    return np.clip(result, 0, 255).astype(np.uint8)


def absolute_difference(a, b):
    result = np.abs(
        a.astype(np.int16) -
        b.astype(np.int16)
    )

    return result.astype(np.uint8)


formulas = {
    "post-pre_clipped": post_minus_pre,
    "pre-post_clipped": pre_minus_post,
    "absolute_difference": absolute_difference
}


# =========================================================
# 对每个 Exp-2 新增通道 0~7
# 尝试 Exp-1 原始9通道所有两两组合
# =========================================================
# Cache the same prepared arrays once; formulas and comparisons are unchanged.
sample_arrays = {rel: (np.load(exp1_files[rel]), np.load(exp2_files[rel]))
                 for rel in common_files}
rows = []

for new_channel in range(8):

    print(
        f"\nChecking Exp-2 new channel {new_channel} ..."
    )

    for formula_name, formula in formulas.items():

        for pre_channel in range(9):

            for post_channel in range(9):

                if pre_channel == post_channel:
                    continue

                exact_files = 0
                total_pixels = 0
                matched_pixels = 0
                max_difference = 0

                for rel in common_files:

                    arr1, arr2 = sample_arrays[rel]

                    pre = arr1[:, :, pre_channel]
                    post = arr1[:, :, post_channel]

                    predicted_sub = formula(
                        post,
                        pre
                    )

                    actual_sub = arr2[
                        :,
                        :,
                        new_channel
                    ]

                    if np.array_equal(
                        predicted_sub,
                        actual_sub
                    ):
                        exact_files += 1

                    difference = np.abs(
                        predicted_sub.astype(np.int16)
                        -
                        actual_sub.astype(np.int16)
                    )

                    matched_pixels += int(
                        np.sum(difference == 0)
                    )

                    total_pixels += difference.size

                    max_difference = max(
                        max_difference,
                        int(difference.max())
                    )

                pixel_match_rate = (
                    matched_pixels /
                    total_pixels
                    if total_pixels > 0
                    else 0
                )

                rows.append({
                    "Exp2 New Channel": new_channel,
                    "Formula": formula_name,
                    "Pre Channel": pre_channel,
                    "Post Channel": post_channel,
                    "Exact Files": exact_files,
                    "Total Files": len(common_files),
                    "File Match Rate": (
                        exact_files / len(common_files)
                    ),
                    "Pixel Match Rate": pixel_match_rate,
                    "Max Abs Difference": max_difference
                })


# =========================================================
# 每个新增通道找最匹配的公式
# =========================================================
df = pd.DataFrame(rows)

best_rows = (
    df.sort_values(
        by=[
            "Exp2 New Channel",
            "Pixel Match Rate",
            "File Match Rate"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .groupby(
        "Exp2 New Channel",
        as_index=False
    )
    .first()
)


print(
    "\n========== Best Subtraction Mapping ==========\n"
)

print(
    best_rows[
        [
            "Exp2 New Channel",
            "Formula",
            "Pre Channel",
            "Post Channel",
            "Exact Files",
            "File Match Rate",
            "Pixel Match Rate",
            "Max Abs Difference"
        ]
    ].to_string(index=False)
)


# =========================================================
# 是否得到 100% 精确对应
# =========================================================
fully_confirmed = (
    (best_rows["File Match Rate"] == 1.0)
    &
    (best_rows["Pixel Match Rate"] == 1.0)
).all()

print("\nFully confirmed:", fully_confirmed)


# =========================================================
# 保存
# =========================================================
output_dir = (
    OUTPUT
    / "fp_fn_cases"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

all_path = (
    output_dir
    / "subtraction_formula_all_candidates.csv"
)

best_path = (
    output_dir
    / "subtraction_formula_best_mapping.csv"
)

df.to_csv(
    all_path,
    index=False
)

best_rows.to_csv(
    best_path,
    index=False
)

print("\nSaved:")
print(all_path)
print(best_path)