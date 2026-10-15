from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict


ROOT = Path((SEG3_DATA))

SPLITS = ["train", "val", "test"]


def analyze_volume(arr):
    """
    Input shape expected: [H, W, D] = [369, 369, 8]

    Returns:
        zero_indices: indices of fully-zero slices
        pattern: Front / Back / Both / Internal / None / AllZero
    """

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    depth = arr.shape[2]

    zero_indices = []

    for i in range(depth):
        if np.all(arr[:, :, i] == 0):
            zero_indices.append(i)

    if len(zero_indices) == 0:
        pattern = "None"

    elif len(zero_indices) == depth:
        pattern = "AllZero"

    else:
        front_count = 0
        for i in range(depth):
            if i in zero_indices:
                front_count += 1
            else:
                break

        back_count = 0
        for i in range(depth - 1, -1, -1):
            if i in zero_indices:
                back_count += 1
            else:
                break

        edge_zero_indices = (
            list(range(front_count))
            + list(range(depth - back_count, depth))
        )

        edge_zero_indices = sorted(set(edge_zero_indices))

        # Are all zero slices located only at the edges?
        only_edges = sorted(zero_indices) == edge_zero_indices

        if only_edges:
            if front_count > 0 and back_count > 0:
                pattern = "Both"
            elif front_count > 0:
                pattern = "Front"
            elif back_count > 0:
                pattern = "Back"
            else:
                pattern = "None"
        else:
            pattern = "Internal"

    return zero_indices, pattern


print("\n" + "=" * 80)
print("3D ZERO-PADDING SLICE AUDIT")
print("=" * 80)

overall_pattern_counter = Counter()
overall_zero_count_counter = Counter()
overall_volumes = 0


for split in SPLITS:

    image_root = ROOT / split / "images"

    npy_files = sorted(image_root.rglob("*.npy"))

    print("\n" + "=" * 80)
    print(f"{split.upper()}")
    print("=" * 80)

    print(f"Total volumes: {len(npy_files)}")

    pattern_counter = Counter()
    zero_count_counter = Counter()

    examples = defaultdict(list)

    for path in npy_files:

        arr = np.load(path)

        if arr.shape != (369, 369, 8):
            print(f"[WARNING] Unexpected shape: {path} -> {arr.shape}")
            continue

        zero_indices, pattern = analyze_volume(arr)

        n_zero = len(zero_indices)

        pattern_counter[pattern] += 1
        zero_count_counter[n_zero] += 1

        overall_pattern_counter[pattern] += 1
        overall_zero_count_counter[n_zero] += 1

        overall_volumes += 1

        # Keep a few examples for each pattern
        if len(examples[pattern]) < 5:
            examples[pattern].append(
                (
                    path.relative_to(image_root),
                    zero_indices
                )
            )

    print("\nPadding pattern:")
    for pattern in [
        "None",
        "Front",
        "Back",
        "Both",
        "Internal",
        "AllZero"
    ]:
        count = pattern_counter[pattern]

        if len(npy_files) > 0:
            pct = 100 * count / len(npy_files)
        else:
            pct = 0

        print(
            f"{pattern:10s}: "
            f"{count:4d} / {len(npy_files):4d} "
            f"({pct:6.2f}%)"
        )

    print("\nNumber of zero slices per volume:")

    for n_zero in sorted(zero_count_counter):
        count = zero_count_counter[n_zero]
        pct = 100 * count / len(npy_files)

        print(
            f"{n_zero} zero slice(s): "
            f"{count:4d} "
            f"({pct:6.2f}%)"
        )

    print("\nRepresentative examples:")

    for pattern, items in examples.items():

        print(f"\n[{pattern}]")

        for relative_path, zero_indices in items:
            print(
                f"{relative_path} "
                f"-> zero slices = {zero_indices}"
            )


print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

print(f"Total audited volumes: {overall_volumes}")

print("\nOverall padding pattern:")

for pattern in [
    "None",
    "Front",
    "Back",
    "Both",
    "Internal",
    "AllZero"
]:
    count = overall_pattern_counter[pattern]

    pct = (
        100 * count / overall_volumes
        if overall_volumes > 0
        else 0
    )

    print(
        f"{pattern:10s}: "
        f"{count:4d} / {overall_volumes:4d} "
        f"({pct:6.2f}%)"
    )

print("\nOverall zero-slice count:")

for n_zero in sorted(overall_zero_count_counter):

    count = overall_zero_count_counter[n_zero]

    pct = 100 * count / overall_volumes

    print(
        f"{n_zero} zero slice(s): "
        f"{count:4d} "
        f"({pct:6.2f}%)"
    )

print("\nAudit complete.")