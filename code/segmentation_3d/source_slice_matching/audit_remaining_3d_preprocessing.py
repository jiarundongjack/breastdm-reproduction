from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
from pathlib import Path
import re
import numpy as np
from PIL import Image


DESKTOP = REPO.parent
BREASTDM = REPO

SEG3D = SEG3_DATA

# ------------------------------------------------------------
# Automatically locate the 2D source dataset
# ------------------------------------------------------------
candidates = [SEG2_DATA]

SEG2D = None

for p in candidates:
    if (p / "train" / "images").exists():
        SEG2D = p
        break

print("=" * 85)
print("3D PREPROCESSING REMAINING AUDIT")
print("=" * 85)

print("\n3D root:")
print(SEG3D)

print("\nDetected 2D source:")
print(SEG2D)


# ============================================================
# Helper functions
# ============================================================

def natural_key(path):
    parts = re.split(r"(\d+)", path.name)
    return [
        int(x) if x.isdigit() else x.lower()
        for x in parts
    ]


def load_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def load_3d_slices(path):
    arr = np.load(path)

    if arr.shape != (369, 369, 8):
        raise ValueError(
            f"Unexpected 3D shape: {path} -> {arr.shape}"
        )

    return arr.astype(np.float32)


def get_nonzero_indices(volume):
    return [
        i for i in range(volume.shape[2])
        if not np.all(volume[:, :, i] == 0)
    ]


def match_slices(volume, source_images):
    """
    For each non-zero slice in the 3D volume,
    find the closest corresponding 2D source image.
    """

    nonzero_indices = get_nonzero_indices(volume)

    source_arrays = [
        load_gray(p)
        for p in source_images
    ]

    matches = []

    for d in nonzero_indices:

        slice3d = volume[:, :, d]

        best_idx = None
        best_mae = float("inf")

        for j, src in enumerate(source_arrays):

            if src.shape != slice3d.shape:
                continue

            mae = np.mean(
                np.abs(slice3d - src)
            )

            if mae < best_mae:
                best_mae = mae
                best_idx = j

        matches.append(
            (d, best_idx, best_mae)
        )

    return matches


# ============================================================
# PART 1
# Check central-8 slice selection
# ============================================================

print("\n")
print("=" * 85)
print("PART 1 — CENTRAL 8-SLICE SELECTION")
print("=" * 85)

if SEG2D is None:

    print(
        "\n[UNVERIFIABLE] "
        "Could not locate the corresponding 2D source dataset."
    )

else:

    total_3d = 0
    matched_volumes = 0

    more_than_8 = 0
    central_verified = 0
    central_failed = 0

    fewer_equal_8 = 0
    source_not_found = 0

    failed_examples = []

    for split in ["train", "val", "test"]:

        image_root_3d = SEG3D / split / "images"

        for volume_path in sorted(
            image_root_3d.rglob("*.npy")
        ):

            total_3d += 1

            patient = volume_path.parent.name
            sequence = volume_path.stem

            source_dir = (
                SEG2D
                / split
                / "images"
                / patient
                / sequence
            )

            if not source_dir.exists():
                source_not_found += 1
                continue

            source_images = sorted(
                [
                    p for p in source_dir.iterdir()
                    if p.suffix.lower()
                    in [".jpg", ".jpeg", ".png", ".bmp"]
                ],
                key=natural_key
            )

            if len(source_images) == 0:
                source_not_found += 1
                continue

            volume = load_3d_slices(
                volume_path
            )

            matches = match_slices(
                volume,
                source_images
            )

            # Need reasonably exact matching
            if (
                len(matches) == 0
                or any(
                    idx is None or mae > 0.01
                    for _, idx, mae in matches
                )
            ):
                continue

            matched_volumes += 1

            matched_indices = [
                idx
                for _, idx, _ in matches
            ]

            n_src = len(source_images)

            if n_src > 8:

                more_than_8 += 1

                # Two possible centered windows when the
                # number of extra slices is odd.
                start_floor = (n_src - 8) // 2
                start_ceil = (
                    n_src - 8 + 1
                ) // 2

                expected_floor = list(
                    range(
                        start_floor,
                        start_floor + 8
                    )
                )

                expected_ceil = list(
                    range(
                        start_ceil,
                        start_ceil + 8
                    )
                )

                if (
                    matched_indices == expected_floor
                    or matched_indices == expected_ceil
                ):
                    central_verified += 1

                else:
                    central_failed += 1

                    if len(failed_examples) < 10:
                        failed_examples.append(
                            (
                                split,
                                patient,
                                sequence,
                                n_src,
                                matched_indices,
                                expected_floor,
                                expected_ceil,
                            )
                        )

            else:
                fewer_equal_8 += 1

    print(f"\nTotal 3D volumes             : {total_3d}")
    print(f"Volumes matched to 2D source: {matched_volumes}")
    print(f"Source folder not found      : {source_not_found}")

    print("\nFor source sequences with >8 slices:")
    print(f"Total                        : {more_than_8}")
    print(f"Central-8 verified           : {central_verified}")
    print(f"Central-8 not matched        : {central_failed}")

    if more_than_8 > 0:

        pct = (
            100
            * central_verified
            / more_than_8
        )

        print(
            f"Central-selection agreement  : "
            f"{pct:.2f}%"
        )

    if failed_examples:

        print("\nExamples not matching central selection:")

        for item in failed_examples:

            (
                split,
                patient,
                sequence,
                n_src,
                actual,
                expected1,
                expected2,
            ) = item

            print(
                f"{split} | "
                f"{patient} | "
                f"{sequence} | "
                f"N={n_src} | "
                f"actual={actual} | "
                f"expected={expected1}/{expected2}"
            )


# ============================================================
# PART 2
# Search for 1×1×1 mm spatial-normalization evidence
# ============================================================

print("\n")
print("=" * 85)
print("PART 2 — 1×1×1 mm³ SPATIAL NORMALIZATION")
print("=" * 85)

keywords = [
    "SimpleITK",
    "sitk.",
    "GetSpacing",
    "SetSpacing",
    "spacing",
    "resample",
    "resampling",
    "ResampleImageFilter",
    "scipy.ndimage.zoom",
    "ndimage.zoom",
    "1.0, 1.0, 1.0",
    "(1, 1, 1)",
]

evidence = []

for py_file in (REPO / "code").rglob("*.py"):

    try:
        lines = py_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    except Exception:
        continue

    for line_no, line in enumerate(
        lines,
        start=1
    ):

        lower = line.lower()

        if any(
            keyword.lower() in lower
            for keyword in keywords
        ):

            evidence.append(
                (
                    py_file,
                    line_no,
                    line.strip()
                )
            )


if evidence:

    print(
        "\nPotential spatial-resampling evidence found:"
    )

    for path, line_no, line in evidence[:50]:

        print(
            f"{path}\n"
            f"  Line {line_no}: {line}"
        )

else:

    print(
        "\nNo Python implementation evidence of "
        "1×1×1 mm³ spatial resampling was found."
    )


# ------------------------------------------------------------
# Check whether current data still contains physical-spacing
# metadata-capable files.
# ------------------------------------------------------------

metadata_extensions = {
    ".dcm",
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
    ".nrrd",
}

metadata_files = []

for p in (REPO / "data").rglob("*"):

    if not p.is_file():
        continue

    name_lower = p.name.lower()

    if (
        p.suffix.lower() in metadata_extensions
        or name_lower.endswith(".nii.gz")
    ):
        metadata_files.append(p)


print("\nFiles capable of storing physical voxel spacing:")
print(f"Count: {len(metadata_files)}")

for p in metadata_files[:20]:
    print(p)


print("\n" + "=" * 85)
print("AUDIT COMPLETE")
print("=" * 85)

print(
    "\nInterpretation:"
)

print(
    "1) If Central-selection agreement is close to 100%, "
    "the central-8 construction is verified."
)

print(
    "2) If no resampling code and no DICOM/NIfTI-like files "
    "are found, 1×1×1 mm³ spatial normalization cannot be "
    "independently verified from the current processed JPG/NPY data."
)