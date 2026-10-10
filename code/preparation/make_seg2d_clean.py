"""Copy existing 2D segmentation splits using the study's patient exclusions.

Python standard library only. See README.md for the required input layout.
"""
import argparse
from pathlib import Path
import shutil

SPLITS = ("train", "val", "test")
EXCLUDED_TEST_PATIENTS = {"BreaDM-Be-1801", "BreaDM-Be-1803", "BreaDM-Be-1804"}


def prepare(source, output, dry_run=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories.")

    # Validate every split before creating any output directories.
    plan = {}
    for split in SPLITS:
        patients = {}
        for kind in ("images", "labels"):
            folder = source / split / kind
            if not folder.is_dir():
                raise ValueError(f"Required input directory is missing: {folder}")
            patients[kind] = {p.name for p in folder.iterdir() if p.is_dir()}
        excluded = EXCLUDED_TEST_PATIENTS if split == "test" else set()
        images = patients["images"] - excluded
        labels = patients["labels"] - excluded
        if images != labels:
            raise ValueError(
                f"{split}: unmatched patient directories; "
                f"missing labels={sorted(images - labels)}, "
                f"missing images={sorted(labels - images)}"
            )
        plan[split] = sorted(images)
        removed = sorted((patients["images"] | patients["labels"]) & excluded)
        print(f"{split}: {len(images)} patient pairs; excluded={removed}")

    print(f"Source: {source}\nOutput: {output}")
    if dry_run:
        print("Input check passed. Dry run: no files copied.")
        return

    output.mkdir(parents=True, exist_ok=False)
    for split, patients in plan.items():
        for kind in ("images", "labels"):
            destination = output / split / kind
            destination.mkdir(parents=True)
            for patient in patients:
                shutil.copytree(source / split / kind / patient, destination / patient)
    print("2D preparation completed. Source files were not modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Existing seg directory containing train, val and test")
    parser.add_argument("--output", required=True, type=Path,
                        help="New destination, normally data/segmentation_2d/seg2D_clean_v2")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without copying")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.dry_run)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
