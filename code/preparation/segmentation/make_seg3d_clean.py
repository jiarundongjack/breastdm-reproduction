"""Copy existing 3D segmentation splits, selecting patients from images.

Python standard library only. See README.md for the required input layout.
"""
import argparse
from pathlib import Path
import shutil

SPLITS = ("train", "val", "test")


def prepare(source, output, dry_run=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories.")

    # Match the historical preparation rule: select patients by images only.
    plan = {}
    for split in SPLITS:
        patients = {}
        for kind in ("images", "labels"):
            folder = source / split / kind
            if not folder.is_dir():
                raise ValueError(f"Required input directory is missing: {folder}")
            patients[kind] = {p.name for p in folder.iterdir() if p.is_dir()}
        missing = patients["images"] - patients["labels"]
        if missing:
            raise ValueError(f"{split}: missing label directories for {sorted(missing)}")
        plan[split] = sorted(patients["images"])
        ignored = sorted(patients["labels"] - patients["images"])
        print(f"{split}: {len(plan[split])} patient pairs; unused label directories={ignored}")

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
    print("3D preparation completed. Source files were not modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Existing seg3D directory containing train, val and test")
    parser.add_argument("--output", required=True, type=Path,
                        help="New destination, normally data/segmentation_3d/seg3D_clean")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without copying")
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.dry_run)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
