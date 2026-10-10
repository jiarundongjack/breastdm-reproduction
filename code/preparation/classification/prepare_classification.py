"""Copy existing 9/17-channel arrays into this study's formal input layout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

SPLITS = ("train", "val", "test")
CLASSES = ("Benign", "Malignant")
EXCLUDED = {"BreaDM-Ma-1802", "BreaDM-Ma-1803", "BreaDM-Ma-1804",
            "BreaDM-Ma-1806", "BreaDM-Ma-1807", "BreaDM-Ma-1808"}
EXPECTED = {"train/Benign": 327, "train/Malignant": 875,
            "val/Benign": 24, "val/Malignant": 93,
            "test/Benign": 114, "test/Malignant": 289}


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def collect(root):
    if not root.is_dir():
        raise ValueError(f"Input directory not found: {root}")
    for split in SPLITS:
        for label in CLASSES:
            if not (root / split / label).is_dir():
                raise ValueError(f"Missing input directory: {root / split / label}")
    files = {}
    for path in sorted(root.rglob("*.npy")):
        rel = path.relative_to(root)
        if len(rel.parts) < 4 or rel.parts[0] not in SPLITS or rel.parts[1] not in CLASSES:
            raise ValueError(f"Unexpected NPY location: {path}")
        if root not in path.resolve().parents:
            raise ValueError(f"Input file resolves outside source: {path}")
        files[rel.as_posix()] = path
    return files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source9, source17, output9, output17, dry_run=False):
    source9, source17, output9, output17 = [Path(p).resolve() for p in (source9, source17, output9, output17)]
    for output in (output9, output17):
        if output.exists():
            raise ValueError(f"Output already exists; choose a new directory: {output}")
        if any(overlap(output, source) for source in (source9, source17)):
            raise ValueError("Outputs must be separate from and not nested with sources")
    if overlap(output9, output17):
        raise ValueError("The two output directories must be separate and non-nested")
    raw9, files17 = collect(source9), collect(source17)
    excluded = {p for p in raw9 if p.split('/')[:2] == ['test', 'Malignant'] and p.split('/')[2] in EXCLUDED}
    files9 = {p: f for p, f in raw9.items() if p not in excluded}
    if files9.keys() != files17.keys():
        raise ValueError("The retained 9-channel and 17-channel relative paths do not match")
    counts = dict(Counter('/'.join(p.split('/')[:2]) for p in files9))
    if counts != EXPECTED:
        raise ValueError(f"Counts differ from this study's formal input counts: {counts}")
    fingerprints = {"exp1": {}, "exp2": {}}
    for rel in files9:
        a = np.load(files9[rel], allow_pickle=False)
        b = np.load(files17[rel], allow_pickle=False)
        if (a.ndim != 3 or b.ndim != 3 or a.shape[-1] != 9 or b.shape[-1] != 17
                or a.shape[:2] != b.shape[:2] or min(a.shape[:2]) < 1
                or a.dtype != np.uint8 or b.dtype != np.uint8):
            raise ValueError(f"Expected paired uint8 HxWx9 / HxWx17 arrays: {rel}")
        if not np.array_equal(a, b[:, :, 8:17]):
            raise ValueError(f"The final nine channels differ: {rel}")
        fingerprints['exp1'][rel] = digest(files9[rel])
        fingerprints['exp2'][rel] = digest(files17[rel])
    report = {"scope": "Organize existing arrays to match the locally verified study layout; not raw-image preprocessing.",
              "counts_per_experiment": counts, "excluded_from_source9": sorted(excluded),
              "paired_arrays_checked": len(files9), "dry_run": dry_run}
    if not dry_run:
        # All input checks above finish before either output is created.
        for key, files, output in [('exp1', files9, output9), ('exp2', files17, output17)]:
            output.mkdir(parents=True, exist_ok=False)
            for rel, source in files.items():
                target = output / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if digest(target) != fingerprints[key][rel]:
                    raise ValueError(f"Copied file checksum mismatch: {target}")
            (output / 'preparation_record.json').write_text(
                json.dumps({**report, 'experiment': key, 'sha256': fingerprints[key]}, indent=2) + '\n',
                encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source9', required=True, type=Path)
    parser.add_argument('--source17', required=True, type=Path)
    parser.add_argument('--output9', required=True, type=Path)
    parser.add_argument('--output17', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Validate only, without creating outputs')
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == '__main__':
    main()
