# Classification input preparation

The preparation wrapper is public and requires no private model/training source. Supply the documented locally acquired inputs and environment.

This script organizes existing 9-channel and 17-channel `.npy` arrays into the input layout used in this reproduction. It copies retained arrays byte-for-byte, preserving their relative paths and existing train/validation/test assignments. It does not generate channels from DICOM or JPG images, compute subtraction images, normalize intensities, resize arrays, or create new splits.

## Inputs

For original resource access, see [DOWNLOADS.md](../../../weight%20download%20and%20restore/DOWNLOADS.md). Select the existing 9-channel and 17-channel directories as `--source9` and `--source17`. The locally inspected directories were named `cls/img9Se` and `cls/img17Se`. Check the downloaded contents: the current upstream download layout has not been verified, and this script requires the structure below.

```text
img9Se/ or img17Se/
  train/Benign/<patient>/*.npy
  train/Malignant/<patient>/*.npy
  val/Benign/<patient>/*.npy
  val/Malignant/<patient>/*.npy
  test/Benign/<patient>/*.npy
  test/Malignant/<patient>/*.npy
```

## Study-specific selection

From the 9-channel input only, omit `.npy` files under `test/Malignant/` for these six patient directories:

```text
BreaDM-Ma-1802
BreaDM-Ma-1803
BreaDM-Ma-1804
BreaDM-Ma-1806
BreaDM-Ma-1807
BreaDM-Ma-1808
```

This rule reconstructs the file selection observed in the saved formal inputs. In the locally inspected 9-channel source, these directories contain 43 files (13, 9, 4, 5, 8, 4 respectively). It is not a general duplicate-removal algorithm or a claim about why those patients were excluded historically. The 17-channel input is retained as supplied. An already filtered 9-channel input is also accepted if all checks pass.

The retained paths must match across both inputs, with these counts per experiment:

| Split | Benign | Malignant | Total |
|---|---:|---:|---:|
| train | 327 | 875 | 1202 |
| val | 24 | 93 | 117 |
| test | 114 | 289 | 403 |
| Total | 465 | 1257 | 1722 |

Each pair must be `uint8`, shaped `H×W×9` and `H×W×17`, with the same spatial dimensions; the final nine channels of the 17-channel array must equal the 9-channel array pixel-for-pixel. These checks match the inspected study inputs; they do not establish full provenance or the original raw-image conversion procedure.

## Commands

Requires Python and NumPy (available in the classification environment). Run from the repository root; replace the two source paths with your actual directories. First validate without copying:

```powershell
python code/preparation/classification/prepare_classification.py --source9 "D:/downloaded/cls/img9Se" --source17 "D:/downloaded/cls/img17Se" --output9 "data/classification_exp1/processed_dataset" --output17 "data/classification_exp2/img17Se" --dry-run
```

After validation, use the same command without `--dry-run` to copy the files:

```powershell
python code/preparation/classification/prepare_classification.py --source9 "D:/downloaded/cls/img9Se" --source17 "D:/downloaded/cls/img17Se" --output9 "data/classification_exp1/processed_dataset" --output17 "data/classification_exp2/img17Se"
```

Both output directories must be new, disjoint from the sources and each other, and must not be nested within one another or the sources. No source files are changed. The script validates all arrays before copying, checks copied file hashes, and writes `preparation_record.json` inside each output. An interrupted copy may leave a partial output; use new output paths for a retry. The existing training loaders perform division by 255 and resizing to 96×96 at load time, so this preparation step must not apply them again.

This script runs directly, not through `code/run.py`. Only the script and documentation are provided here; obtain the input arrays through the original resources.
