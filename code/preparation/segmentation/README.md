# Segmentation data preparation

The preparation wrappers are public and require no private model/training source. They organize the documented local inputs; they do not replace all upstream raw-data conversion steps.

These scripts implement the existing patient-directory selection and copying steps used in this reproduction. They accept explicit source and output paths and use only the Python standard library. Python 3.11 is the version used for this project.

## Inputs

Obtain original code and data access information from the [original authors' repository](https://github.com/smallboy-code/Breast-cancer-dataset). Follow the original providers' access and use requirements.

The scripts require **existing split segmentation directories** with this structure. Select the appropriate 2D or 3D input directory as `--source`:

```text
source/
  train/
    images/<patient_directory>/...
    labels/<patient_directory>/...
  val/
    images/<patient_directory>/...
    labels/<patient_directory>/...
  test/
    images/<patient_directory>/...
    labels/<patient_directory>/...
```

The study's local input folders were named `seg` (2D) and `seg3D` (3D). These names describe the inputs used in the study; they do not establish that every upstream download has this layout. Confirm the contents after extraction. If your download does not contain these split directories, these scripts alone cannot prepare it for training.

The scripts do not convert DICOM, generate volumes, resample images, create masks, generate classification channels, or create new train/validation/test splits. They check patient-directory pairing, not voxel alignment or file-level image/mask correspondence.

## Selection rules

| Script | Rule |
|---|---|
| `make_seg2d_clean.py` | Copy paired patient directories; exclude `BreaDM-Be-1801`, `BreaDM-Be-1803` and `BreaDM-Be-1804` from **test only**, following the study's existing duplicate-patient exclusion list. |
| `make_seg3d_clean.py` | Select patients present in each split's `images` directory and copy their corresponding `labels`; label-only patient directories are not copied. |

The 2D exclusion list is fixed for this study, not a general duplicate-detection algorithm. Both scripts preserve file contents, patient-directory names and the existing splits.

## Commands

Run from the reproduction repository root. Replace `D:/downloaded_BreastDM/seg` and `D:/downloaded_BreastDM/seg3D` with your actual input directories. Paths containing spaces must be quoted.

First check inputs without copying:

```powershell
python code/preparation/segmentation/make_seg2d_clean.py --source "D:/downloaded_BreastDM/seg" --output "data/segmentation_2d/seg2D_clean_v2" --dry-run
python code/preparation/segmentation/make_seg3d_clean.py --source "D:/downloaded_BreastDM/seg3D" --output "data/segmentation_3d/seg3D_clean" --dry-run
```

Then copy the selected data:

```powershell
python code/preparation/segmentation/make_seg2d_clean.py --source "D:/downloaded_BreastDM/seg" --output "data/segmentation_2d/seg2D_clean_v2"
python code/preparation/segmentation/make_seg3d_clean.py --source "D:/downloaded_BreastDM/seg3D" --output "data/segmentation_3d/seg3D_clean"
```

Use these standalone commands directly; these scripts are not tasks registered with `code/run.py`. The output locations match the segmentation input paths in the [workflow reference](../../../docs/WORKFLOW_GUIDE.md#contents). Model dependencies and training requirements are described there separately.

## Output handling

Source data are read and copied, never edited. The output directory must not already exist and must not be nested inside the source. All required split directories and selected patient pairs are checked before copying starts. If copying is interrupted by a filesystem error, the incomplete destination is left for inspection; rerunning refuses to overwrite it. A successful dry run checks directory structure only, not sufficient disk space or dataset validity.
