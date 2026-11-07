# Resource access

The archive verification/restoration utility is public and needs no private model source. Actual inference may require workflow-specific [source setup](../THIRD_PARTY_CODE.md).

## Original code and data

Original BreastDM paper: [https://doi.org/10.1016/j.compbiomed.2023.107255](https://doi.org/10.1016/j.compbiomed.2023.107255).

This repository does not redistribute original or preprocessed BreastDM data. Case-level CSVs, case images, masks, predictions and other BreastDM-derived patient-level files with unclear redistribution permission are excluded from the current `main` tree. Obtain data through the [original authors' repository and data links](https://github.com/smallboy-code/Breast-cancer-dataset), then generate these outputs privately using the existing scripts and the local-generation entry points linked below. Aggregate metrics and statistical figures are retained.

Generation commands and prerequisites: [local generation guide](../code/LOCAL_GENERATION.md). The original-to-prepared input limitations remain documented in the preparation guides.

The original BreastDM code and data access information are provided in the authors' repository:

[Original BreastDM repository](https://github.com/smallboy-code/Breast-cancer-dataset)

Please obtain the data through the Google Drive links in the original repository's README. Follow the original providers' access and use requirements. If a link is unavailable or access is restricted, contact the original providers.

## Reproduction environment and instructions

Environment specifications are provided in [environment guide](../environment/README.md). Reproduction commands and task descriptions are provided in [workflow reference](../docs/WORKFLOW_GUIDE.md).

The experiments use task-specific prepared inputs. Files downloaded from the original source should not be assumed to match the prepared directory structure automatically.

## Source and model versions

The model-weight release and the source-code release are versioned separately. Use [v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.1-review) at commit `7e7ce75a9770c7f59a93da8745da42e410e717ce` for current review source and instructions. Use [v1.0.0-rc1](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1) only for the model assets listed in ASSETS.json. Do not use the rc1 automatic “Source code” archives as the current source. Historical [v1.0.0 source](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.0) remains available for traceability. See [release_manifest.json](../provenance/release_manifest.json).

In `ASSETS.json`, `version` identifies the historical manifest version, not the current review source; `asset_version` remains rc1. Asset filenames, sizes and checksums are unchanged. Case-level predictions, labels and private review materials are not uploaded as public release attachments.

## Archive restoration

`restore_assets.py` is a restoration utility for the reproduction-specific ZIP archives listed in `ASSETS.json`. It does not download or prepare the original BreastDM data and should not be run on the original authors' downloaded archives.

The model archives listed in [ASSETS.json](ASSETS.json) are available from the [rc1 release](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1). `FILES.json` records checksums for the 12 model files inside those three archives; it is not a source-code inventory or a dataset manifest.

Download all three listed ZIP files into one local directory, then run from the repository root:

```powershell
python "weight download and restore/restore_assets.py" --asset-dir "D:/downloaded_model_assets" --verify-only
python "weight download and restore/restore_assets.py" --asset-dir "D:/downloaded_model_assets"
```

Replace the example path with your download directory. The first command verifies the archives and member checksums; the second restores the model files. Saved-prediction statistics do not require these model archives; see [saved-statistics verification](../code/analysis/SAVED_PREDICTION_VERIFICATION.md).

## Sources and redistribution

For original resource access, see the links above. For model-branch licenses, see the [license guide](Classification%20model%20branch%20licenses/README.md).

## Reproducibility & Verification

See [verification guide](../REVIEW_NAVIGATION.md) for the eight formal runs, fixed source commit, private-input requirements and historical weight release. Historical v1.0.0 descriptions retain their original meaning; the current fixed review is v1.0.1-review.
