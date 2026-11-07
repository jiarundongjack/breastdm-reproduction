# Reproducibility & Verification

[Project overview](README.md) · [Experimental results](result/README.md) · [Code](code/README.md) · [Environment evidence](environment/README.md)

This is the public verification guide for all readers. The fixed review source is [v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.1-review), commit `7e7ce75a9770c7f59a93da8745da42e410e717ce`. This guide is a later documentation update on `main`; it does not alter that snapshot.

## Start with saved-statistics verification

Follow the four commands in [SAVED_PREDICTION_VERIFICATION.md](code/analysis/SAVED_PREDICTION_VERIFICATION.md): two classification invocations and one each for 2D and 3D segmentation. They use the Python standard library and perform no training or inference.

| Scope | Local inputs | Public implementation |
|---|---|---|
| Six classification runs | Per-run `test_predictions_two_class.csv` and `author_style_metrics.csv` | [Classification verification](code/analysis/classification_metrics_from_predictions.py) |
| Two segmentation runs | Per-run `test_predictions.csv` and `test_metrics.csv`; established evaluation dimensions and mask conditions | [Segmentation verification](code/analysis/segmentation_metrics_from_predictions.py) |

Case-level prediction/statistic CSVs are private inputs; public metrics alone are insufficient. Use the original retained records for a historical comparison. A newly generated file is not automatically identical to a historical run. Output reports include local paths and must be inspected before sharing.

Classification verification distinguishes flattened two-column AUC from malignant-class AUROC and uses sample SD across seeds. Segmentation preserves the distinct 2D/3D DSC, mIoU and PPV aggregation rules. The no-ignored-voxel confirmation flag requires evidence about the evaluated masks and dimensions; `ignore_index=255` in code alone does not establish ignored pixels. In the inspected prepared segmentation inputs, stored mask value 255 maps to foreground through binarization, not to an ignored class.

Table 6 numerical differences can be checked against the formal metrics and published values. Its Confirmed/Unconfirmed provenance judgments require documentary evidence and cannot be established by a successful statistics script.

## Eight formal runs

| Experiment | Seed | Retained evidence |
|---|---|---|
| Exp-1 | 31415 | [Run records](result/fusion_exp1_100ep_seed31415_final_multiseed/) |
| Exp-1 | 27182 | [Run records](result/fusion_exp1_100ep_seed27182_final_multiseed/) |
| Exp-1 | 16180 | [Run records](result/fusion_exp1_100ep_seed16180_final_multiseed/) |
| Exp-2 | 48271 | [Run records](result/fusion_exp2_100ep_seed48271_final_multiseed/) |
| Exp-2 | 59317 | [Run records](result/fusion_exp2_100ep_seed59317_final_multiseed/) |
| Exp-2 | 84629 | [Run records](result/fusion_exp2_100ep_seed84629_final_multiseed/) |
| 2D U-Net | 123 | [Run records](result/seg2d_unet_seed123_gpu_20260911/) |
| 3D U-Net | 42 | [Run records](result/seg3d_unet_locked_volume_ppv_seed42_20260911/) |

These directories retain configurations, logs/histories, aggregate metrics, hashes and the publishable source snapshots. Some full implementations and patient-level records are excluded; the presence of a snapshot directory is not proof of a complete executable training source tree. See the [canonical result map](result/README.md#manuscript-final-results), [result manifest](provenance/manuscript_results.json) and [source setup](THIRD_PARTY_CODE.md).

## Verification levels

| Level | What it establishes | Current scope |
|---|---|---|
| **A — Saved-statistics verification** | Recalculation from retained predictions and per-sample statistics | Completed for six classification and two segmentation runs; private inputs are required to repeat it. |
| **B — Model reevaluation** | Executing evaluation with the required code, dependencies, checkpoint and exact prepared inputs | Prerequisites have been investigated; full dependency closure and a fresh formal reevaluation have not been independently established for all eight runs. |
| **C — End-to-end reconstruction** | Rebuilding exact inputs from original downloads, then training and evaluating | Not established for any of the four experiment groups. The upstream preparation gaps below remain. |

Historical release checks under [reproduction checks/](reproduction%20checks/README.md) record their own limited installation, preflight and inference tests. They do not establish full B/C-level verification of the formal experiments. No new training or inference was performed for this documentation update.

## Six retained technical limitations

1. **Classification channel construction.** The inspected 9/17-channel prepared arrays have corresponding last-nine channels, but the complete upstream channel construction is not independently established. One specified positional `clip(Post - Pre)` candidate did not match the prepared inputs. This does not rule out every subtraction formula or establish the semantic channel order.
2. **DICOM to 3D volumes.** The complete original-DICOM-to-`.npy` construction procedure has not been recovered. All 696 inspected prepared volumes already had depth eight; the loader therefore did not trigger padding or central-slice selection on those inputs. This observation does not recover upstream volume construction.
3. **Original downloads to 2D inputs.** The inspected 6,851 prepared test images were square and matched their masks. The full upstream path from the original downloads to the formal 2D images is still unproven. Organizing existing split data is not raw-data preprocessing.
4. **Historical segmentation environment and source binding.** Retained snapshots and later observed environments do not prove the exact training-day environment or an immutable binding of every segmentation source file to the formal run.
5. **Executable dependency closure.** Public tools, local adaptations and upstream dependencies have been mapped, but the complete closure and fresh model reevaluation have not been comprehensively validated. A-level success does not establish B- or C-level success.
6. **Published evaluation provenance.** The connection between the original released evaluators and some published metrics lacks sufficient evidence. Published 2D DSC/mIoU provenance remains **Unconfirmed**; saved-statistics agreement does not upgrade other unresolved provenance judgments.

## Inputs, source setup and models

- [Data preparation](code/preparation/README.md): organization of existing prepared inputs, with upstream boundaries.
- [Local generation](code/LOCAL_GENERATION.md): prerequisites and entry points for locally generated manifests, predictions, case figures and selection tables.
- [Source provenance](THIRD_PARTY_CODE.md) and [training modifications](provenance/TRAINING_MODIFICATIONS.md): public, local-only and upstream components.
- [Environment evidence](environment/README.md): run records, observed environments and release-validation lock files.
- [Model downloads and restoration](weight%20download%20and%20restore/DOWNLOADS.md): three rc1 archives, with [archive hashes](weight%20download%20and%20restore/ASSETS.json) and [12 model-file checksums](weight%20download%20and%20restore/FILES.json).
- [Full-record audits](docs/WORKFLOW_GUIDE.md#full-audits): additional checks that need completion manifests, logs and weights; these are distinct from prediction-only verification.

Original resources: [BreastDM paper](https://doi.org/10.1016/j.compbiomed.2023.107255) and [authors' repository/data links](https://github.com/smallboy-code/Breast-cancer-dataset). Obtain restricted inputs from authorized sources; patient-level data and private adaptations are not public release attachments.

## Versions and historical evidence

Use [v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.1-review) for the fixed review source. [v1.0.0](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.0) remains a historical source tag; it predates the new saved-statistics verification tools. [v1.0.0-rc1](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1) contains the historical model archives. Its automatic source archives are not the current review source.

The [release manifest](provenance/release_manifest.json) records these distinct roles. [Provenance documentation](provenance/README.md) explains historical file inventories and later path/disposition overlays. None of these historical manifests are rewritten to make them appear to describe a later tree.
