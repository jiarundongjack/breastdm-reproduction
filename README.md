# BreastDM Reproducibility

**Classification · Segmentation · Reproducibility · Discrepancy Analysis**

An investigation of the reproducibility of BreastDM breast MRI classification and segmentation benchmarks, with an emphasis on protocol reconstruction, metric definitions and the provenance of reported results.

The study covers six classification runs and one run each of 2D and 3D U-Net. This repository provides public study-specific tools, retained experimental records and saved-statistics verification. The findings support **partial reproduction**: several results remain discrepant, and the available resources do not establish complete reconstruction from the original data.

[Experimental results](result/README.md#manuscript-final-results) · [Verification guide](REVIEW_NAVIGATION.md) · [Fixed review source](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.1-review)

## Key findings

| Focus | Finding |
|---|---|
| Classification | Specificity is substantially lower than the published values in both experiments. Three independent seeds are reported for each experiment. |
| AUC definitions | Released-code-compatible flattened two-column AUC and conventional malignant-class AUROC are evaluated separately; they are not interchangeable. |
| Segmentation | The 3D volume-mean PPV shows a substantial discrepancy. The 2D and 3D aggregation rules must be preserved when comparing metrics. |
| Evaluation provenance | Agreement with saved statistics does not establish the original published evaluator. Published 2D DSC/mIoU provenance remains **Unconfirmed**. |

Exact metrics and canonical output locations are indexed in [Experimental Results](result/README.md). Verification definitions and required inputs are documented in the [saved-statistics guide](code/analysis/SAVED_PREDICTION_VERIFICATION.md).

## Experimental overview

| Experiment | Formal seeds | Reporting |
|---|---|---|
| Classification Exp-1 | 31415, 27182, 16180 | Three runs; mean ± sample SD |
| Classification Exp-2 | 48271, 59317, 84629 | Three runs; mean ± sample SD |
| 2D U-Net | 123 | Single run |
| 3D U-Net | 42 | Single run |

[Browse the eight runs](REVIEW_NAVIGATION.md#eight-formal-runs) and the authoritative [result manifest](provenance/manuscript_results.json).

## Explore the repository

| Resource | Start here |
|---|---|
| **Experimental Results** | [Formal runs, aggregate metrics and figures](result/README.md) |
| **Code & Implementation** | [Scientific task index and execution entry points](code/README.md) |
| **Reproducibility & Verification** | [Verification scope, inputs and evidence limits](REVIEW_NAVIGATION.md) |
| **Data Preparation** | [Classification and segmentation input preparation](code/preparation/README.md) |
| **Environments & Dependencies** | [Setup instructions and environment evidence](environment/README.md) |
| **Models & Resources** | [Weight downloads, checksums and restoration](weight%20download%20and%20restore/DOWNLOADS.md) |
| **Provenance & Documentation** | [Source attribution, manifests and historical records](provenance/README.md) |

Detailed commands and historical analysis tasks remain in the [workflow reference](docs/WORKFLOW_GUIDE.md).

## Reproducibility notes

Saved-statistics checks have been completed for the eight formal runs. This verifies consistency with the retained predictions and statistics; it does not establish a fresh model evaluation or end-to-end reconstruction from raw downloads. Upstream input preparation, historical segmentation environments and complete executable dependencies retain evidence gaps. See the [verification levels and six technical limitations](REVIEW_NAVIGATION.md#verification-levels).

Original and preprocessed BreastDM data are not redistributed here. Case-level predictions, labels, images and other patient-level derived files with unclear redistribution permission remain private. Workflows requiring excluded source implementations need the documented [local source setup](THIRD_PARTY_CODE.md).

## Versioned resources

| Resource | Fixed version | Purpose |
|---|---|---|
| Review source | [v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.1-review) | Public source snapshot with the saved-statistics verification tools |
| Historical source | [v1.0.0](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.0) | Earlier source tree retained for traceability |
| Model archives | [v1.0.0-rc1](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1) | Initialization weights and trained checkpoints |

**Fixed review commit:** [`7e7ce75a9770c7f59a93da8745da42e410e717ce`](https://github.com/jiarundongjack/breastdm-reproduction/commit/7e7ce75a9770c7f59a93da8745da42e410e717ce).

The current `main` documentation may be newer than this fixed snapshot. Documentation updates do not move the tag or enter its source archive. The rc1 automatic source archives are historical; use its model assets with the documented source and prerequisites. See the [version manifest](provenance/release_manifest.json).

## Original resources

The original [BreastDM paper](https://doi.org/10.1016/j.compbiomed.2023.107255) and [authors' repository with data links](https://github.com/smallboy-code/Breast-cancer-dataset) are the starting points for data access and original implementations. Follow the providers' access and use requirements. Source attribution and redistribution boundaries are recorded in [THIRD_PARTY_CODE.md](THIRD_PARTY_CODE.md).
