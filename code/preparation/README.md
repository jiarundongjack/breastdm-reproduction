# Data Preparation

[Project overview](../../README.md) · [Code index](../README.md) · [Reconstruction limits](../../REVIEW_NAVIGATION.md#six-retained-technical-limitations)

Obtain original data through the [BreastDM authors' repository and download links](https://github.com/smallboy-code/Breast-cancer-dataset); the original paper is [doi:10.1016/j.compbiomed.2023.107255](https://doi.org/10.1016/j.compbiomed.2023.107255). Access conditions are controlled by the providers. Original/preprocessed data and patient-level outputs with unclear redistribution permission are not redistributed here.

## Existing prepared inputs

| Experiment | Guide | What the supplied preparation code does |
|---|---|---|
| Classification Exp-1 / Exp-2 | [Classification preparation](classification/README.md) | Selects, validates and copies existing split 9/17-channel `.npy` arrays while preserving their contents. |
| 2D / 3D U-Net | [Segmentation preparation](segmentation/README.md) | Organizes existing split images, masks and volumes according to the documented study-specific selection rules. |

Follow those guides for exact source layout, local destination paths and commands. The [workflow input table](../../docs/WORKFLOW_GUIDE.md#contents) describes the expected local experiment layout. Generating private manifests and downstream outputs is covered by [LOCAL_GENERATION.md](../LOCAL_GENERATION.md).

## Upstream boundary

These preparation scripts do not reconstruct the entire original-download-to-experiment pipeline. The 9/17-channel construction, the conversion from original DICOM to 3D `.npy` volumes, and the creation of formal 2D images from raw downloads remain incompletely established.

Being able to read or copy existing prepared inputs does not prove that the same inputs can be regenerated from the original download. In particular, the observed eight-slice depth of prepared 3D volumes does not recover their upstream construction, and agreement of the last nine classification channels does not establish all channel semantics. Read the [six technical limitations](../../REVIEW_NAVIGATION.md#six-retained-technical-limitations) before making end-to-end reproducibility claims.
