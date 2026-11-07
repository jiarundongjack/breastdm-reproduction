# Code & Implementation

[Project overview](../README.md) · [Verification](../REVIEW_NAVIGATION.md) · [Results](../result/README.md)

Code is organized by scientific function. The public tree contains study-specific preparation, analysis, verification and plotting tools, together with permitted source components. Workflows depending on excluded original-author or mixed-source implementations need [local source setup](../THIRD_PARTY_CODE.md); a listed task does not guarantee that all its inputs or implementations are public.

## Main entry points

| Purpose | Entry point | Instructions |
|---|---|---|
| Verify saved formal statistics | [analysis/](analysis/) | [Four verification commands and input schemas](analysis/SAVED_PREDICTION_VERIFICATION.md) |
| Summarize classification predictions | [summarize_predictions.py](analysis/summarize_predictions.py) | [CSV-only summary guide](analysis/README.md) |
| Organize existing prepared data | [preparation/](preparation/) | [Preparation scope and guides](preparation/README.md) |
| Run registered scientific tasks | [run.py](run.py) | [Launcher, inputs and task reference](../docs/WORKFLOW_GUIDE.md#entry-point) |
| Generate private case-level outputs | [local_generation.py](local_generation.py) | [Local generation and prerequisites](LOCAL_GENERATION.md) |

From the repository root, list the task registry without training or inference:

```text
python -B code/run.py --list
```

Registered classification/segmentation tasks run through `code/run.py`, which stages their conventional module layout temporarily. Standalone preparation and analysis tools run directly as documented in their own guides. Private-data and model-dependent commands require the appropriate local setup.

## Scientific task groups

| Group | Public functionality and navigation |
|---|---|
| [Classification](classification/) | Data loaders, metric calculations, run provenance, per-seed summaries, ROC/confusion matrices, channel checks and case analysis |
| [2D segmentation](segmentation_2d/) | Retained loader and permitted model components; training/evaluation source requirements are documented separately |
| [3D segmentation](segmentation_3d/) | Data loading, metric aggregation, protocol checks, source/intensity comparisons, volume selection and saved-run audits |

The [eight-run index](../REVIEW_NAVIGATION.md#eight-formal-runs) links configurations, historical snapshots and logs. [Training modifications](../provenance/TRAINING_MODIFICATIONS.md) records study-specific protocol facts; the [workflow reference](../docs/WORKFLOW_GUIDE.md#training) preserves conditional training commands. No training is needed for the saved-statistics tools.

## Supporting analyses and historical names

- [Channel mapping](classification/channel_mapping/) and [subtraction checks](classification/subtraction_check/): investigations of existing prepared arrays, not a recovered raw-input construction pipeline.
- [Grad-CAM analysis](classification/fig10_11_gradcam/): local case visualization requiring private inputs and compatible model source.
- [3D case predictions](segmentation_3d/fig13_predictions/), [slice selection](segmentation_3d/fig13_slice_selection/) and [volume selection](segmentation_3d/oversegmentation_volume_selection/): local case analyses with distinct input requirements.
- [3D result audit](segmentation_3d/table7_result_audit/) and [protocol checks](segmentation_3d/protocol_checks/): supporting checks, not additional formal experiments.

Names such as `fig10_11`, `fig13` and `table7` are historical identifiers. They must not be interpreted as final manuscript figure/table numbering. Files and task identifiers are retained unchanged.

The detailed [analysis task reference](../docs/WORKFLOW_GUIDE.md#analysis) preserves input aliases, commands, output locations and full-audit requirements. [Source provenance and path overlays](../provenance/README.md) provide file-level history and attribution; this index makes no new authorship or license claim.
