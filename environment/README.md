# Environments & Dependencies

[Project overview](../README.md) · [Verification scope](../REVIEW_NAVIGATION.md) · [Installation commands](../docs/WORKFLOW_GUIDE.md#environments)

Use separate classification and segmentation environments. These files serve different evidentiary purposes; they are not interchangeable historical environment records.

| Files | Purpose and limitation |
|---|---|
| [classification.txt](classification.txt), [segmentation.txt](segmentation.txt) | Dependency specifications based on the documented local environments. They support setup, but do not establish every dependency present during every original training run. |
| [classification.lock.txt](classification.lock.txt), [segmentation.lock.txt](segmentation.lock.txt) | Complete package resolutions from fresh Python 3.11.5 environments used during release validation; not retrospective training-day locks. |
| [classification_observed_packages.json](classification_observed_packages.json), [segmentation_observed_packages.json](segmentation_observed_packages.json) | Later environment observations. Their collection context must be distinguished from contemporaneous run evidence. |
| [reference_evaluator.txt](reference_evaluator.txt) | Additional dependencies for the separately configured reference evaluator; not proof that it generated the original published results. |

## Evidence standard

Inspect each formal run's configuration, logs and retained provenance records through the [eight-run index](../REVIEW_NAVIGATION.md#eight-formal-runs). A logged training-time version is stronger evidence of that run's environment than a later package inventory. Historical segmentation training versions and immutable source binding retain evidence gaps; do not infer them from the observed JSON or release locks.

The documented setup includes Python 3.11.5, Windows 11, an NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU (12 GB), classification PyTorch 2.10.0+cu130 and segmentation PyTorch 2.7.1+cu128. These reported/observed details are not a complete independently established training-day environment for all eight runs.

## Setup and verification

Use the preserved [installation commands](../docs/WORKFLOW_GUIDE.md#environments) for separate virtual environments, and read the dated [release checks](../reproduction%20checks/README.md) for what was actually tested. Those checks do not establish another-machine equivalence or complete fresh evaluation of all formal runs.

The [formal saved-statistics tools](../code/analysis/SAVED_PREDICTION_VERIFICATION.md) use the Python standard library and do not load PyTorch models. Their success cannot validate the training/inference dependency environment. See [source setup](../THIRD_PARTY_CODE.md) for implementations that must be supplied locally.
