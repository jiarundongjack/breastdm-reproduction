# Workflow and command reference

[Project overview](../README.md) · [Code index](../code/README.md) · [Reproducibility & Verification](../REVIEW_NAVIGATION.md)

This guide preserves the detailed technical instructions previously on the project homepage.
**All commands run from the repository root**, even though this document is in `docs/`.
Task identifiers and historical figure/table numbers are retained for traceability; they do not assign current manuscript numbering.

For the formal saved-statistics checks, use the dedicated [verification commands and input schemas](../code/analysis/SAVED_PREDICTION_VERIFICATION.md).
For private source setup and generation, use [local generation](../code/LOCAL_GENERATION.md).
Commands requiring excluded implementations, prepared inputs or checkpoints are conditional on those prerequisites; their presence here does not establish complete evaluation or raw-data reconstruction.

[Data sources](#resources) · [Input layout](#contents) · [Environment installation](#environments) · [Launcher](#entry-point) · [Analysis tasks](#analysis) · [Full-record audits](#full-audits) · [Training commands](#training) · [Model downloads](#versions)

<a id="resources"></a>
## Original resources

Original BreastDM paper: [https://doi.org/10.1016/j.compbiomed.2023.107255](https://doi.org/10.1016/j.compbiomed.2023.107255).

This repository does not redistribute original or preprocessed BreastDM data. Case-level CSVs, case images, masks, predictions and other BreastDM-derived patient-level files with unclear redistribution permission are excluded from the current `main` tree. Obtain data through the [original authors' repository and data links](https://github.com/smallboy-code/Breast-cancer-dataset), then generate these outputs privately using the existing scripts and the local-generation entry points linked below. Aggregate metrics and statistical figures are retained.

See [local generation commands](../code/LOCAL_GENERATION.md) for manifests, predictions, Grad-CAM, segmentation examples and case-selection tables.

The original BreastDM code and data access information are available from the [original authors' repository](https://github.com/smallboy-code/Breast-cancer-dataset). Please obtain the data through the download links provided there and follow the original providers' access and use requirements.

Original-author data and complete external/mixed implementations with unresolved redistribution permission are not republished in the current tree. Project-specific scripts are public, and licensed third-party files retain their attribution and license texts. See [training modifications](../provenance/TRAINING_MODIFICATIONS.md) for preserved reproduction-specific protocol facts without republishing the excluded training bodies.

See [DOWNLOADS.md](../weight%20download%20and%20restore/DOWNLOADS.md) for resource access instructions and [model-branch license guide](../weight%20download%20and%20restore/Classification%20model%20branch%20licenses/README.md) for upstream model-branch licensing information.

The reported experiments used task-specific prepared datasets. Files obtained from the original resources require preparation before they can be used with the experiment commands below.

<a id="classification-preparation"></a>
## Classification data preparation

See [classification preparation instructions](../code/preparation/classification/README.md) for the existing 9-channel and 17-channel input layout, the study-specific file selection, validation checks, and copy commands.

The script organizes existing split `.npy` arrays and preserves their contents. It does not generate channels from raw images or create new data splits. The inspected local inputs reproduce the formal input files exactly; the current upstream download layout has not been verified.

<a id="segmentation-preparation"></a>
## Segmentation data preparation

See [segmentation preparation instructions](../code/preparation/segmentation/README.md) for the required input structure, patient-selection rules, and commands for the 2D and 3D preparation scripts.

These scripts organize existing split segmentation data; they do not perform the complete conversion from original downloads to training inputs.

<a id="contents"></a>
## Local input and output layout

The paths below describe the local directory structure used by the experiments. A listed path does not imply that its data or model files are included in this repository.

| Path | Purpose |
|---|---|
| `data/classification_exp1/processed_dataset` | Exp-1, nine channels |
| `data/classification_exp2/img17Se` | Exp-2, seventeen channels |
| `data/segmentation_2d/seg2D_clean_v2` | 2D images and masks |
| `data/segmentation_3d/seg3D_clean` | 3D volumes and masks |
| `code/classification`, `code/segmentation_2d`, `code/segmentation_3d` | Scientific task implementations and shared modules; see the cleanup inventory |
| `code/preparation` | Existing split classification and segmentation input preparation |
| `code/analysis` | Classification statistics from saved prediction CSVs |
| `code/pretrained` | Two classification initialization checkpoints |
| `result/fusion_*`, `result/seg*` | Eight unchanged formal runs |
| `result/paper_outputs` | Canonical cross-seed summaries, flattened ROC and confusion matrices |
| `result/reproduced` | Retained diagnostic/reproduction evidence; see result/README.md |
| `environment` | Observed dependencies for two environments |

Use **`result`**, singular. Historical configuration files retain their original absolute
paths and auxiliary metric fields as evidence; they are not executable configuration for this release.

<a id="environments"></a>
## Environments

The `classification.lock.txt` and `segmentation.lock.txt` files record the complete
package versions resolved in fresh Python 3.11.5 environments during release preparation.
They are release-validation records, not retrospective records of the original training installations.
For the pinned release environment, substitute the corresponding `.lock.txt` for `.txt`
in the installation commands below.

The documented local setup records Python 3.11.5, Windows 11, an NVIDIA RTX PRO 3000
Blackwell Generation Laptop GPU (12 GB), classification PyTorch 2.10.0+cu130,
and segmentation PyTorch 2.7.1+cu128. These reported/observed versions do not independently
establish the segmentation training-day environment. See the [environment evidence guide](../environment/README.md)
for the distinction between run records, later observations and release-validation locks.
Use separate environments for these tasks.

Example installation with Python 3.11:

```powershell
python -m venv .venv_classification
.\.venv_classification\Scripts\python.exe -m pip install -r environment/classification.txt
python -m venv .venv_segmentation
.\.venv_segmentation\Scripts\python.exe -m pip install -r environment/segmentation.txt
```

Both dependency sets were installed in fresh Python 3.11.5 virtual environments during the release checks documented in [reproduction checks](../reproduction%20checks/README.md). This does not establish identical ancillary package versions during historical training or successful installation on another computer.
Use the relevant interpreter for every command below; `python` denotes that interpreter.

<a id="entry-point"></a>
## Entry point

Run commands from the repository root:

```powershell
python code/run.py --list
python code/run.py classification/exp1_training -- --help
python code/run.py segmentation_3d/table7_fig8_9_12_training -- --help
```

Do not execute files inside task folders directly. `code/run.py` copies the selected task's
modules into a temporary conventional package layout, launches them with the same interpreter,
and removes the temporary directory on completion. Canonical source files remain in their task folders.
Repository locations are resolved from the launcher, so no personal drive letters are required.
Additional options follow `--`.

The standalone scripts in `code/preparation/` and `code/analysis/` run directly with Python, using the commands in their respective READMEs. They are not registered tasks of `code/run.py`.

<a id="analysis"></a>
## Analysis scripts

These tasks describe existing implementations; listing a task does not mean its samples or weights are included. This guide does not assign independent authorship or a new license to upstream code.

### 1. Environment and entry point

Use the classification or segmentation environment described in [Environments](#environments). Run commands from the repository root, using that environment's Python interpreter. Use `code/run.py` for the task identifiers in the tables below, not the individual task files directly. The standalone CSV summary in Section 3 uses its own command shown there.

For any task identifier in the tables, use this template, replacing `TASK` and the output directory:

```powershell
python code/run.py TASK --output-root result/reproduced/my_analysis
```

`OUT` below means the selected `--output-root`. Use a fresh output directory for a new analysis; some tasks refuse existing outputs, while others overwrite their own generated files. Use the same output directory only when a task explicitly consumes another task's generated outputs. Console-only tasks print their results without creating a report file.

`--runs-root` is used by the classification summary tasks through `RUNS`. It does not redirect all tasks: most plotting scripts read the fixed historical `result/` paths below. `--output-root` changes output placement, not input data locations.

### 2. Input locations

| Short name | Repository-relative location |
|---|---|
| C1 | `result/fusion_exp1_100ep_seed{seed}_final_multiseed/`; seeds: `31415`, `27182`, `16180` |
| C2 | `result/fusion_exp2_100ep_seed{seed}_final_multiseed/`; seeds: `48271`, `59317`, `84629` |
| S3 | `result/seg3d_unet_locked_volume_ppv_seed42_20260911/` |
| D1 | `data/classification_exp1/processed_dataset/` |
| D2 | `data/classification_exp2/img17Se/` |
| D2D | `data/segmentation_2d/seg2D_clean_v2/` |
| D3D | `data/segmentation_3d/seg3D_clean/` |

The `data/` paths above are local input directories populated by the reader following the preparation instructions; sample data are not included in this repository. The preparation scripts organize existing prepared arrays or split segmentation inputs and do not perform a complete raw-image-to-training conversion.

The patient-level input CSVs below are no longer distributed. First follow [local generation](../code/LOCAL_GENERATION.md). For case analyses, use the private copy's `.local/reproduction/code/run.py`; paths C1/C2/S2/S3/D1/D2/D2D/D3D then refer to that private copy. The commands in this section describe required inputs, not files bundled with the public repository.

### 3. Analyses using saved records

#### Prediction-only classification summary

Use the existing [CSV-only summary guide](../code/analysis/README.md) for commands, required columns, output schemas and the flattened-AUC definition. It reads three prediction CSVs for the selected experiment using the Python standard library, without images, weights or completion manifests. Full-record audits below have different requirements.

For checks against formal metric records and the additional malignant-class AUROC, use the [formal saved-statistics verification guide](../code/analysis/SAVED_PREDICTION_VERIFICATION.md). Both guides require new output directories and private local input records; their commands remain maintained at those entry points.

#### Plotting and case selection

These tasks do not load model checkpoints or raw sample arrays. They still require the listed records and the appropriate Python dependencies.

| Task | Required inputs | Main outputs |
|---|---|---|
| `classification/exp1_mean_roc` | All three C1 `test_predictions_two_class.csv` files | `OUT/exp1_mean_roc.png`, `exp1_auc_summary.csv`, `exp1_confusion_per_seed.csv`, `exp1_mean_confusion_matrix.csv`, report |
| `classification/exp2_mean_outputs` | All three C2 `test_predictions_two_class.csv` and `author_style_metrics.csv` files | `OUT/exp2_roc/`: ROC, confusion-matrix PNGs, AUC and confusion CSVs, report |
| `classification/exp2_mean_confusion_matrix` | All three C2 `test_predictions_two_class.csv` files | `OUT/exp2_mean_confusion_matrix_exact.csv`, `exp2_mean_confusion_matrix_integer.csv`, `exp2_mean_confusion_matrix_integer.png`, note |
| `classification/exp2_false_positive_cases` | All three C2 `test_predictions_two_class.csv` files | Three `OUT/exp2_top10_*.csv` files: single-run cases, persistent cases and details |
| `segmentation_3d/oversegmentation_volume_selection` | S3 `test_predictions.csv` | `OUT/volume_overseg_two_level_selection/all_eligible_volumes_ranked.csv` and `TOP3_volumes_two_level_selection.csv` |

For classification prediction CSVs, preserve the original columns, including `True Label`, `Predicted Label`, `Benign Probability` and `Malignant Probability`. Case-selection tasks also use sample-identifying columns. Do not substitute a summary-only CSV for the per-sample predictions.

Example: regenerate Exp-1 ROC and aggregate confusion counts:

```powershell
python code/run.py classification/exp1_mean_roc --output-root result/reproduced/exp1_record_analysis
```

Example: regenerate Exp-2 ROC:

```powershell
python code/run.py classification/exp2_mean_outputs --output-root result/reproduced/exp2_record_analysis
```

The Exp-2 ROC task creates an `exp2_roc` subdirectory and requires it not to exist. These examples recalculate saved predictions; they do not rerun training or establish the original paper's evaluation procedure.

<a id="full-audits"></a>
### 4. Full-record summaries and audits

| Task | Required inputs | Main outputs |
|---|---|---|
| `classification/table4_exp1_summary` | Complete C1 runs: completion manifest and every file it references, training history, prediction/metric CSVs, configuration and `best_fusion_model.pth` | `OUT/exp1_final_multiseed_summary/`: per-seed metrics, mean/SD and comparison CSVs |
| `classification/table5_exp2_summary` | Complete C2 runs with the corresponding completion manifests, referenced files and weights | `OUT/exp2_final_multiseed_summary/`: per-seed metrics, mean/SD and comparison CSVs |
| `segmentation_3d/table7_result_audit` | S3 history, predictions, metrics, `run_config.txt`, `training_log.txt`, `code_snapshot/`, `best_model.pth`, `last_checkpoint.pth` | `OUT/audit_results.json`, `recomputed_metric_check.csv`, `reconstructed_counts.csv`, `evidence_chain.csv`, `paper_vs_reproduction.csv`, before/after hashes |

The current classification summary functions verify the completed-run file hashes and require checkpoints. They are not CSV-only summary commands. Missing weights or other manifest-listed files will cause verification to fail. Keep these checks when performing a full audit; do not treat a partial record set as a verified complete run.

```powershell
python code/run.py classification/table4_exp1_summary --output-root result/reproduced/exp1_full_summary
python code/run.py classification/table5_exp2_summary --output-root result/reproduced/exp2_full_summary
python code/run.py segmentation_3d/table7_result_audit --output-root result/reproduced/seg3d_full_audit
```

### 5. Analyses requiring prepared samples

| Task | Required inputs | Main outputs |
|---|---|---|
| `classification/channel_mapping` | D1 and D2 test arrays; no weights | `OUT/fp_fn_cases/exp1_exp2_channel_mapping_verification.csv` |
| `classification/subtraction_check` | D1 and D2 test arrays; no weights | `OUT/fp_fn_cases/subtraction_formula_all_candidates.csv`, `subtraction_formula_best_mapping.csv` |
| `classification/exp2_seed59317_case_source` | D2 test data and C2 seed59317 `code_snapshot/data_loader_exp2.py`; no weights | Console: sample index 96 source |
| `classification/fig10_11_gradcam` | D2 test data; C2 seed59317 predictions, `best_fusion_model.pth`, and required model/loader snapshot modules | `OUT/exp2_seed59317_index96_gradcam/`: input, Grad-CAM, channel PNGs and metadata |
| `segmentation_3d/protocol_checks` | D3D all splits and current segmentation modules; no trained weights | `OUT/test_report.json`, synthetic `test_metrics.csv`, console |
| `segmentation_3d/zero_padding_check` | D3D volumes; no weights | Console zero-slice statistics |
| `segmentation_3d/source_slice_matching` | D3D, candidate D2D images and current code; no weights | Console matching results |
| `segmentation_3d/rgb_intensity_check` | The specific paired D2D images and D3D volume selected in the script; no weights | Console intensity comparisons |
| `segmentation_3d/rgb_prediction_impact` | The selected D2D/D3D images and mask, S3 `best_model.pth`, current 3D model implementation | Console prediction-impact comparisons |
| `segmentation_3d/threshold_analysis` | D3D test data, S3 `best_model.pth` and `test_metrics.csv`, current model/dataset/training modules | `OUT/threshold_sensitivity_volume_ppv.csv` |

These are study-specific checks; several select fixed cases or assert the study's expected shapes and counts. Changing datasets is not equivalent to rerunning the reported analysis. Inference tasks also require compatible model code and dependencies. Providing only a data link does not supply these inputs automatically.

### 6. 3D example figures

| Task | Required inputs | Main outputs |
|---|---|---|
| `segmentation_3d/fig13_predictions` | D3D test data; S3 `test_predictions.csv`, `best_model.pth`, model and dataset snapshots | `OUT/top3_overseg_maxfp_v3/`: prediction PNGs, slice CSVs and summary |
| `segmentation_3d/fig13_slice_selection` | Same inputs as `fig13_predictions`; it executes that script internally | `OUT/volume55_slice_overseg_selection/`: `all_slices_volume55.csv`, `eligible_slices_ranked.csv`, `SELECTED_slice.csv`, representative PNG; also prediction outputs |

With all inference inputs available, run the retained slice-selection task below. It also invokes prediction generation. Separate publication-layout helpers from earlier versions are no longer registered tasks; historical case PNGs are not distributed.

```powershell
python code/run.py segmentation_3d/fig13_slice_selection --output-root result/reproduced/seg3d_example
```

### 7. Reference evaluator

`segmentation_2d/ppv_reference` is a reference evaluator adapted from the original materials, not a newly authored analysis implementation or the main training evaluator. It reads separately exported predicted masks and ground-truth masks and prints precision, recall, accuracy and F1. Install its extra dependencies from `environment/reference_evaluator.txt` and provide matching mask directories explicitly:

```powershell
python -m pip install -r environment/reference_evaluator.txt
python code/run.py segmentation_2d/ppv_reference -- --test_images "D:/exported_masks/predictions" --ground_truth_images "D:/exported_masks/ground_truth"
```

### Reported metric notes

Exp-2 ROC uses flattened two-column probabilities and one-hot labels in the retained implementation: **0.8735928017 ± 0.0218863413**, with sample SD (`ddof=1`). This numerical agreement does not establish the original authors' evaluation procedure. The obsolete single-column ROC is not the final Exp-2 figure in `result/paper_outputs`.

3D PPV is an equally weighted mean over all 141 test volumes; empty predictions contribute zero, and no volume is excluded for TP=0 or TN=0. Formal PPV is **0.604198665930658**.

<a id="training"></a>
## Retrain the eight runs

Historical runs under `result` are protected. New training goes under `result/new_runs`.
For each classification experiment, run its seeds in the listed order; provenance checks compare
successive runs' data, code, and environment. Do not mix new runs with historical runs.

Classification environment:

```powershell
foreach ($seed in 31415,27182,16180) {
  python code/run.py classification/exp1_training -- --model fusion --random_seed $seed --output-dir "result/new_runs/fusion_exp1_100ep_seed${seed}_final_multiseed"
  if ($LASTEXITCODE -ne 0) { throw "Exp-1 failed" }
}
foreach ($seed in 48271,59317,84629) {
  python code/run.py classification/exp2_training -- --model fusion --random_seed $seed --output-dir "result/new_runs/fusion_exp2_100ep_seed${seed}_final_multiseed"
  if ($LASTEXITCODE -ne 0) { throw "Exp-2 failed" }
}
```

Add `--preflight-only` to the first seed command to check data and initialization files without training.

Segmentation environment:

```powershell
python code/run.py segmentation_2d/table6_fig6_7_training -- --data-root data/segmentation_2d/seg2D_clean_v2 --output-dir result/new_runs/seg2d_unet_seed123 --seed 123 --epochs 100 --batch-size 16 --lr 0.01 --device cuda --num-workers 8 --input-size 224 --lr-patience 10 --lr-factor 0.1
python code/run.py segmentation_3d/table7_fig8_9_12_training -- --data-root data/segmentation_3d/seg3D_clean --output-dir result/new_runs/seg3d_unet_seed42 --seed 42 --epochs 100 --batch-size 8 --lr 0.001 --device cuda --num-workers 8 --input-size 224 --depth 8 --lr-patience 10 --lr-factor 0.1
```

Summarize newly completed classification runs:

```powershell
python code/run.py classification/table4_exp1_summary --runs-root result/new_runs --output-root result/new_summary
python code/run.py classification/table5_exp2_summary --runs-root result/new_runs --output-root result/new_summary
```

<a id="versions"></a>
## Versions and model archives

**Current review source: [v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.1-review)** — complete public source snapshot at commit `7e7ce75a9770c7f59a93da8745da42e410e717ce`, including the saved-statistics verification entry points. Use this fixed version for the current review; later `main` documentation updates do not move the tag. See [release_manifest.json](../provenance/release_manifest.json). Private review materials are not included in public release attachments.

Historical [v1.0.0 source](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.0) remains available for traceability; it does not contain the new verification entry points.

**Model assets: [v1.0.0-rc1](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1)** — retained solely as the historical model-weight prerelease. Its three model ZIPs and checksum file are unchanged. **Do not use the rc1 “Source code” archive as the current reproduction source.** Those automatic archives reflect the historical tagged tree.

To identify the fixed review commit in a Git clone, run `git rev-parse 'v1.0.1-review^{commit}'` after fetching the tag. It must resolve to `7e7ce75a9770c7f59a93da8745da42e410e717ce`. The release manifest records this literal fixed SHA; later documentation commits on main are not the tagged review source.

When using a tagged source archive, consult the current release description for the interpretation of historical version wording in that snapshot.

### Download model weights

Download all three ZIP archives below. These links download the files directly; there is no need to visit the release page.

| Archive | Download |
|---|---|
| Pretrained initialization weights | [pretrained-01.zip](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/breastdm-v1.0.0-rc1-pretrained-01.zip) |
| Trained checkpoints, archive 1 | [trained-models-01.zip](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/breastdm-v1.0.0-rc1-trained-models-01.zip) |
| Trained checkpoints, archive 2 | [trained-models-02.zip](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/breastdm-v1.0.0-rc1-trained-models-02.zip) |

[SHA-256 checksums](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/SHA256SUMS.txt)

After downloading, follow the [verification and restoration instructions](../weight%20download%20and%20restore/DOWNLOADS.md).

## Scientific-function organization

Repository organization is based on scientific function rather than manuscript figure numbering. Final manuscript panel composition and publication layout are not part of the reproducibility codebase. This is a post-v1.0.0 navigation cleanup on main; the fixed v1.0.0 source tag and all scientific result records remain unchanged.

The [file-level cleanup inventory](../provenance/figure_code_cleanup.json) maps old/new paths and explains decisions; [SHA-256 verification](../provenance/figure_code_sha256_before_after.json) records byte identity. Scripts mixing scientific computation and explicit manuscript composition remain untouched for manual review, including the Grad-CAM and over-segmentation figure tasks that still have historical figure-number folder names. They were not split or renamed. Historical source/result inventories retain their original paths and hashes; use this cleanup inventory as the path/disposition overlay for current main. Excluded private-source restoration paths are unchanged.

Table-calculation and training tasks retain their existing names.

The standalone Exp-1 confusion-matrix renderer only styled an existing aggregate CSV and was removed; the computation of that CSV remains in `classification/exp1_mean_roc`, and its canonical saved PNG is unchanged. Scientific case/slice selection and Grad-CAM computation remain available; only isolated presentation-only scripts were removed.

### Formal saved-statistics verification

For the six formal classification runs, both flattened two-column AUC and conventional malignant-class AUROC are available through `classification_metrics_from_predictions.py`. For the formal 2D and 3D runs, `segmentation_metrics_from_predictions.py` reconstructs aggregate segmentation metrics from saved per-sample statistics. See [commands, input schemas and limitations](../code/analysis/SAVED_PREDICTION_VERIFICATION.md). These new entry points require saved predictions/statistics and official metrics CSVs; they perform no training or inference and are not included in the historical v1.0.0 source tag.
