# Reconstruct reported metrics from saved test statistics

These Python-standard-library tools verify the eight formal reproduction runs without training, inference, checkpoint loading or patient image access. Supply locally held CSV files. Outputs are aggregate JSON reports, with input hashes and local paths; inspect those paths before sharing a report publicly. Inputs are not redistributed here.

## Classification — Table 4 and supplementary malignant-class AUROC

The runs root contains `fusion_exp1_100ep_seed{31415,27182,16180}_final_multiseed` and `fusion_exp2_100ep_seed{48271,59317,84629}_final_multiseed`. Each requires `test_predictions_two_class.csv` and `author_style_metrics.csv`.

Required prediction columns: `Index`, `True Label`, `Predicted Label`, `Benign Probability`, `Malignant Probability`. Formal runs contain 403 records, 289 malignant and 114 benign. Index must be unique. Malignant is 1, benign is 0. Probabilities must be finite, in [0,1], and sum to one within 1e-6.

```text
python code/analysis/classification_metrics_from_predictions.py --experiment exp1 --runs-root "<classification results parent>" --output "<new output>/exp1"
python code/analysis/classification_metrics_from_predictions.py --experiment exp2 --runs-root "<classification results parent>" --output "<new output>/exp2"
```

`AUC_flattened_two_column` reproduces the existing released-code-compatible aggregation. `AUROC_malignant_class` uses only malignant probability and the binary true label. Both use ranks with half credit for ties; the latter is independently verified by all positive-negative pairs. Mean and sample SD (ddof=1) are computed across the three seeds, not pooled predictions; SD is not a confidence interval. The original selection of checkpoints remains unchanged. The original published BreastDM AUC aggregation is not established by this calculation.

## Segmentation — Table 5

Each run requires `test_predictions.csv` and `test_metrics.csv`. Required prediction fields are `index`, `dice`, `iou`, `ppv`, `gt_positive_pixels`, `pred_positive_pixels`. Preserve original contiguous index order starting at zero.

```text
python code/analysis/segmentation_metrics_from_predictions.py --run-dir "<seg2d_unet_seed123_gpu_20260911>" --mode 2d --spatial-size 224 --confirm-no-ignored-voxels --output "<new output>/2d"
python code/analysis/segmentation_metrics_from_predictions.py --run-dir "<seg3d_unet_locked_volume_ppv_seed42_20260911>" --mode 3d --spatial-size 224 --depth 8 --confirm-no-ignored-voxels --output "<new output>/3d"
```

Only use the confirmation flag after establishing the binary, fixed-size, no-ignored-voxel evaluation protocol. TP is reconstructed from saved PPV times predicted-positive count; counts must be integral and consistent with saved Dice/IoU. TN depends on the specified evaluation size. Global two-class mIoU is reconstructed from pooled counts, not averaged foreground IoUs.

2D uses sequential float32 DSC accumulation, global PPV and eligible-image PPV (excluding TP=0 or TN=0). 3D averages per-volume float32 DSC and includes zero-prediction volumes with PPV=0. These reproduce the audited formal protocols. They are not a general evaluator for arbitrary masks, dimensions, ignored voxels or batching schemes.

## Outputs and limitations

Every invocation creates `verification.json` in a new output directory. Existing directories are refused. Reports include metrics, input SHA-256 and script SHA-256, comparison residuals and scope. Segmentation allows absolute floating-point differences up to 1e-7 and separately records whether equality is exact.

The scripts verify saved-statistics consistency, not independent segmentation-mask accuracy, raw-data preparation, historical environment equivalence, or published-result provenance. Table 6 provenance requires documentary evidence separately. Obtain original data through https://github.com/smallboy-code/Breast-cancer-dataset ; original paper https://doi.org/10.1016/j.compbiomed.2023.107255 . No original or preprocessed dataset is included.

## Source lineage

Classification retains project statistics functions from `code/analysis/summarize_predictions.py` at commit `b26a3def1542dbaa0a1ec8997f6c5c96d34ce856` (source SHA-256 `5b2f7196bdd958f24f1b2cc2c9d9e4c563513923c1d7029a8e0cdfd2848a2932`), with additional AUROC and verification logic. Segmentation is the standalone saved-count reconstruction prepared during the 2026-11-04 local verification. No external model/evaluator source is bundled, and no new license grant is asserted.
