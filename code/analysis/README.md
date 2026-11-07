# Saved prediction statistics

The standalone CSV summarizer is public and needs no private model or training source. Supply local prediction CSVs; full historical audits have additional evidence requirements.

`summarize_predictions.py` recomputes classification metrics from saved prediction CSVs. It requires only the Python standard library (Python 3.12 used for validation). It does not require sample data or model weights. The existing full-audit tasks remain separate.

The required prediction CSVs are private local inputs, not included in the current public tree. Generate them using [local generation](../LOCAL_GENERATION.md), or supply your own private records.

## Inputs

Under `--runs-root` (default: this repository's `result/`), each run must contain `test_predictions_two_class.csv`:

```text
fusion_exp1_100ep_seed31415_final_multiseed/test_predictions_two_class.csv
fusion_exp1_100ep_seed27182_final_multiseed/test_predictions_two_class.csv
fusion_exp1_100ep_seed16180_final_multiseed/test_predictions_two_class.csv
fusion_exp2_100ep_seed48271_final_multiseed/test_predictions_two_class.csv
fusion_exp2_100ep_seed59317_final_multiseed/test_predictions_two_class.csv
fusion_exp2_100ep_seed84629_final_multiseed/test_predictions_two_class.csv
```

Only the three files for the selected experiment are read. Required columns: `True Label`, `Predicted Label`, `Benign Probability`, `Malignant Probability`. Labels are benign=0 and malignant=1. Predictions use the saved hard labels; probabilities are used for AUC.

## Commands

Run directly from the repository root:

```bash
python code/analysis/summarize_predictions.py --experiment exp1 --runs-root .local/reproduction/result --output .local/statistics_exp1
python code/analysis/summarize_predictions.py --experiment exp2 --runs-root .local/reproduction/result --output .local/statistics_exp2
```

For records stored elsewhere, add `--runs-root "D:/my_records"`. Choose a new output directory for each invocation; existing output directories are refused. This standalone script is not a task registered with `code/run.py`.

## Outputs and definitions

- `per_seed.csv`: per-run Accuracy, Sensitivity, Specificity, malignant-class Precision, weighted Precision, AUC, TP/TN/FP/FN and sample count.
- `mean_sd.csv`: mean and sample standard deviation across the three runs (ddof=1); values are fractions, not percentages.
- `input_record.json`: input paths, SHA-256 fingerprints and calculation scope.

AUC uses both probability columns and flattened one-hot labels, following this project's existing classification metric implementation. It is not the AUC calculated using only malignant-class probabilities. Weighted precision uses class support as weights and zero for undefined class precision. Other undefined ratios are `NaN`; a summary metric is `NaN` if any run's value is undefined.

This checks the saved prediction values and recomputes statistics; it does not verify training, checkpoint selection, data splitting, or the original authors' evaluation procedure. Use the full-audit tasks listed in [full-record audits](../../docs/WORKFLOW_GUIDE.md#full-audits) when the required experimental records are available.

## Formal-run verification and both AUC definitions

The new [saved-prediction verification guide](SAVED_PREDICTION_VERIFICATION.md) covers Tables 4–5: `classification_metrics_from_predictions.py` verifies the six formal classification runs and additionally reports malignant-class AUROC; `segmentation_metrics_from_predictions.py` verifies the formal 2D/3D aggregate metrics with distinct aggregation rules selected through `--mode 2d/3d`. Both use the Python standard library and require private saved statistics plus the corresponding official metrics CSV. Existing `summarize_predictions.py` commands remain available with their original flattened-AUC definition.

These are saved-statistics consistency checks, not retraining or inference. They do not establish the original published evaluator or complete reconstruction from raw data. Outputs include local input paths and must be reviewed before public sharing. All commands require a new output directory.
