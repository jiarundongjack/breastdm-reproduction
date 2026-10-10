# Saved prediction statistics / 已有预测记录统计

`summarize_predictions.py` recomputes classification metrics from saved prediction CSVs. It requires only the Python standard library (Python 3.12 used for validation). It does not require sample data or model weights. The existing full-audit tasks remain separate.

本脚本仅凭已有预测 CSV 重算分类指标，只依赖 Python 标准库（使用 Python 3.12 验证），不需要样本或模型权重。原有完整审计任务另行保留。

## Inputs / 输入

Under `--runs-root` (default: this repository's `result/`), each run must contain `test_predictions_two_class.csv`:

在 `--runs-root` 指定的目录下（默认为本仓库 `result/`），各次运行须包含 `test_predictions_two_class.csv`：

```text
fusion_exp1_100ep_seed31415_final_multiseed/test_predictions_two_class.csv
fusion_exp1_100ep_seed27182_final_multiseed/test_predictions_two_class.csv
fusion_exp1_100ep_seed16180_final_multiseed/test_predictions_two_class.csv
fusion_exp2_100ep_seed48271_final_multiseed/test_predictions_two_class.csv
fusion_exp2_100ep_seed59317_final_multiseed/test_predictions_two_class.csv
fusion_exp2_100ep_seed84629_final_multiseed/test_predictions_two_class.csv
```

Only the three files for the selected experiment are read. Required columns: `True Label`, `Predicted Label`, `Benign Probability`, `Malignant Probability`. Labels are benign=0 and malignant=1. Predictions use the saved hard labels; probabilities are used for AUC.

每次只读取所选实验的三个文件。必需列为上述四列；良性为 0，恶性为 1。分类预测采用 CSV 中保存的预测标签，概率用于计算 AUC。

## Commands / 运行命令

Run directly from the repository root:

在仓库根目录直接运行：

```bash
python code/analysis/summarize_predictions.py --experiment exp1 --output result/prediction_statistics_exp1
python code/analysis/summarize_predictions.py --experiment exp2 --output result/prediction_statistics_exp2
```

For records stored elsewhere, add `--runs-root "D:/my_records"`. Choose a new output directory for each invocation; existing output directories are refused. This standalone script is not a task registered with `code/run.py`.

若记录在其他位置，添加 `--runs-root "D:/my_records"`。每次指定新的输出目录；脚本拒绝覆盖已有目录。该脚本独立运行，不通过 `code/run.py` 调用。

## Outputs and definitions / 输出与定义

- `per_seed.csv`: per-run Accuracy, Sensitivity, Specificity, malignant-class Precision, weighted Precision, AUC, TP/TN/FP/FN and sample count. / 每次运行的六项指标、混淆计数与样本数。
- `mean_sd.csv`: mean and sample standard deviation across the three runs (ddof=1); values are fractions, not percentages. / 三次运行的均值及样本标准差，使用小数而非百分数。
- `input_record.json`: input paths, SHA-256 fingerprints and calculation scope. / 输入路径、文件指纹及计算范围。

AUC uses both probability columns and flattened one-hot labels, following this project's existing classification metric implementation. It is not the AUC calculated using only malignant-class probabilities. Weighted precision uses class support as weights and zero for undefined class precision. Other undefined ratios are `NaN`; a summary metric is `NaN` if any run's value is undefined.

AUC 沿用本项目现有分类指标实现，将两列概率与独热标签分别展平后计算，不等同于仅使用恶性概率的 AUC。加权精确率按类别样本数加权，单类精确率无定义时按零处理。其他无定义比值记为 `NaN`；若某次运行的指标无定义，其汇总均值和标准差也记为 `NaN`。

This checks the saved prediction values and recomputes statistics; it does not verify training, checkpoint selection, data splitting, or the original authors' evaluation procedure. Use the full-audit tasks listed in [ANALYSIS.md](../../ANALYSIS.md) when the required experimental records are available.

本脚本检查已有预测数值并重算统计，不验证训练过程、检查点选择、数据划分或原作者的评估流程。具备完整实验记录时，可使用 [ANALYSIS.md](../../ANALYSIS.md) 中列出的完整审计任务。
