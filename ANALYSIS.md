# Analysis script guide / 分析脚本使用索引

This guide lists the current analysis tasks, their inputs, and their outputs. It describes existing implementations; listing a task does not mean its required data or weights are bundled with the repository. Training commands are documented separately in [README.md](README.md).

本索引列出当前分析任务的输入、输出及使用方式。列出任务不代表仓库已包含该任务所需的样本或权重；重新训练命令另见[首页 README](README.md)。

## 1. Environment and entry point / 环境与入口

Use the classification or segmentation environment documented in [README.md](README.md) and [environment/](environment/). Run commands from the repository root, using that environment's Python interpreter. Use `code/run.py` for the task identifiers in the tables below, not the individual task files directly. The standalone CSV summary in Section 3 uses its own command shown there.

按 [README.md](README.md) 和 [environment/](environment/) 配置相应的分类或分割环境。在仓库根目录使用该环境的 Python 执行命令。下列表格中的任务标识通过 `code/run.py` 运行，不直接执行任务目录内的文件。第 3 节的独立 CSV 汇总脚本使用该节单独列出的命令。

```powershell
python code/run.py --list
```

For any task identifier in the tables, use this template, replacing `TASK` and the output directory:

表格中的任务标识对应以下通用命令，将 `TASK` 替换为完整任务标识，并选择输出目录：

```powershell
python code/run.py TASK --output-root result/reproduced/my_analysis
```

`OUT` below means the selected `--output-root`. Use a fresh output directory for a new analysis; some tasks refuse existing outputs, while others overwrite their own generated files. Use the same output directory for explicitly linked steps such as Exp-1 ROC followed by its confusion-matrix plot. Console-only tasks print their results without creating a report file.

下文 `OUT` 指 `--output-root` 指定的目录。新一次分析使用新的输出目录；部分任务拒绝已有输出，另一些任务会覆盖自己生成的文件。明确衔接的步骤（如 Exp-1 ROC 后接混淆矩阵绘图）使用同一输出目录。仅终端输出的任务不会自动生成报告文件。

`--runs-root` is used by the classification summary tasks through `RUNS`. It does not redirect all tasks: most plotting scripts read the fixed historical `result/` paths below. `--output-root` changes output placement, not input data locations.

分类汇总任务通过 `RUNS` 使用 `--runs-root`；该参数并不重定向所有任务，多数绘图脚本读取下文固定的历史 `result/` 路径。`--output-root` 只改变输出位置，不改变数据输入路径。

## 2. Input locations / 输入位置

| Short name / 简称 | Repository-relative location / 仓库相对路径 |
|---|---|
| C1 | `result/fusion_exp1_100ep_seed{seed}_final_multiseed/`; seeds / 种子：`31415`, `27182`, `16180` |
| C2 | `result/fusion_exp2_100ep_seed{seed}_final_multiseed/`; seeds / 种子：`48271`, `59317`, `84629` |
| S3 | `result/seg3d_unet_locked_volume_ppv_seed42_20260911/` |
| D1 | `data/classification_exp1/processed_dataset/` |
| D2 | `data/classification_exp2/img17Se/` |
| D2D | `data/segmentation_2d/seg2D_clean_v2/` |
| D3D | `data/segmentation_3d/seg3D_clean/` |

Resource access is described in [DOWNLOADS.md](DOWNLOADS.md). The [classification preparation script](code/preparation/classification/README.md) organizes existing 9-channel and 17-channel arrays using the study-specific file selection. The [segmentation preparation scripts](code/preparation/README.md) organize existing split segmentation inputs. These scripts require the documented prepared input structures; they do not perform a complete raw-image-to-training conversion.

资源获取见 [DOWNLOADS.md](DOWNLOADS.md)。[分类准备脚本](code/preparation/classification/README.md)按本研究的文件选择规则整理已有九通道、十七通道数组；[分割准备脚本](code/preparation/README.md)整理已有划分的分割数据。这些脚本要求输入符合说明中的既有预处理结构，不完成从原始图像到训练输入的全部转换。

## 3. Analyses using saved records / 使用已有结果记录的分析


### Prediction-only classification summary / 仅凭预测记录汇总分类指标

[Script and usage instructions](code/analysis/README.md). This standalone script reads only the three `test_predictions_two_class.csv` files for the selected experiment (C1 or C2). It uses the Python standard library and requires no sample data, model weights, or completion manifests.

[脚本与使用说明](code/analysis/README.md)。该独立脚本只读取所选实验（C1 或 C2）的三份 `test_predictions_two_class.csv`，仅使用 Python 标准库，不需要样本、模型权重或完整运行清单。

Run from the repository root / 在仓库根目录运行：

```powershell
python code/analysis/summarize_predictions.py --experiment exp1 --output result/prediction_statistics_exp1
python code/analysis/summarize_predictions.py --experiment exp2 --output result/prediction_statistics_exp2
```

Outputs in each selected directory: `per_seed.csv`, `mean_sd.csv`, and `input_record.json`. They contain per-run metrics and confusion counts, three-run means and sample standard deviations (ddof=1), and input SHA-256 fingerprints. AUC uses flattened two-column probabilities and one-hot labels, following the existing classification metric implementation.

每个指定目录输出 `per_seed.csv`、`mean_sd.csv` 和 `input_record.json`，分别记录各次运行指标与混淆计数、三次运行的均值与样本标准差（ddof=1），以及输入文件的 SHA-256 指纹。AUC 沿用现有分类指标实现，将两列概率与独热标签分别展平后计算。

Use a new output directory. Add `--runs-root "D:/my_records"` if records are stored elsewhere. This command recomputes saved predictions; the full-record audits in Section 4 remain separate.

请使用新的输出目录。记录位于其他位置时，添加 `--runs-root "D:/my_records"`。此命令重算已有预测记录；第 4 节的完整记录审计另行保留。

### Plotting and case selection / 绘图与病例筛选

These tasks do not load model checkpoints or raw sample arrays. They still require the listed records and the appropriate Python dependencies.

以下任务不加载模型检查点或原始样本数组，但仍需表中所列记录及相应 Python 依赖。

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `classification/fig2_3_exp1_roc` | All three C1 `test_predictions_two_class.csv` files / C1 三个种子的预测记录 | `OUT/exp1_mean_roc.png`, `exp1_auc_summary.csv`, `exp1_confusion_per_seed.csv`, `exp1_mean_confusion_matrix.csv`, report / 报告 |
| `classification/fig2_confusion_matrix` | `OUT/exp1_mean_confusion_matrix.csv`; fallback / 回退：`result/paper_outputs/exp1_figures/exp1_mean_confusion_matrix.csv` | `OUT/exp1_mean_confusion_matrix.png` |
| `classification/fig4_5_exp2_roc` | All three C2 `test_predictions_two_class.csv` and `author_style_metrics.csv` files / C2 三组预测及指标记录 | `OUT/exp2_roc/`: ROC, confusion-matrix PNGs, AUC and confusion CSVs, report / ROC、混淆矩阵图、指标 CSV 和报告 |
| `classification/fig4_confusion_matrix` | All three C2 `test_predictions_two_class.csv` files / C2 三组预测记录 | `OUT/exp2_mean_confusion_matrix_exact.csv`, `exp2_mean_confusion_matrix_integer.csv`, `exp2_mean_confusion_matrix_integer.png`, note / 说明 |
| `classification/exp2_false_positive_cases` | All three C2 `test_predictions_two_class.csv` files / C2 三组预测记录 | Three `OUT/exp2_top10_*.csv` files: single-run cases, persistent cases and details / 单次、持续假阳性及详细记录 |
| `segmentation_3d/fig13_volume_selection` | S3 `test_predictions.csv` | `OUT/volume_overseg_two_level_selection/all_eligible_volumes_ranked.csv` and `TOP3_volumes_two_level_selection.csv` |

For classification prediction CSVs, preserve the original columns, including `True Label`, `Predicted Label`, `Benign Probability` and `Malignant Probability`. Case-selection tasks also use sample-identifying columns. Do not substitute a summary-only CSV for the per-sample predictions.

分类预测 CSV 应保留原列名，包括上述标签与两类概率字段；病例筛选还使用样本标识列。不能用仅含汇总指标的 CSV 替代逐样本预测记录。

Example: regenerate Exp-1 ROC and then its confusion-matrix plot / 示例：先生成 Exp-1 ROC，再生成混淆矩阵图：

```powershell
python code/run.py classification/fig2_3_exp1_roc --output-root result/reproduced/exp1_record_analysis
python code/run.py classification/fig2_confusion_matrix --output-root result/reproduced/exp1_record_analysis
```

Example: regenerate Exp-2 ROC / 示例：重新生成 Exp-2 ROC：

```powershell
python code/run.py classification/fig4_5_exp2_roc --output-root result/reproduced/exp2_record_analysis
```

The Exp-2 ROC task creates an `exp2_roc` subdirectory and requires it not to exist. These examples recalculate saved predictions; they do not rerun training or establish the original paper's evaluation procedure.

Exp-2 ROC 任务会新建 `exp2_roc` 子目录，要求该子目录尚不存在。以上示例重算已有预测记录，不重新训练，也不能据此确认原论文的评估实现。

## 4. Full-record summaries and audits / 完整记录汇总与审计

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `classification/table4_exp1_summary` | Complete C1 runs: completion manifest and every file it references, training history, prediction/metric CSVs, configuration and `best_fusion_model.pth` / C1 完整运行记录及清单引用文件，包含权重 | `OUT/exp1_final_multiseed_summary/`: per-seed metrics, mean/SD and comparison CSVs / 各种子指标、均值与标准差、对照 CSV |
| `classification/table5_exp2_summary` | Complete C2 runs with the corresponding completion manifests, referenced files and weights / C2 完整运行记录、清单引用文件及权重 | `OUT/exp2_final_multiseed_summary/`: per-seed metrics, mean/SD and comparison CSVs / 各种子指标、均值与标准差、对照 CSV |
| `segmentation_3d/table7_result_audit` | S3 history, predictions, metrics, `run_config.txt`, `training_log.txt`, `code_snapshot/`, `best_model.pth`, `last_checkpoint.pth` / 三维运行记录、代码快照及两个检查点 | `OUT/audit_results.json`, `recomputed_metric_check.csv`, `reconstructed_counts.csv`, `evidence_chain.csv`, `paper_vs_reproduction.csv`, before/after hashes / 前后哈希记录 |

The current classification summary functions verify the completed-run file hashes and require checkpoints. They are not CSV-only summary commands. Missing weights or other manifest-listed files will cause verification to fail. Keep these checks when performing a full audit; do not treat a partial record set as a verified complete run.

当前分类汇总函数核对已完成运行的文件哈希，并要求检查点存在，因此不是只需 CSV 的汇总命令。缺少权重或清单引用文件时会失败。完整审计应保留这些检查，不能把不完整记录视为已验证的完整运行。

```powershell
python code/run.py classification/table4_exp1_summary --output-root result/reproduced/exp1_full_summary
python code/run.py classification/table5_exp2_summary --output-root result/reproduced/exp2_full_summary
python code/run.py segmentation_3d/table7_result_audit --output-root result/reproduced/seg3d_full_audit
```

## 5. Analyses requiring prepared samples / 需要预处理样本的分析

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `classification/channel_mapping` | D1 and D2 test arrays / 两组分类测试数组；no weights / 无需权重 | `OUT/fp_fn_cases/exp1_exp2_channel_mapping_verification.csv` |
| `classification/subtraction_check` | D1 and D2 test arrays / 两组分类测试数组；no weights / 无需权重 | `OUT/fp_fn_cases/subtraction_formula_all_candidates.csv`, `subtraction_formula_best_mapping.csv` |
| `classification/fig10_11_case_source` | D2 test data and C2 seed59317 `code_snapshot/data_loader_exp2.py` / 测试数据及历史读取代码；no weights / 无需权重 | Console: sample index 96 source / 终端显示索引 96 的样本来源 |
| `classification/fig10_11_gradcam` | D2 test data; C2 seed59317 predictions, `best_fusion_model.pth`, and required model/loader snapshot modules / 样本、预测 CSV、模型权重和快照模块 | `OUT/exp2_seed59317_index96_gradcam/`: input, Grad-CAM, channel PNGs and metadata / 输入图、热力图、通道图与元数据 |
| `segmentation_3d/protocol_checks` | D3D all splits and current segmentation modules / 全部分割数据及当前代码；no trained weights / 无需训练权重 | `OUT/test_report.json`, synthetic `test_metrics.csv`, console / 合成测试 CSV 及终端结果 |
| `segmentation_3d/zero_padding_check` | D3D volumes / 三维体积；no weights / 无需权重 | Console zero-slice statistics / 终端零切片统计 |
| `segmentation_3d/source_slice_matching` | D3D, candidate D2D images and current code / 三维体积、候选二维图像及当前代码；no weights / 无需权重 | Console matching results / 终端匹配结果 |
| `segmentation_3d/rgb_intensity_check` | The specific paired D2D images and D3D volume selected in the script / 脚本指定的二维图像及三维体积；no weights / 无需权重 | Console intensity comparisons / 终端强度比较 |
| `segmentation_3d/rgb_prediction_impact` | The selected D2D/D3D images and mask, S3 `best_model.pth`, current 3D model implementation / 指定样本和掩膜、权重及三维模型实现 | Console prediction-impact comparisons / 终端预测影响比较 |
| `segmentation_3d/threshold_analysis` | D3D test data, S3 `best_model.pth` and `test_metrics.csv`, current model/dataset/training modules / 测试样本、权重、原指标及相关代码 | `OUT/threshold_sensitivity_volume_ppv.csv` |

These are study-specific checks; several select fixed cases or assert the study's expected shapes and counts. Changing datasets is not equivalent to rerunning the reported analysis. Inference tasks also require compatible model code and dependencies. Providing only a data link does not supply these inputs automatically.

这些是针对本研究的核查，部分任务选择固定病例或检查特定形状与数量。换用其他数据不等于复现本研究分析。推理任务还需要兼容的模型代码与依赖，提供数据链接不会自动补齐这些输入。

## 6. 3D example figures / 三维示例图

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `segmentation_3d/fig13_predictions` | D3D test data; S3 `test_predictions.csv`, `best_model.pth`, model and dataset snapshots / 测试数据、预测记录、权重及快照 | `OUT/top3_overseg_maxfp_v3/`: prediction PNGs, slice CSVs and summary / 预测图、切片统计及汇总 |
| `segmentation_3d/fig13_slice_selection` | Same inputs as `fig13_predictions`; it executes that script internally / 同上，内部会执行预测绘图脚本 | `OUT/volume55_slice_overseg_selection/`: `all_slices_volume55.csv`, `eligible_slices_ranked.csv`, `SELECTED_slice.csv`, representative PNG / 代表切片图；also prediction outputs / 同时生成预测输出 |
| `segmentation_3d/fig13_layout` | Existing PNG from `OUT/volume55_slice_overseg_selection/`; fallback to the same subdirectory under S3 / 已有 PNG，缺失时读取 S3 下同名子目录 | `OUT/representative_overseg_slice_2x2.png`; no model or arrays loaded by this layout task / 排版任务本身不加载模型或样本数组 |

With all inference inputs available, run slice selection followed by layout in the same output root. Running layout alone may use the retained historical PNG rather than newly generated predictions.

具备全部推理输入后，先运行切片选择，再使用相同输出目录运行排版。单独运行排版时，可能使用已保存的历史 PNG，而非新生成的预测结果。

```powershell
python code/run.py segmentation_3d/fig13_slice_selection --output-root result/reproduced/seg3d_example
python code/run.py segmentation_3d/fig13_layout --output-root result/reproduced/seg3d_example
```

## 7. Reference evaluator / 参考评估器

`segmentation_2d/ppv_reference` is a reference evaluator adapted from the original materials, not a newly authored analysis implementation or the main training evaluator. It reads separately exported predicted masks and ground-truth masks and prints precision, recall, accuracy and F1. Install its extra dependencies from `environment/reference_evaluator.txt` and provide matching mask directories explicitly:

`segmentation_2d/ppv_reference` 是基于原始材料的参考评估器，不归为本项目新写的分析实现，也不是主训练评估入口。它读取单独导出的预测掩膜和真实掩膜，打印 precision、recall、accuracy 和 F1。需安装额外依赖，并明确指定配对的掩膜目录：

```powershell
python -m pip install -r environment/reference_evaluator.txt
python code/run.py segmentation_2d/ppv_reference -- --test_images "D:/exported_masks/predictions" --ground_truth_images "D:/exported_masks/ground_truth"
```

For original sources and redistribution notes, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This functional index does not assign a new license or independent authorship to upstream model, training or evaluation code. For previously executed validation and its limitations, see [VALIDATION.md](VALIDATION.md).

原始来源及再分发说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本功能索引不为原作者模型、训练或评估代码赋予新许可或独立原创身份。已有运行验证及其范围见 [VALIDATION.md](VALIDATION.md)。
