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
## Original resources / 原始资源

Original BreastDM paper: [https://doi.org/10.1016/j.compbiomed.2023.107255](https://doi.org/10.1016/j.compbiomed.2023.107255).

This repository does not redistribute original or preprocessed BreastDM data. Case-level CSVs, case images, masks, predictions and other BreastDM-derived patient-level files with unclear redistribution permission are excluded from the current `main` tree. Obtain data through the [original authors' repository and data links](https://github.com/smallboy-code/Breast-cancer-dataset), then generate these outputs privately using the existing scripts and the local-generation entry points linked below. Aggregate metrics and statistical figures are retained.

本仓库不重新分发原始或预处理后的 BreastDM 数据。当前 `main` 不公开再分发许可不明确的病例级 CSV、病例图像、mask、预测及其他 BreastDM 派生病例文件。请通过[原作者仓库及数据入口](https://github.com/smallboy-code/Breast-cancer-dataset)自行获取数据，再按下述本地生成入口私下生成输出；公开仓库保留汇总指标及统计图。

See [local generation commands](../code/LOCAL_GENERATION.md) for manifests, predictions, Grad-CAM, segmentation examples and case-selection tables.

清单、预测、Grad-CAM、分割病例图及筛选表的命令见[本地生成说明](../code/LOCAL_GENERATION.md)。

The original BreastDM code and data access information are available from the [original authors' repository](https://github.com/smallboy-code/Breast-cancer-dataset). Please obtain the data through the download links provided there and follow the original providers' access and use requirements.

BreastDM 原始代码与数据获取信息见[原作者仓库](https://github.com/smallboy-code/Breast-cancer-dataset)。请通过其中提供的下载链接获取数据，并遵守原提供方的访问和使用要求。

Original-author data and complete external/mixed implementations with unresolved redistribution permission are not republished in the current tree. Project-specific scripts are public, and licensed third-party files retain their attribution and license texts. See [training modifications](../provenance/TRAINING_MODIFICATIONS.md) for preserved reproduction-specific protocol facts without republishing the excluded training bodies.

当前树不重新分发许可未明确的原作者数据及外部/混合来源完整实现。本项目专用脚本公开，许可明确的第三方文件保留来源与许可文本。[训练改动说明](../provenance/TRAINING_MODIFICATIONS.md)记录复现协议，不复制被排除的完整训练实现。

See [DOWNLOADS.md](../weight%20download%20and%20restore/DOWNLOADS.md) for resource access instructions and [model-branch license guide](../weight%20download%20and%20restore/Classification%20model%20branch%20licenses/README.md) for upstream model-branch licensing information.

资源获取说明见 [DOWNLOADS.md](../weight%20download%20and%20restore/DOWNLOADS.md)，模型分支上游许可见 [model-branch license guide](../weight%20download%20and%20restore/Classification%20model%20branch%20licenses/README.md)。

The reported experiments used task-specific prepared datasets. Files obtained from the original resources require preparation before they can be used with the experiment commands below.

论文中的实验使用按任务整理的预处理数据。从原始资源获取的文件需要经过整理，才能用于下述实验命令。

<a id="classification-preparation"></a>
## Classification data preparation / 分类数据整理

See [classification preparation instructions](../code/preparation/classification/README.md) for the existing 9-channel and 17-channel input layout, the study-specific file selection, validation checks, and copy commands.

已有九通道、十七通道输入的目录结构、本研究的文件选择规则、检查项目和复制命令，见[分类数据整理说明](../code/preparation/classification/README.md)。

The script organizes existing split `.npy` arrays and preserves their contents. It does not generate channels from raw images or create new data splits. The inspected local inputs reproduce the formal input files exactly; the current upstream download layout has not been verified.

脚本整理已有划分的 `.npy` 数组并保留文件内容，不从原始图像生成通道，也不重新划分数据。已核对的本地输入可整理出与正式输入完全一致的文件；尚未核实上游当前下载包结构。

<a id="segmentation-preparation"></a>
## Segmentation data preparation / 分割数据整理

See [segmentation preparation instructions](../code/preparation/segmentation/README.md) for the required input structure, patient-selection rules, and commands for the 2D and 3D preparation scripts.

二维、三维分割准备脚本所需的输入结构、患者筛选规则和运行命令，见[分割数据整理说明](../code/preparation/segmentation/README.md)。

These scripts organize existing split segmentation data; they do not perform the complete conversion from original downloads to training inputs.

这些脚本用于整理已有划分的分割数据，不负责从原始下载物到训练输入的全部转换。

<a id="contents"></a>
## Contents / 目录

The paths below describe the local directory structure used by the experiments. A listed path does not imply that its data or model files are included in this repository.

下表列出实验使用的本地目录结构。列出某一路径不代表本仓库包含该路径下的数据或模型文件。

| Path / 路径 | Purpose / 用途 |
|---|---|
| `data/classification_exp1/processed_dataset` | Exp-1, nine channels / 实验一9通道输入 |
| `data/classification_exp2/img17Se` | Exp-2, seventeen channels / 实验二17通道输入 |
| `data/segmentation_2d/seg2D_clean_v2` | 2D images and masks / 2D图像及标签 |
| `data/segmentation_3d/seg3D_clean` | 3D volumes and masks / 3D体积及标签 |
| `code/classification`, `code/segmentation_2d`, `code/segmentation_3d` | Scientific task implementations and shared modules; see the cleanup inventory / 科学任务实现与共享模块，见整理清单 |
| `code/preparation` | Existing split classification and segmentation input preparation / 已有划分的分类与分割输入整理 |
| `code/analysis` | Classification statistics from saved prediction CSVs / 基于已有预测 CSV 的分类统计 |
| `code/pretrained` | Two classification initialization checkpoints / 分类训练使用的两份预训练权重 |
| `result/fusion_*`, `result/seg*` | Eight unchanged formal runs / 八次未经改写的正式运行记录 |
| `result/paper_outputs` | Canonical cross-seed summaries, flattened ROC and confusion matrices / 正式多种子汇总、展平 ROC 与混淆矩阵 |
| `result/reproduced` | Retained diagnostic/reproduction evidence; see result/README.md / 保留的诊断与复现证据，见结果说明 |
| `environment` | Observed dependencies for two environments / 两套环境的依赖记录 |

Use **`result`**, singular. Historical configuration files retain their original absolute
paths and auxiliary metric fields as evidence; they are not executable configuration for this release.
统一使用单数`result`。历史配置中的绝对路径和辅助指标字段作为原始凭据保留，不作为本发布版运行配置。

<a id="environments"></a>
## Environments / 环境

The `classification.lock.txt` and `segmentation.lock.txt` files record the complete
package versions resolved in fresh Python 3.11.5 environments during release preparation.
They are release-validation records, not retrospective records of the original training installations.
两份`.lock.txt`记录发布检查中新建Python 3.11.5环境的完整依赖，不追溯替代原训练环境记录。
For the pinned release environment, substitute the corresponding `.lock.txt` for `.txt`
in the installation commands below.
需要固定完整依赖时，可将下面安装命令中的对应`.txt`替换为`.lock.txt`。

The documented local setup records Python 3.11.5, Windows 11, an NVIDIA RTX PRO 3000
Blackwell Generation Laptop GPU (12 GB), classification PyTorch 2.10.0+cu130,
and segmentation PyTorch 2.7.1+cu128. These reported/observed versions do not independently
establish the segmentation training-day environment. See the [environment evidence guide](../environment/README.md)
for the distinction between run records, later observations and release-validation locks.
Use separate environments for these tasks.
上述配置为已报告/观察到的本地环境，不独立证明分割训练当日的完整环境；运行记录、后采集环境与发布验证锁文件的证据层级见环境说明。

Example installation with Python 3.11 / 使用Python 3.11的安装示例：

```powershell
python -m venv .venv_classification
.\.venv_classification\Scripts\python.exe -m pip install -r environment/classification.txt
python -m venv .venv_segmentation
.\.venv_segmentation\Scripts\python.exe -m pip install -r environment/segmentation.txt
```

Both dependency sets were installed in fresh Python 3.11.5 virtual environments during the release checks documented in [reproduction checks](../reproduction%20checks/README.md). This does not establish identical ancillary package versions during historical training or successful installation on another computer.
两套依赖已在发布检查中新建的 Python 3.11.5 虚拟环境中安装验证，详见[复现检查记录](../reproduction%20checks/README.md)；这不证明历史训练时所有辅助包版本完全一致，也不代表已在另一台电脑上验证。
Use the relevant interpreter for every command below; `python` denotes that interpreter.
以下`python`均指相应任务环境中的解释器。

<a id="entry-point"></a>
## Entry point / 统一入口

Run commands from the repository root / 在仓库根目录运行：

```powershell
python code/run.py --list
python code/run.py classification/exp1_training -- --help
python code/run.py segmentation_3d/table7_fig8_9_12_training -- --help
```

Do not execute files inside task folders directly. `code/run.py` copies the selected task's
modules into a temporary conventional package layout, launches them with the same interpreter,
and removes the temporary directory on completion. Canonical source files remain in their task folders.
分类与分割任务目录内的程序通过`code/run.py`运行；它临时组织导入结构，执行后清理临时目录，不改变任务文件夹中的源码。
Repository locations are resolved from the launcher, so no personal drive letters are required.
路径由入口文件定位，无需修改个人电脑盘符。Additional options follow `--` / 任务参数放在`--`之后。

The standalone scripts in `code/preparation/` and `code/analysis/` run directly with Python, using the commands in their respective READMEs. They are not registered tasks of `code/run.py`.

`code/preparation/` 和 `code/analysis/` 中的独立脚本按各自 README 的命令直接用 Python 运行，不通过 `code/run.py` 调用。

<a id="analysis"></a>
## Analysis scripts / 分析脚本使用指南

These tasks describe existing implementations; listing a task does not mean its samples or weights are included. This guide does not assign independent authorship or a new license to upstream code.

以下列出已有分析实现；列出任务不代表仓库包含所需样本或权重，也不为上游代码赋予独立原创身份或新许可。

### 1. Environment and entry point / 环境与入口

Use the classification or segmentation environment described in [Environments](#environments). Run commands from the repository root, using that environment's Python interpreter. Use `code/run.py` for the task identifiers in the tables below, not the individual task files directly. The standalone CSV summary in Section 3 uses its own command shown there.

按上文[环境说明](#environments)配置相应的分类或分割环境。在仓库根目录使用该环境的 Python 执行命令。下列表格中的任务标识通过 `code/run.py` 运行，不直接执行任务目录内的文件。第 3 节的独立 CSV 汇总脚本使用该节单独列出的命令。

For any task identifier in the tables, use this template, replacing `TASK` and the output directory:

表格中的任务标识对应以下通用命令，将 `TASK` 替换为完整任务标识，并选择输出目录：

```powershell
python code/run.py TASK --output-root result/reproduced/my_analysis
```

`OUT` below means the selected `--output-root`. Use a fresh output directory for a new analysis; some tasks refuse existing outputs, while others overwrite their own generated files. Use the same output directory only when a task explicitly consumes another task's generated outputs. Console-only tasks print their results without creating a report file.

下文 `OUT` 指 `--output-root` 指定的目录。新一次分析使用新的输出目录；部分任务拒绝已有输出，另一些任务会覆盖自己生成的文件。仅在任务明确读取前一步生成的输出时，衔接步骤才使用同一输出目录。仅终端输出的任务不会自动生成报告文件。

`--runs-root` is used by the classification summary tasks through `RUNS`. It does not redirect all tasks: most plotting scripts read the fixed historical `result/` paths below. `--output-root` changes output placement, not input data locations.

分类汇总任务通过 `RUNS` 使用 `--runs-root`；该参数并不重定向所有任务，多数绘图脚本读取下文固定的历史 `result/` 路径。`--output-root` 只改变输出位置，不改变数据输入路径。

### 2. Input locations / 输入位置

| Short name / 简称 | Repository-relative location / 仓库相对路径 |
|---|---|
| C1 | `result/fusion_exp1_100ep_seed{seed}_final_multiseed/`; seeds / 种子：`31415`, `27182`, `16180` |
| C2 | `result/fusion_exp2_100ep_seed{seed}_final_multiseed/`; seeds / 种子：`48271`, `59317`, `84629` |
| S3 | `result/seg3d_unet_locked_volume_ppv_seed42_20260911/` |
| D1 | `data/classification_exp1/processed_dataset/` |
| D2 | `data/classification_exp2/img17Se/` |
| D2D | `data/segmentation_2d/seg2D_clean_v2/` |
| D3D | `data/segmentation_3d/seg3D_clean/` |

The `data/` paths above are local input directories populated by the reader following the preparation instructions; sample data are not included in this repository. The preparation scripts organize existing prepared arrays or split segmentation inputs and do not perform a complete raw-image-to-training conversion.

上述 `data/` 路径指读者按照准备说明在本地整理数据后使用的输入目录，不表示本仓库包含样本。准备脚本整理已有预处理数组或已划分的分割数据，不完成从原始图像到训练输入的全部转换。

The patient-level input CSVs below are no longer distributed. First follow [local generation](../code/LOCAL_GENERATION.md). For case analyses, use the private copy's `.local/reproduction/code/run.py`; paths C1/C2/S2/S3/D1/D2/D2D/D3D then refer to that private copy. The commands in this section describe required inputs, not files bundled with the public repository.

以下病例级输入 CSV 不再公开提供，请先按[本地生成说明](../code/LOCAL_GENERATION.md)准备。病例分析使用私有副本的 `.local/reproduction/code/run.py`，对应运行与数据路径均指向私有副本。

### 3. Analyses using saved records / 使用已有结果记录的分析


#### Prediction-only classification summary / 仅凭预测记录汇总分类指标

Use the existing [CSV-only summary guide](../code/analysis/README.md) for commands, required columns, output schemas and the flattened-AUC definition. It reads three prediction CSVs for the selected experiment using the Python standard library, without images, weights or completion manifests. Full-record audits below have different requirements.

For checks against formal metric records and the additional malignant-class AUROC, use the [formal saved-statistics verification guide](../code/analysis/SAVED_PREDICTION_VERIFICATION.md). Both guides require new output directories and private local input records; their commands remain maintained at those entry points.

仅凭预测 CSV 的汇总命令及字段定义统一维护在上述指南；正式指标对照与恶性类 AUROC 使用独立的保存统计复核入口。完整运行审计的输入要求仍见下文。

#### Plotting and case selection / 绘图与病例筛选

These tasks do not load model checkpoints or raw sample arrays. They still require the listed records and the appropriate Python dependencies.

以下任务不加载模型检查点或原始样本数组，但仍需表中所列记录及相应 Python 依赖。

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `classification/exp1_mean_roc` | All three C1 `test_predictions_two_class.csv` files / C1 三个种子的预测记录 | `OUT/exp1_mean_roc.png`, `exp1_auc_summary.csv`, `exp1_confusion_per_seed.csv`, `exp1_mean_confusion_matrix.csv`, report / 报告 |
| `classification/exp2_mean_outputs` | All three C2 `test_predictions_two_class.csv` and `author_style_metrics.csv` files / C2 三组预测及指标记录 | `OUT/exp2_roc/`: ROC, confusion-matrix PNGs, AUC and confusion CSVs, report / ROC、混淆矩阵图、指标 CSV 和报告 |
| `classification/exp2_mean_confusion_matrix` | All three C2 `test_predictions_two_class.csv` files / C2 三组预测记录 | `OUT/exp2_mean_confusion_matrix_exact.csv`, `exp2_mean_confusion_matrix_integer.csv`, `exp2_mean_confusion_matrix_integer.png`, note / 说明 |
| `classification/exp2_false_positive_cases` | All three C2 `test_predictions_two_class.csv` files / C2 三组预测记录 | Three `OUT/exp2_top10_*.csv` files: single-run cases, persistent cases and details / 单次、持续假阳性及详细记录 |
| `segmentation_3d/oversegmentation_volume_selection` | S3 `test_predictions.csv` | `OUT/volume_overseg_two_level_selection/all_eligible_volumes_ranked.csv` and `TOP3_volumes_two_level_selection.csv` |

For classification prediction CSVs, preserve the original columns, including `True Label`, `Predicted Label`, `Benign Probability` and `Malignant Probability`. Case-selection tasks also use sample-identifying columns. Do not substitute a summary-only CSV for the per-sample predictions.

分类预测 CSV 应保留原列名，包括上述标签与两类概率字段；病例筛选还使用样本标识列。不能用仅含汇总指标的 CSV 替代逐样本预测记录。

Example: regenerate Exp-1 ROC and aggregate confusion counts / 示例：生成 Exp-1 ROC 与汇总混淆计数：

```powershell
python code/run.py classification/exp1_mean_roc --output-root result/reproduced/exp1_record_analysis
```

Example: regenerate Exp-2 ROC / 示例：重新生成 Exp-2 ROC：

```powershell
python code/run.py classification/exp2_mean_outputs --output-root result/reproduced/exp2_record_analysis
```

The Exp-2 ROC task creates an `exp2_roc` subdirectory and requires it not to exist. These examples recalculate saved predictions; they do not rerun training or establish the original paper's evaluation procedure.

Exp-2 ROC 任务会新建 `exp2_roc` 子目录，要求该子目录尚不存在。以上示例重算已有预测记录，不重新训练，也不能据此确认原论文的评估实现。

<a id="full-audits"></a>
### 4. Full-record summaries and audits / 完整记录汇总与审计

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

### 5. Analyses requiring prepared samples / 需要预处理样本的分析

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `classification/channel_mapping` | D1 and D2 test arrays / 两组分类测试数组；no weights / 无需权重 | `OUT/fp_fn_cases/exp1_exp2_channel_mapping_verification.csv` |
| `classification/subtraction_check` | D1 and D2 test arrays / 两组分类测试数组；no weights / 无需权重 | `OUT/fp_fn_cases/subtraction_formula_all_candidates.csv`, `subtraction_formula_best_mapping.csv` |
| `classification/exp2_seed59317_case_source` | D2 test data and C2 seed59317 `code_snapshot/data_loader_exp2.py` / 测试数据及历史读取代码；no weights / 无需权重 | Console: sample index 96 source / 终端显示索引 96 的样本来源 |
| `classification/fig10_11_gradcam` | D2 test data; C2 seed59317 predictions, `best_fusion_model.pth`, and required model/loader snapshot modules / 样本、预测 CSV、模型权重和快照模块 | `OUT/exp2_seed59317_index96_gradcam/`: input, Grad-CAM, channel PNGs and metadata / 输入图、热力图、通道图与元数据 |
| `segmentation_3d/protocol_checks` | D3D all splits and current segmentation modules / 全部分割数据及当前代码；no trained weights / 无需训练权重 | `OUT/test_report.json`, synthetic `test_metrics.csv`, console / 合成测试 CSV 及终端结果 |
| `segmentation_3d/zero_padding_check` | D3D volumes / 三维体积；no weights / 无需权重 | Console zero-slice statistics / 终端零切片统计 |
| `segmentation_3d/source_slice_matching` | D3D, candidate D2D images and current code / 三维体积、候选二维图像及当前代码；no weights / 无需权重 | Console matching results / 终端匹配结果 |
| `segmentation_3d/rgb_intensity_check` | The specific paired D2D images and D3D volume selected in the script / 脚本指定的二维图像及三维体积；no weights / 无需权重 | Console intensity comparisons / 终端强度比较 |
| `segmentation_3d/rgb_prediction_impact` | The selected D2D/D3D images and mask, S3 `best_model.pth`, current 3D model implementation / 指定样本和掩膜、权重及三维模型实现 | Console prediction-impact comparisons / 终端预测影响比较 |
| `segmentation_3d/threshold_analysis` | D3D test data, S3 `best_model.pth` and `test_metrics.csv`, current model/dataset/training modules / 测试样本、权重、原指标及相关代码 | `OUT/threshold_sensitivity_volume_ppv.csv` |

These are study-specific checks; several select fixed cases or assert the study's expected shapes and counts. Changing datasets is not equivalent to rerunning the reported analysis. Inference tasks also require compatible model code and dependencies. Providing only a data link does not supply these inputs automatically.

这些是针对本研究的核查，部分任务选择固定病例或检查特定形状与数量。换用其他数据不等于复现本研究分析。推理任务还需要兼容的模型代码与依赖，提供数据链接不会自动补齐这些输入。

### 6. 3D example figures / 三维示例图

| Task / 任务标识 | Required inputs / 必需输入 | Main outputs / 主要输出 |
|---|---|---|
| `segmentation_3d/fig13_predictions` | D3D test data; S3 `test_predictions.csv`, `best_model.pth`, model and dataset snapshots / 测试数据、预测记录、权重及快照 | `OUT/top3_overseg_maxfp_v3/`: prediction PNGs, slice CSVs and summary / 预测图、切片统计及汇总 |
| `segmentation_3d/fig13_slice_selection` | Same inputs as `fig13_predictions`; it executes that script internally / 同上，内部会执行预测绘图脚本 | `OUT/volume55_slice_overseg_selection/`: `all_slices_volume55.csv`, `eligible_slices_ranked.csv`, `SELECTED_slice.csv`, representative PNG / 代表切片图；also prediction outputs / 同时生成预测输出 |

With all inference inputs available, run the retained slice-selection task below. It also invokes prediction generation. Separate publication-layout helpers from earlier versions are no longer registered tasks; historical case PNGs are not distributed.

具备全部推理输入后，运行下列保留的切片筛选任务，它也会调用预测生成。早期独立论文排版工具不再属于当前注册任务；历史病例 PNG 不公开提供。

```powershell
python code/run.py segmentation_3d/fig13_slice_selection --output-root result/reproduced/seg3d_example
```

### 7. Reference evaluator / 参考评估器

`segmentation_2d/ppv_reference` is a reference evaluator adapted from the original materials, not a newly authored analysis implementation or the main training evaluator. It reads separately exported predicted masks and ground-truth masks and prints precision, recall, accuracy and F1. Install its extra dependencies from `environment/reference_evaluator.txt` and provide matching mask directories explicitly:

`segmentation_2d/ppv_reference` 是基于原始材料的参考评估器，不归为本项目新写的分析实现，也不是主训练评估入口。它读取单独导出的预测掩膜和真实掩膜，打印 precision、recall、accuracy 和 F1。需安装额外依赖，并明确指定配对的掩膜目录：

```powershell
python -m pip install -r environment/reference_evaluator.txt
python code/run.py segmentation_2d/ppv_reference -- --test_images "D:/exported_masks/predictions" --ground_truth_images "D:/exported_masks/ground_truth"
```

### Reported metric notes / 已报告指标说明

Exp-2 ROC uses flattened two-column probabilities and one-hot labels in the retained implementation: **0.8735928017 ± 0.0218863413**, with sample SD (`ddof=1`). This numerical agreement does not establish the original authors' evaluation procedure. The obsolete single-column ROC is not the final Exp-2 figure in `result/paper_outputs`.

实验二 ROC 沿用现有实现，将双列概率与独热标签分别展平：**0.8735928017 ± 0.0218863413**，标准差采用样本标准差。数值一致不代表已确认原作者的评估流程；旧单列 ROC 不作为最终图片收录。

3D PPV is an equally weighted mean over all 141 test volumes; empty predictions contribute zero, and no volume is excluded for TP=0 or TN=0. Formal PPV is **0.604198665930658**.

3D PPV 对全部 141 个测试体积等权平均，空预测记零，不因 TP 或 TN 为零筛选体积；正式结果为 **0.604198665930658**。



<a id="training"></a>
## Retrain the eight runs / 重新训练八次实验

Historical runs under `result` are protected. New training goes under `result/new_runs`.
For each classification experiment, run its seeds in the listed order; provenance checks compare
successive runs' data, code, and environment. Do not mix new runs with historical runs.
历史结果保持不动；新训练写入`result/new_runs`。分类种子按下面顺序执行，以便严格比较各次训练的协议、代码和环境。

Classification environment / 分类环境：

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
首个种子命令加`--preflight-only`可只验证数据和初始化文件。

Segmentation environment / 分割环境：

```powershell
python code/run.py segmentation_2d/table6_fig6_7_training -- --data-root data/segmentation_2d/seg2D_clean_v2 --output-dir result/new_runs/seg2d_unet_seed123 --seed 123 --epochs 100 --batch-size 16 --lr 0.01 --device cuda --num-workers 8 --input-size 224 --lr-patience 10 --lr-factor 0.1
python code/run.py segmentation_3d/table7_fig8_9_12_training -- --data-root data/segmentation_3d/seg3D_clean --output-dir result/new_runs/seg3d_unet_seed42 --seed 42 --epochs 100 --batch-size 8 --lr 0.001 --device cuda --num-workers 8 --input-size 224 --depth 8 --lr-patience 10 --lr-factor 0.1
```

Summarize newly completed classification runs / 汇总新的分类运行：

```powershell
python code/run.py classification/table4_exp1_summary --runs-root result/new_runs --output-root result/new_summary
python code/run.py classification/table5_exp2_summary --runs-root result/new_runs --output-root result/new_summary
```


<a id="versions"></a>
## Versions and model archives / 版本与模型附件

**Current review source: [v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.1-review)** — complete public source snapshot at commit `7e7ce75a9770c7f59a93da8745da42e410e717ce`, including the saved-statistics verification entry points. Use this fixed version for the current review; later `main` documentation updates do not move the tag. See [release_manifest.json](../provenance/release_manifest.json).

**当前审阅源码：[v1.0.1-review](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.1-review)**，固定提交为 `7e7ce75a9770c7f59a93da8745da42e410e717ce`，包含完整公开源码及保存统计复核入口；后续 main 文档更新不改变该标签。私人审阅材料不包含在公开发布附件中。

Historical [v1.0.0 source](https://github.com/jiarundongjack/breastdm-reproduction/tree/v1.0.0) remains available for traceability; it does not contain the new verification entry points. 历史 v1.0.0 标签保留用于追溯，不作为当前审阅入口。

**Model assets: [v1.0.0-rc1](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1)** — retained solely as the historical model-weight prerelease. Its three model ZIPs and checksum file are unchanged. **Do not use the rc1 “Source code” archive as the current reproduction source.** Those automatic archives reflect the historical tagged tree.

**模型附件：v1.0.0-rc1**，保留为历史模型权重预发布版本，三份模型 ZIP 和校验文件不变。**不要用 rc1 的 “Source code” 压缩包作为当前复现源码**；它对应历史标签树。

To identify the fixed review commit in a Git clone, run `git rev-parse 'v1.0.1-review^{commit}'` after fetching the tag. It must resolve to `7e7ce75a9770c7f59a93da8745da42e410e717ce`. The release manifest records this literal fixed SHA; later documentation commits on main are not the tagged review source.

Git 克隆可用上述命令核对固定审阅提交。版本清单记录固定 SHA，main 后续文档提交不等于审阅标签对应的源码版本；下载标签源码时其中的历史措辞以新版发布说明为准。

### Download model weights / 下载模型权重

Download all three ZIP archives below. These links download the files directly; there is no need to visit the release page.

请下载以下全部三个模型压缩包，点击即可直接下载，无需进入发布页。

| Archive / 压缩包 | Download / 下载 |
|---|---|
| Pretrained initialization weights / 预训练初始化权重 | [pretrained-01.zip](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/breastdm-v1.0.0-rc1-pretrained-01.zip) |
| Trained checkpoints, archive 1 / 训练后权重包一 | [trained-models-01.zip](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/breastdm-v1.0.0-rc1-trained-models-01.zip) |
| Trained checkpoints, archive 2 / 训练后权重包二 | [trained-models-02.zip](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/breastdm-v1.0.0-rc1-trained-models-02.zip) |

[SHA-256 checksums / 校验文件](https://github.com/jiarundongjack/breastdm-reproduction/releases/download/v1.0.0-rc1/SHA256SUMS.txt)

After downloading, follow the [verification and restoration instructions](../weight%20download%20and%20restore/DOWNLOADS.md).

下载后，请按照[校验与恢复说明](../weight%20download%20and%20restore/DOWNLOADS.md)将权重恢复到程序所需的位置。

## Scientific-function organization / 按科学功能组织代码

Repository organization is based on scientific function rather than manuscript figure numbering. Final manuscript panel composition and publication layout are not part of the reproducibility codebase. This is a post-v1.0.0 navigation cleanup on main; the fixed v1.0.0 source tag and all scientific result records remain unchanged.

代码按科学功能组织，不以论文图号作为科学标识。最终论文的面板拼接与版面设计不属于复现代码库。本次仅整理发布后的 main 导航，不改变固定 v1.0.0 标签或科学结果。

The [file-level cleanup inventory](../provenance/figure_code_cleanup.json) maps old/new paths and explains decisions; [SHA-256 verification](../provenance/figure_code_sha256_before_after.json) records byte identity. Scripts mixing scientific computation and explicit manuscript composition remain untouched for manual review, including the Grad-CAM and over-segmentation figure tasks that still have historical figure-number folder names. They were not split or renamed. Historical source/result inventories retain their original paths and hashes; use this cleanup inventory as the path/disposition overlay for current main. Excluded private-source restoration paths are unchanged.

混合计算与排版的脚本原样保留待人工复核，因此部分入口仍保留历史图号名称；没有拆分或改写。历史来源/结果清单保留原始路径与哈希，当前路径和纯展示脚本移除情况以本轮清单补充说明。论文表格计算及训练任务未改名。

The standalone Exp-1 confusion-matrix renderer only styled an existing aggregate CSV and was removed; the computation of that CSV remains in `classification/exp1_mean_roc`, and its canonical saved PNG is unchanged. Scientific case/slice selection and Grad-CAM computation remain available; only isolated presentation-only scripts were removed.

### Formal saved-statistics verification / 正式保存统计复核

For the six formal classification runs, both flattened two-column AUC and conventional malignant-class AUROC are available through `classification_metrics_from_predictions.py`. For the formal 2D and 3D runs, `segmentation_metrics_from_predictions.py` reconstructs aggregate segmentation metrics from saved per-sample statistics. See [commands, input schemas and limitations](../code/analysis/SAVED_PREDICTION_VERIFICATION.md). These new entry points require saved predictions/statistics and official metrics CSVs; they perform no training or inference and are not included in the historical v1.0.0 source tag.

六次正式分类运行可通过新入口复核展平 AUC 并补充常规恶性类 AUROC；2D/3D 新入口从保存的逐样本统计重建汇总指标。详见[命令、输入格式及适用限制](../code/analysis/SAVED_PREDICTION_VERIFICATION.md)。需要本地预测/统计及正式指标 CSV，不运行训练或推理；这些新增入口不属于历史 v1.0.0 源码标签。
