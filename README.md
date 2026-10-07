# BreastDM reproduction / BreastDM复现

**Local candidate `v1.0.0-rc1`; not yet published.** Download and restore the matching
Release assets before running any experiment: see [DOWNLOADS.md](DOWNLOADS.md).
Redistribution permissions are still being checked; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
**本地候选版，尚未发布。** 数据和模型保存在配套附件中，先按下载说明恢复目录，再运行实验。
公开再分发许可仍在核实，详见第三方来源说明。

Prepared datasets, training and analysis code, and experimental results for
**Reproducibility and Discrepancy Analysis of Classification and Segmentation Benchmarks in BreastDM**.

本目录提供该论文的预处理数据、训练与分析代码，以及八次正式实验结果。
The reproduction starts from the supplied prepared datasets, not raw DICOM conversion.
复现起点是已提供的预处理数据，不包含从原始DICOM重新构建样本的流程。

## Contents / 目录

| Path / 路径 | Purpose / 用途 |
|---|---|
| `data/classification_exp1/processed_dataset` | Exp-1, nine channels / 实验一9通道输入 |
| `data/classification_exp2/img17Se` | Exp-2, seventeen channels / 实验二17通道输入 |
| `data/segmentation_2d/seg2D_clean_v2` | 2D images and masks / 2D图像及标签 |
| `data/segmentation_3d/seg3D_clean` | 3D volumes and masks / 3D体积及标签 |
| `code/classification`, `code/segmentation_2d`, `code/segmentation_3d` | 46 curated source files in paper-task folders / 按论文任务保留的46个程序 |
| `code/pretrained` | Two classification initialization checkpoints / 分类训练使用的两份预训练权重 |
| `result/fusion_*`, `result/seg*` | Eight unchanged formal runs / 八次未经改写的正式运行记录 |
| `result/paper_outputs` | Cross-seed summaries, corrected Exp-2 ROC, Grad-CAM and confusion matrices / 多种子汇总、修正后的实验二ROC、Grad-CAM及混淆矩阵 |
| `result/reproduced` | New analysis outputs, separate from historical evidence / 本次重新分析输出，与历史记录分开 |
| `environment` | Observed dependencies for two environments / 两套环境的依赖记录 |

Use **`result`**, singular. Historical configuration files retain their original absolute
paths and auxiliary metric fields as evidence; they are not executable configuration for this release.
统一使用单数`result`。历史配置中的绝对路径和辅助指标字段作为原始凭据保留，不作为本发布版运行配置。

## Environments / 环境

The `classification.lock.txt` and `segmentation.lock.txt` files record the complete
package versions resolved in fresh Python 3.11.5 environments during release preparation.
They are release-validation records, not retrospective records of the original training installations.
两份`.lock.txt`记录发布检查中新建Python 3.11.5环境的完整依赖，不追溯替代原训练环境记录。
For the pinned release environment, substitute the corresponding `.lock.txt` for `.txt`
in the installation commands below.
需要固定完整依赖时，可将下面安装命令中的对应`.txt`替换为`.lock.txt`。

The reported formal environments used Python 3.11.5, Windows 11, an NVIDIA RTX PRO 3000
Blackwell Generation Laptop GPU (12 GB), classification PyTorch 2.10.0+cu130,
and segmentation PyTorch 2.7.1+cu128. Use separate environments for these tasks.
论文记录的正式环境为Python 3.11.5、Windows 11、12 GB显存的上述GPU；分类和分割使用两套不同的PyTorch环境。

Example installation with Python 3.11 / 使用Python 3.11的安装示例：

```powershell
python -m venv .venv_classification
.\.venv_classification\Scripts\python.exe -m pip install -r environment/classification.txt
python -m venv .venv_segmentation
.\.venv_segmentation\Scripts\python.exe -m pip install -r environment/segmentation.txt
```

Dependency files record locally observed versions; they have not been tested through a fresh
network installation. They do not prove every ancillary package had the same version during historical training.
依赖文件来自本机环境实测记录；尚未在全新联网环境中重新安装验证，也不据此声称历史训练时所有辅助包版本均完全相同。
Use the relevant interpreter for every command below; `python` denotes that interpreter.
以下`python`均指相应任务环境中的解释器。

## Entry point / 统一入口

Run commands from this directory / 在本目录运行：

```powershell
python code/run.py --list
python code/run.py classification/exp1_training -- --help
python code/run.py segmentation_3d/table7_fig8_9_12_training -- --help
```

Do not execute files inside task folders directly. `code/run.py` copies the selected task's
modules into a temporary conventional package layout, launches them with the same interpreter,
and removes the temporary directory on completion. Canonical source files remain in their task folders.
所有任务通过`code/run.py`运行；它临时组织导入结构，执行后清理临时目录，不改变任务文件夹中的源码。
Repository locations are resolved from the launcher, so no personal drive letters are required.
路径由入口文件定位，无需修改个人电脑盘符。Additional options follow `--` / 任务参数放在`--`之后。

## Reproduce tables and figures / 复现表格和图片

Classification environment / 分类环境：

```powershell
python code/run.py classification/table4_exp1_summary
python code/run.py classification/table5_exp2_summary
python code/run.py classification/fig2_3_exp1_roc
python code/run.py classification/fig2_confusion_matrix
python code/run.py classification/fig4_5_exp2_roc
python code/run.py classification/fig4_confusion_matrix
python code/run.py classification/exp2_false_positive_cases
python code/run.py classification/fig10_11_case_source
python code/run.py classification/fig10_11_gradcam
```

Exp-2 ROC uses both probability columns and flattened one-hot labels, matching Table 5:
**0.8735928017 ± 0.0218863413**, sample SD (`ddof=1`). The obsolete single-column ROC is not included
as the final Exp-2 figure in `paper_outputs`.
实验二ROC使用双列概率与展开的独热标签，和表5一致；标准差采用样本标准差，旧单列ROC不作为最终图片收录。

Segmentation environment / 分割环境：

```powershell
python code/run.py segmentation_3d/protocol_checks
python code/run.py segmentation_3d/table7_result_audit
python code/run.py segmentation_3d/threshold_analysis
python code/run.py segmentation_3d/fig13_volume_selection
python code/run.py segmentation_3d/fig13_slice_selection
python code/run.py segmentation_3d/fig13_layout
```

The slice-selection task also executes the retained top-three prediction script. Layout generation
prefers the newly generated volume-55 PNG and falls back to the retained formal PNG.
切片选择任务会调用三例预测脚本；布局任务优先使用新生成的第55体积PNG，没有新输出时才读取正式结果中的PNG。
3D PPV is an equally weighted mean over all 141 test volumes; empty predictions contribute zero,
and no volume is excluded for TP=0 or TN=0. Formal PPV is **0.604198665930658**.
3D PPV对全部141个测试体积等权平均，空预测记零，不因TP或TN为零筛选体积。

New outputs go to `result/reproduced`. For a second execution, choose another output root
if a task refuses existing outputs / 重复运行时，如任务拒绝覆盖，可另选输出目录：

```powershell
python code/run.py classification/fig4_5_exp2_roc --output-root result/reproduced_again
```

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

## Additional method analyses / 其他方法分析

`classification/channel_mapping`, `classification/subtraction_check`,
`segmentation_3d/source_slice_matching`, `segmentation_3d/zero_padding_check`,
`segmentation_3d/rgb_intensity_check`, and `segmentation_3d/rgb_prediction_impact`
are available through the same entry point. The 2D `ppv_reference` task evaluates separately
exported predicted and ground-truth masks; it is an author-style reference evaluator,
not the main training entry. Use its `--help` for mask-directory options.
以上方法分析也使用统一入口；2D的`ppv_reference`需要单独提供预测和真值掩膜，用于作者式独立评估器对照，不是训练入口。
Install its optional dependencies with `python -m pip install -r environment/reference_evaluator.txt`.
该独立参考评估器的额外依赖用上述命令安装；主训练程序不需要这些额外依赖。
The source-slice audit searches only the published `code` and `data` directories.
源切片审计只检查发布目录内的`code`和`data`；不搜索其他个人目录。

The temporary runtime is an execution aid, not an additional experiment. Numerical definitions,
training losses, architecture, augmentation and checkpoint-selection criteria are retained.
临时运行结构仅负责导入和路径衔接，不构成新增实验；保留已有指标、损失、架构、增强和选模规则。
See `VALIDATION.md` for exactly what was executed and its limitations / 实际验证范围见`VALIDATION.md`。
