# Local generation of case outputs / 本地生成病例输出

Initialization, preparation and CSV-only analysis use public scripts. Only workflows importing excluded model/training/evaluation implementations require the [selected source setup](../THIRD_PARTY_CODE.md); initialization does not require restoring all sources. / 初始化、整理及纯 CSV 分析使用公开脚本；只有依赖被排除实现的工作流才需要针对性源码准备。

Case-level CSVs, images, masks, predictions and case-linked metadata are not distributed in the current `main` tree when redistribution permission is unclear. Obtain inputs via the [original authors' repository](https://github.com/smallboy-code/Breast-cancer-dataset); cite the [original paper](https://doi.org/10.1016/j.compbiomed.2023.107255). Neither original nor preprocessed BreastDM data are provided here.

当前 `main` 不公开再分发许可不明确的病例级 CSV、影像、mask、预测与病例关联元数据，也不提供原始或预处理 BreastDM 数据。请从原作者入口自行获取数据并引用原论文。

## 1. Create a private workspace / 建立私有本地目录

Use a Git clone of the current `main`, with the classification/segmentation environments described in the root README. From the public checkout root:

使用最新 `main` 的 Git 克隆及根 README 指定的分类/分割环境。在仓库根目录执行：

```powershell
python code/local_generation.py init
```

This copies tracked working files into `.local/reproduction/`, without Git history, data or weights. It refuses an existing destination. All generation commands below run against that ignored private copy, so retained public aggregate results are not overwritten. Do not upload the private copy. `init` requires Git; users who downloaded a source ZIP should first clone the repository. It reads the working tree: use a clean checkout of the cleanup commit or later.

该命令把已跟踪文件复制到被忽略的 `.local/reproduction/`，不复制 Git 历史、数据或权重，并拒绝覆盖已有目录。下列命令只使用这个私有副本，不覆盖公开汇总结果。不要上传私有副本；下载 ZIP 的用户需先 Git 克隆仓库。

Prepare these directories **inside `.local/reproduction/data/`**, following the existing [classification](preparation/classification/README.md) and [segmentation](preparation/segmentation/README.md) instructions:

```text
classification_exp1/processed_dataset/{train,val,test}/...
classification_exp2/img17Se/{train,val,test}/...
segmentation_2d/seg2D_clean_v2/{train,val,test}/...
segmentation_3d/seg3D_clean/{train,val,test}/...
```

Those preparation scripts organize already split inputs; they do not implement every conversion from upstream raw downloads. Obtaining the required channel arrays and matching split layout remains a prerequisite. Do not invent replacement arrays, splits or case labels.

准备脚本整理已有划分的数据，不覆盖上游原始下载物到实验输入的全部转换。须先取得符合要求的通道数组及划分，不能用虚构数组、划分或标签替代。

For inference, use the existing model download/restoration instructions, running the unchanged restorer **from the private copy** after downloading the three listed archives yourself:

```powershell
python ".local/reproduction/weight download and restore/restore_assets.py" --asset-dir "D:/downloaded_model_assets" --verify-only
python ".local/reproduction/weight download and restore/restore_assets.py" --asset-dir "D:/downloaded_model_assets"
```

This restores into the private copy's `code/pretrained/` and `result/` folders. It does not download BreastDM data. No release, tag or asset is changed by these commands.

Before a model-dependent command, check and restore only its sources in the private copy:

```powershell
python .local/reproduction/code/setup_third_party.py --check --workflow classification-exp1
python .local/reproduction/code/setup_third_party.py --private-source "D:/private/breastdm-reproduction-source" --workflow classification-exp1
```

Choose `classification-exp2`, `seg2d`, `seg3d`, or a task name from `code/run.py --list` as appropriate. Manifest generation uses the existing evidence collector, which also hashes classification model/training code and pretrained weights; it retains those prerequisites. Pure plotting/statistics requires local CSVs and packages, not private model source. Figure 13 and threshold tasks use specific historical snapshots; the workflow check names those exact files. Full completion audits additionally require the original recorded hashes, data outputs and checkpoints; they are not CSV-only statistics.

先初始化，再按具体任务在私有副本恢复所需源码；不需要一次补齐全部 41 项。分类 manifest 沿用证据收集器，会核对源码和预训练权重，因此仍有这些前提。病例图与完整审计另需本地数据、权重和历史记录，不能用虚构记录替代。

## 2. Manifests and checkpoint inference / 清单和检查点推理

Run the classification commands with the classification environment. The wrapper calls the existing `collect_evidence` and `evaluate_test_set` functions, without executing training setup or training. Manifest checks retain the original counts, shapes and split-separation requirements and also require the original pretrained files. Classification inference uses the full trained checkpoint and the existing 32-sample test loader configuration.

分类命令调用原有清单函数和测试评估函数，不调用训练。清单仍验证原有样本数、形状、划分隔离及预训练文件。以下循环保留原有六个种子：

```powershell
foreach ($seed in 31415,27182,16180) {
  python .local/reproduction/code/local_generation.py manifest --experiment exp1 --seed $seed
  python .local/reproduction/code/local_generation.py predict --experiment exp1 --seed $seed --device cuda
}
foreach ($seed in 48271,59317,84629) {
  python .local/reproduction/code/local_generation.py manifest --experiment exp2 --seed $seed
  python .local/reproduction/code/local_generation.py predict --experiment exp2 --seed $seed --device cuda
}
```

Each private run receives `dataset_manifest.csv` and `test_predictions_two_class.csv`; its private metrics/ROC/PR files are generated by the unchanged evaluator. Existing prediction/manifest files are refused. No new completion manifest or historical checksum claim is created.

Use the segmentation environment for the following commands. They call the existing test dataset, transforms, model loading, `evaluate`, `save_test_predictions` and `save_representative_examples` functions. Input size remains 224, 3D depth remains 8, seeds remain 123/42, and test batch size remains 1.

```powershell
python .local/reproduction/code/local_generation.py predict --experiment seg2d --device cuda
python .local/reproduction/code/local_generation.py predict --experiment seg3d --device cuda
```

The private segmentation run receives `test_predictions.csv`, `test_metrics.csv`, `representative_success.png` and `representative_failure.png`. This is user-invoked checkpoint inference, not retraining or a replacement for the published experimental record. Numerical identity is not guaranteed across environments. The first-round cleanup itself did not run these commands on real data.

生成结果只保存在私有副本。本地推理不替换论文实验记录；跨环境不保证逐位一致。本轮仓库清理未在真实数据上运行上述推理。

## 3. Existing case analyses / 原有病例分析入口

After generating the required private prediction CSVs, run these from the public checkout root. All outputs remain inside `.local/`:

```powershell
python .local/reproduction/code/run.py classification/exp2_false_positive_cases
python .local/reproduction/code/run.py classification/channel_mapping
python .local/reproduction/code/run.py classification/fig10_11_gradcam
python .local/reproduction/code/run.py segmentation_3d/fig13_volume_selection
python .local/reproduction/code/run.py segmentation_3d/fig13_predictions
python .local/reproduction/code/run.py segmentation_3d/fig13_slice_selection
python .local/reproduction/code/run.py segmentation_3d/fig13_layout
python .local/reproduction/code/local_generation.py legacy-selection
python .local/reproduction/code/run.py segmentation_3d/table7_result_audit
```

| Removed output type / 移除类型 | Existing generation function/task / 本地入口 |
|---|---|
| Dataset manifests / 数据清单 | `local_generation.py manifest` → `exp1_provenance.collect_evidence` / `exp2_provenance.collect_evidence` |
| Classification predictions / 分类预测 | `local_generation.py predict --experiment exp1/exp2` → existing `evaluate_test_set` |
| 2D/3D predictions and representative examples / 分割预测与代表病例图 | `local_generation.py predict --experiment seg2d/seg3d` → existing evaluation/export functions |
| False-positive case tables / 假阳性病例表 | `classification/exp2_false_positive_cases` |
| Sample-to-path channel mapping / 通道对应路径表 | `classification/channel_mapping` |
| Input, Grad-CAM, channel figures and metadata / 输入、热图、通道图及元数据 | `classification/fig10_11_gradcam` |
| Slice stats, selected slices, volume rankings and example figures / 切片统计、筛选、体积排名及示例图 | `segmentation_3d/fig13_predictions`, then `fig13_slice_selection`, `fig13_layout`; `fig13_volume_selection` |
| Legacy top-20 candidates and selected top-3 table / 旧筛选表 | `local_generation.py legacy-selection` calls retained `find_overseg_case.py` with only its run path replaced, then `make_top3_overseg_figures.py` |
| Reconstructed per-volume counts and before/after hashes / 逐体积重建计数与路径哈希 | `segmentation_3d/table7_result_audit` |

Old duplicate figures and layout variants are represented by the current canonical figure tasks; the same output types are generated, not every superseded filename. Some figure scripts intentionally select the original study cases and assert their prediction characteristics. Keep those checks: mismatching inputs or predictions must fail rather than silently selecting different cases.

旧的重复图和排版变体由现有规范绘图入口生成同类型输出，不承诺恢复每个旧文件名。部分脚本固定选取研究中的案例并验证预测特征；保留验证，输入不匹配时应报错，不自动换病例。

The 3D audit also needs the retained history/configuration and both checkpoints. Classification full-audit tasks require the original completed-run hashes; locally regenerated records are not automatically valid historical records. Use the standalone CSV summarizer for local statistics instead of changing hashes or fabricating completion evidence:

```powershell
python code/analysis/summarize_predictions.py --experiment exp1 --runs-root .local/reproduction/result --output .local/statistics_exp1
python code/analysis/summarize_predictions.py --experiment exp2 --runs-root .local/reproduction/result --output .local/statistics_exp2
```

The `.gitignore` protects known output names and the whole `.local/` workspace. It is not a content classifier; renamed patient files outside those locations still require review before any commit.

`.gitignore` 保护已知文件名及整个 `.local/`。它不能判断内容；把病例文件改名或放到其他目录后，仍需提交前检查。
