# Release validation / 发布版验证

Date / 日期: 2026-10-07. This report concerns the portable local release, not a new training experiment.

本报告针对本地发布版运行整理，不代表重新完成了八次100轮训练。

## Changes / 修改范围

- Preserved all 46 task source files and their task-folder names. Added `code/run.py` and
  `code/release_paths.py` to stage a temporary import layout and resolve repository-relative paths.
  保留46个任务程序及其文件夹名，新增两个运行辅助程序。
- Copied the two initialization checkpoints into `code/pretrained`, verifying SHA-256.
  两份预训练权重复制到上述位置并校验一致。
- Copied 33 existing paper-output files to `result/paper_outputs`, including corrected Exp-2 ROC
  and cross-seed summaries. Newly regenerated outputs are under `result/reproduced`.
  补齐33个已有论文输出文件；新生成结果另外存放。
- Portable training defaults use the formal fusion model and first formal seed. Protocol checks,
  separate output directories and sequential cross-seed consistency checks remain enforced.
  分类默认模型和首个种子改为正式实验对应值，继续保留协议与多种子一致性检查。
- The Fig-13 sample identity check compares the full relative sample path below `seg3D_clean`,
  retaining the split/patient/sequence check while permitting a changed drive/root.
  图13路径核对保留集合、患者和序列身份比较，允许根目录迁移。
- RGB checkpoint loading explicitly permits `argparse.Namespace` under `weights_only=True`.
  RGB脚本修复新版PyTorch加载正式权重时的参数对象兼容问题。
- The subtraction audit caches the same arrays once instead of reopening them in every candidate loop.
  Formula enumeration and numeric comparisons remain unchanged. This resolved excessive repeated I/O.
  减影检查仅增加数据缓存，候选公式及比较逻辑不变。
- Thirteen core architecture, data-loader, loss, metric and training-loop modules were compared
  against the pre-edit AST: identical except the new path import.
  13个核心模块的语法树对比确认，除路径导入外未改变其实现。

## Executed checks / 实际执行

| Check / 检查 | Result / 结果 |
|---|---|
| Four training entry points, `--help` / 四个训练入口 | Passed / 通过 |
| Exp-1 seed31415 and Exp-2 seed48271 preflight / 两组分类首种子预检查 | Passed, full prepared-data manifests and weight hashes / 完整数据清单及权重校验通过 |
| Classification model construction with both initialization weights / 分类初始化 | Both models loaded; actual 9/17-channel samples produced finite two-class outputs / 两模型实际样本前向通过 |
| 2D/3D formal checkpoint loading / 分割正式权重 | Strict loading and finite synthetic forward outputs passed / 严格加载及合成输入前向通过 |
| Both three-seed summaries / 两组多种子汇总 | Completed from retained formal runs / 正式结果汇总完成 |
| Both ROC and confusion-matrix pipelines / 两组ROC与混淆矩阵 | Completed; Exp-2 AUC 0.8735928017 ± 0.0218863413 / 完成 |
| False-positive case tables, case trace and Grad-CAM / 假阳性表、来源及热图 | Completed / 完成 |
| 3D protocol checks / 3D协议检查 | Batch invariance, empty cases, missing-class behavior, depth normalization, scheduler and CSV writer passed / 通过 |
| 3D formal result audit / 3D正式结果复核 | DSC, mIoU and volume-mean PPV reconstructed / 三项指标重建一致 |
| Threshold sensitivity / 阈值分析 | All 141 test volumes evaluated; 0.50 baseline within existing tolerance / 全部141体积完成，基线通过既有容差 |
| Fig-13 predictions, volume/slice selection and layout / 图13相关程序 | Completed / 完成 |
| Channel mapping and subtraction candidates / 通道与减影检查 | Completed; execution success does not imply candidate subtraction formulas were confirmed / 运行完成不代表候选减影公式得到证实 |
| RGB intensity and prediction analysis / RGB分析 | Completed; hard predictions unchanged for the audited volume / 指定核查体积硬预测不变 |
| Source-slice and zero-padding checks / 源切片及零填充 | Completed / 完成 |
| Separate 2D reference evaluator / 独立2D参考评估器 | Optional dependencies installed in isolated audit directory; synthetic identical masks returned precision/recall/F1=1 / 独立依赖环境中合成掩膜验证通过 |
| Python syntax / Python语法 | All 48 release Python files passed / 全部通过 |
| Historical result integrity / 历史结果完整性 | All 217 files matched their pre-release SHA-256 values / 全部保持原样 |

The threshold=0.50 re-inference differed from retained aggregate metrics by at most
1.1910355e-6 in fraction units, within the existing baseline check's tolerance.
阈值0.50重新推理与历史汇总指标最大差异为1.1910355e-6（比例单位），处于原脚本允许范围内。

No full training rerun or installation on a separate physical computer was performed.
The checks above initially used the available local environments. An additional release test
installed both requirement sets in fresh Python 3.11.5 virtual environments with no access to
system site packages, restored all twelve ZIP assets into a separate directory, and ran sixteen
checks successfully: two dependency checks, two import/GPU-availability checks, four training
entry-point help checks, two classification preflights, two three-seed summaries, two ROC pipelines,
the 3D protocol checks and the 3D formal-result audit. All restored asset files passed SHA-256
verification. Exp-2 AUC remained 0.873593 ± 0.021886, and 3D volume-mean PPV remained 0.6041986659.
未重跑完整训练，也未换另一台物理电脑测试。除此前现有环境检查外，额外在不继承系统包的两套全新
Python 3.11.5虚拟环境中安装依赖，并将12个附件恢复到独立目录；上述16项检查全部通过，所有附件文件
的SHA-256核对通过。实验二AUC和3D逐体积PPV保持一致。

Machine-readable results and logs are in `validation/`; complete fresh-install package versions
are in `environment/*.lock.txt`. These describe release validation, not historical training.
检查记录见`validation/`，本次新安装环境完整版本见`environment/*.lock.txt`，均不冒充历史训练记录。
The optional reference evaluator was tested separately as noted above; it was not part of the
two new base-environment installations.
独立参考评估器按上表单独验证，不属于这两套新基础环境安装范围。

Historical snapshots and log paths remain unchanged as evidence. They can differ from the portable
release in path handling and auxiliary fields, including the historical 3D global auxiliary PPV.
历史快照和日志保留原样；其路径处理和辅助输出字段可能与发布版不同，包括历史3D全局辅助PPV。
