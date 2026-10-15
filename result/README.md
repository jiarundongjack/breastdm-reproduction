# Experimental results / 实验结果

This directory contains records from eight experimental runs and paper analysis outputs. Model weights are supplied separately as Release assets.

本目录保存八次实验的运行记录及论文分析输出，模型权重通过Release附件单独提供。

The current public tree retains aggregate metrics, statistical figures and non-patient-level run records. Case-level outputs with unclear redistribution permission have been removed. Use [local generation](../code/LOCAL_GENERATION.md) to generate them privately; historical hashes are references, not a claim that regenerated files match the original run.

当前公开目录保留汇总指标、统计图和非病例级运行记录。许可不明确的病例输出已移除；请按[本地生成说明](../code/LOCAL_GENERATION.md)私下生成。历史哈希仅作原始记录参考，不表示重新生成文件与历史运行逐位一致。

## Manuscript-final results

This section is the authoritative map for the manuscript-final experiment set specified for this cleanup. The machine-readable [manifest and complete output audit](../provenance/manuscript_results.json) records each baseline result file, its category, hash and disposition. This is an alignment of existing records, not a new experiment or a claim to have checked the manuscript Word/PDF.

本节为本轮指定的论文正式实验集合与现有结果文件的对应表；[机器可读清单](../provenance/manuscript_results.json)逐文件记录分类、原哈希及保留/移除决定。本轮只整理已有记录，没有重算结果，也没有修改或核验论文 Word/PDF。

| Experiment / 实验 | Formal seeds / 正式种子 | Reporting / 报告方式 | Run directories / 正式目录 |
|---|---|---|---|
| Classification Exp-1 | 31415 / 27182 / 16180 | 3 independent runs; mean ± SD (sample SD, ddof=1) | [fusion_exp1_100ep_seed31415_final_multiseed](fusion_exp1_100ep_seed31415_final_multiseed/); [fusion_exp1_100ep_seed27182_final_multiseed](fusion_exp1_100ep_seed27182_final_multiseed/); [fusion_exp1_100ep_seed16180_final_multiseed](fusion_exp1_100ep_seed16180_final_multiseed/) |
| Classification Exp-2 | 48271 / 59317 / 84629 | 3 independent runs; mean ± SD (sample SD, ddof=1) | [fusion_exp2_100ep_seed48271_final_multiseed](fusion_exp2_100ep_seed48271_final_multiseed/); [fusion_exp2_100ep_seed59317_final_multiseed](fusion_exp2_100ep_seed59317_final_multiseed/); [fusion_exp2_100ep_seed84629_final_multiseed](fusion_exp2_100ep_seed84629_final_multiseed/) |
| 2D U-Net | 123 | Single run / 单次正式运行 | [seg2d_unet_seed123_gpu_20260911](seg2d_unet_seed123_gpu_20260911/) |
| 3D U-Net | 42 | Single run / 单次正式运行 | [seg3d_unet_locked_volume_ppv_seed42_20260911](seg3d_unet_locked_volume_ppv_seed42_20260911/) |

Both classification experiments use author-released-code-compatible **flattened two-class AUC**. Exp-2 seed **59317** remains the formal run for persistent false-positive/Grad-CAM analysis. Segmentation seeds 123 and 42 belong to separate experiment namespaces and are not obsolete classification runs.

两组分类均采用与作者公开代码兼容的两列概率/独热标签展平 AUC。Exp-2 的 59317 是正式病例分析所用 seed；分割的 123/42 是正式单次运行，不能与旧分类 seeds 混淆。

Files outside this canonical set may represent intermediate, diagnostic, historical, or reproduction-evidence outputs and must not be interpreted as manuscript-final results unless explicitly identified here.

未在本节明确列为正式结果的文件，可能属于中间分析、诊断、历史输出或复现证据，不应自动视为论文最终结果。

### Authoritative aggregate and figure locations / 正式汇总与图表位置

| Location / 位置 | Role / 用途 |
|---|---|
| [exp1_final_multiseed_summary](paper_outputs/exp1_final_multiseed_summary/) | Exp-1 formal per-seed table, mean/SD and original-paper comparison / 正式逐种子指标、均值标准差与原论文对照 |
| [exp1_figures](paper_outputs/exp1_figures/) | Formal seeds 31415/27182/16180: AUC table, confusion counts, mean ROC/CM and report / 正式三种子 ROC、混淆矩阵及说明 |
| [exp2_final_multiseed_summary](paper_outputs/exp2_final_multiseed_summary/) | Exp-2 formal per-seed table, mean/SD and original-paper comparison / 正式逐种子汇总与对照 |
| [exp2_corrected_flattened_roc_20261007](paper_outputs/exp2_corrected_flattened_roc_20261007/) | Formal 48271/59317/84629 flattened ROC and associated tables/CM / 正式两类展平 ROC 及关联结果 |
| [exp2_confusion](paper_outputs/exp2_confusion/) | Exact mean and rounded display confusion matrices; rounding note / 精确均值矩阵、整数展示矩阵及取整说明 |
| [2D test metrics](seg2d_unet_seed123_gpu_20260911/test_metrics.csv) | Formal single-run segmentation metrics; distinguish evaluator-specific PPV columns / 正式单次分割指标，各 PPV 列定义不能混用 |
| [3D test metrics](seg3d_unet_locked_volume_ppv_seed42_20260911/test_metrics.csv) | Formal single-run metrics; volume-mean PPV is distinct from auxiliary global PPV / 正式指标，体积平均 PPV 与辅助全局 PPV 不同 |

The eight runs' `author_style_metrics.csv` / `test_metrics.csv` and saved `*_curve.png` support the formal experiment set. Their configs, training histories/logs, completion/hash records and retained source snapshots are reproduction evidence. All of these files were preserved byte-for-byte.

Historical `Fig2`/`Fig3`, `fig2_3_*`, `fig4_5_*` and `exp1_table5_comparison.csv` names are retained: numbering is not a different experiment, and “Original Table 5” refers to the original-paper comparison. No file was removed solely because its figure/table number differs from the manuscript layout.

正式八次运行的指标和曲线保持原字节；配置、日志、完成记录、哈希和源码快照作为复现证据保留。历史图表编号不表示旧实验；`Original Table 5` 指原论文对照，不据此删改正式数值。

### Reproduction evidence / 复现与差异分析证据

- `reproduced/audit_results.json`, `evidence_chain.csv`, `recomputed_metric_check.csv`, `paper_vs_reproduction.csv`: saved 3D checkpoint/metric/discrepancy verification, not recomputed in this cleanup.
- `reproduced/test_metrics.csv` and `test_report.json`: four-volume **synthetic protocol tests**, not formal BreastDM segmentation results.
- Both retained `threshold_sensitivity_volume_ppv.csv` files: threshold analysis, not a replacement for formal argmax metrics; their small differences from formal metrics must not be silently substituted.
- `reproduced/fp_fn_cases/subtraction_formula_*.csv`: aggregate subtraction/channel-formula checks, not an extra classification experiment.
- Formal configs, logs, hashes, completion records, snapshots and case-selection scripts remain unchanged. Earlier validation logs under `reproduction checks/` are historical evidence, not a list of every currently published file.

上述内容用于复现验证和差异分析，不能作为额外正式实验或替代正式指标。合成测试、阈值扫描与正式评价尤其需要区分。

### Legacy cleanup and unresolved variants / 旧输出清理与待确认版本

The old Exp-1 `exp1_three_seed_results.csv` (42/123/2026) and its companion `exp1_mean_std.csv` were superseded by the formal three-seed summaries and removed. The other six files in `exp1_figures/` belong to the formal seeds and remain. Nineteen byte-identical generated copies under `reproduced/` were removed in favor of their authoritative `paper_outputs/` counterparts; exact replacements and original hashes are recorded in the manifest. The formal paper outputs were not rewritten.

旧 Exp-1 的 42/123/2026 两份表已移除，同目录其余六份正式输出保留。另移除 `reproduced/` 下 19 个与正式输出逐字节相同的副本，清单中逐项指向保留版本；未改写正式图表。

These three different-byte PNG variants remain **AMBIGUOUS_DO_NOT_TOUCH**, not designated manuscript-final. Their precise rendering/selection relationship is not established, so they were not deleted:

- `reproduced/exp1_mean_confusion_matrix.png`
- `reproduced/exp1_mean_roc.png`
- `reproduced/exp2_mean_confusion_matrix_integer.png`

这三份 PNG 与正式版本字节不同，尚不能确定其排版/选择关系，故保留待确认，不列为正式图表。后续用户生成结果也不自动成为正式结果；以本节及清单为准。
