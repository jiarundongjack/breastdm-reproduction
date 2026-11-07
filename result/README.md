# Experimental results

This directory contains records from eight experimental runs and paper analysis outputs. Model weights are supplied separately as Release assets.

The current public tree retains aggregate metrics, statistical figures and non-patient-level run records. Case-level outputs with unclear redistribution permission have been removed. Use [local generation](../code/LOCAL_GENERATION.md) to generate them privately; historical hashes are references, not a claim that regenerated files match the original run.

## Manuscript-final results

This section is the authoritative map for the manuscript-final experiment set specified for this cleanup. The machine-readable [manifest and complete output audit](../provenance/manuscript_results.json) records each baseline result file, its category, hash and disposition. This is an alignment of existing records, not a new experiment or a claim to have checked the manuscript Word/PDF.

| Experiment | Formal seeds | Reporting | Run directories |
|---|---|---|---|
| Classification Exp-1 | 31415 / 27182 / 16180 | 3 independent runs; mean ± SD (sample SD, ddof=1) | [fusion_exp1_100ep_seed31415_final_multiseed](fusion_exp1_100ep_seed31415_final_multiseed/); [fusion_exp1_100ep_seed27182_final_multiseed](fusion_exp1_100ep_seed27182_final_multiseed/); [fusion_exp1_100ep_seed16180_final_multiseed](fusion_exp1_100ep_seed16180_final_multiseed/) |
| Classification Exp-2 | 48271 / 59317 / 84629 | 3 independent runs; mean ± SD (sample SD, ddof=1) | [fusion_exp2_100ep_seed48271_final_multiseed](fusion_exp2_100ep_seed48271_final_multiseed/); [fusion_exp2_100ep_seed59317_final_multiseed](fusion_exp2_100ep_seed59317_final_multiseed/); [fusion_exp2_100ep_seed84629_final_multiseed](fusion_exp2_100ep_seed84629_final_multiseed/) |
| 2D U-Net | 123 | Single run | [seg2d_unet_seed123_gpu_20260911](seg2d_unet_seed123_gpu_20260911/) |
| 3D U-Net | 42 | Single run | [seg3d_unet_locked_volume_ppv_seed42_20260911](seg3d_unet_locked_volume_ppv_seed42_20260911/) |

Both classification experiments use author-released-code-compatible **flattened two-class AUC**. Exp-2 seed **59317** remains the formal run for persistent false-positive/Grad-CAM analysis. Segmentation seeds 123 and 42 belong to separate experiment namespaces and are not obsolete classification runs.

Files outside this canonical set may represent intermediate, diagnostic, historical, or reproduction-evidence outputs and must not be interpreted as manuscript-final results unless explicitly identified here.

### Authoritative aggregate and figure locations

| Location | Role |
|---|---|
| [exp1_final_multiseed_summary](paper_outputs/exp1_final_multiseed_summary/) | Exp-1 formal per-seed table, mean/SD and original-paper comparison |
| [exp1_figures](paper_outputs/exp1_figures/) | Formal seeds 31415/27182/16180: AUC table, confusion counts, mean ROC/CM and report |
| [exp2_final_multiseed_summary](paper_outputs/exp2_final_multiseed_summary/) | Exp-2 formal per-seed table, mean/SD and original-paper comparison |
| [exp2_corrected_flattened_roc_20261007](paper_outputs/exp2_corrected_flattened_roc_20261007/) | Formal 48271/59317/84629 flattened ROC and associated tables/CM |
| [exp2_confusion](paper_outputs/exp2_confusion/) | Exact mean and rounded display confusion matrices; rounding note |
| [2D test metrics](seg2d_unet_seed123_gpu_20260911/test_metrics.csv) | Formal single-run segmentation metrics; distinguish evaluator-specific PPV columns |
| [3D test metrics](seg3d_unet_locked_volume_ppv_seed42_20260911/test_metrics.csv) | Formal single-run metrics; volume-mean PPV is distinct from auxiliary global PPV |

The eight runs' `author_style_metrics.csv` / `test_metrics.csv` and saved `*_curve.png` support the formal experiment set. Their configs, training histories/logs, completion/hash records and retained source snapshots are reproduction evidence. All of these files were preserved byte-for-byte.

Historical `Fig2`/`Fig3`, `fig2_3_*`, `fig4_5_*` and `exp1_table5_comparison.csv` names are retained: numbering is not a different experiment, and `Original Table 5` refers to the original-paper comparison. No file was removed solely because its figure/table number differs from the manuscript layout.

### Reproduction evidence

- `reproduced/audit_results.json`, `evidence_chain.csv`, `recomputed_metric_check.csv`, `paper_vs_reproduction.csv`: saved 3D checkpoint/metric/discrepancy verification, not recomputed in this cleanup.
- `reproduced/test_metrics.csv` and `test_report.json`: four-volume **synthetic protocol tests**, not formal BreastDM segmentation results.
- Both retained `threshold_sensitivity_volume_ppv.csv` files: threshold analysis, not a replacement for formal argmax metrics; their small differences from formal metrics must not be silently substituted.
- `reproduced/fp_fn_cases/subtraction_formula_*.csv`: aggregate subtraction/channel-formula checks, not an extra classification experiment.
- Formal configs, logs, hashes, completion records, snapshots and case-selection scripts remain unchanged. Earlier validation logs under `reproduction checks/` are historical evidence, not a list of every currently published file.

### Legacy cleanup and resolved variants

The old Exp-1 `exp1_three_seed_results.csv` (42/123/2026) and its companion `exp1_mean_std.csv` were superseded by the formal three-seed summaries and removed. The other six files in `exp1_figures/` belong to the formal seeds and remain. Nineteen byte-identical generated copies under `reproduced/` were removed in favor of their authoritative `paper_outputs/` counterparts; exact replacements and original hashes are recorded in the manifest. The formal paper outputs were not rewritten.

Round 4 resolved the three previously ambiguous PNG variants as **DUPLICATE_VARIANT** and removed only the `reproduced/` copies. All decoded RGBA pixels, image dimensions and DPI match the canonical figures exactly. PNG chunks differ only in the Software text metadata (Matplotlib 3.11.0 versus 3.11.1); there is no crop, layout or numerical difference.

| Removed variant | Canonical replacement |
|---|---|
| `reproduced/exp1_mean_confusion_matrix.png` | [Formal Exp-1 CM](paper_outputs/exp1_figures/exp1_mean_confusion_matrix.png) |
| `reproduced/exp1_mean_roc.png` | [Formal Exp-1 ROC](paper_outputs/exp1_figures/exp1_mean_roc.png) |
| `reproduced/exp2_mean_confusion_matrix_integer.png` | [Formal Exp-2 display CM](paper_outputs/exp2_confusion/exp2_mean_confusion_matrix_integer.png) |

No unresolved image variants remained after Round 4. Before/after hashes, dimensions, DPI, metadata and replacement paths are recorded under `round4_image_identity` in the manifest. No figures were redrawn, metrics recalculated or formal results changed.

This result map originated with the historical `v1.0.0` source alignment. The current fixed review source is `v1.0.1-review`; see [release_manifest.json](../provenance/release_manifest.json). Later documentation updates and local outputs do not automatically enter that fixed snapshot or become manuscript-final.

## Post-release code organization

Scientific result files, values and canonical experiment mapping are unchanged. Pure presentation-only Python helpers were removed; mixed case-computation/layout scripts remain unchanged for manual review. Historical result-manifest entries describe the original baseline; current code paths/dispositions are overlaid by [figure_code_cleanup.json](../provenance/figure_code_cleanup.json). Repository tasks describe scientific function rather than final manuscript panel composition.

## Reproducibility & Verification

See [verification guide](../REVIEW_NAVIGATION.md) for the eight formal runs, fixed source commit, private-input requirements and historical weight release. Historical v1.0.0 descriptions retain their original meaning; the current fixed review is v1.0.1-review.
