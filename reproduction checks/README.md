# Release validation

Date: 2026-10-07. This report concerns the portable local release, not a new training experiment.

The dated checks did not repeat the eight complete 100-epoch training runs.

Current output locations and status are defined by the [manuscript-final map](../result/README.md#manuscript-final-results). The counts and execution statements below describe the dated historical checks. Round 3 removed two obsolete classification tables and 19 byte-identical generated copies; it did not rerun these checks or change the recorded outcomes.

## Changes

- Preserved all 46 task source files and their task-folder names. Added `code/run.py` and
  `code/release_paths.py` to stage a temporary import layout and resolve repository-relative paths.
- Copied the two initialization checkpoints into `code/pretrained`, verifying SHA-256.
- Copied 33 existing paper-output files to `result/paper_outputs`, including corrected Exp-2 ROC
  and cross-seed summaries. Newly regenerated outputs are under `result/reproduced`.
- Portable training defaults use the formal fusion model and first formal seed. Protocol checks,
  separate output directories and sequential cross-seed consistency checks remain enforced.
- The Fig-13 sample identity check compares the full relative sample path below `seg3D_clean`,
  retaining the split/patient/sequence check while permitting a changed drive/root.
- RGB checkpoint loading explicitly permits `argparse.Namespace` under `weights_only=True`.
- The subtraction audit caches the same arrays once instead of reopening them in every candidate loop.
  Formula enumeration and numeric comparisons remain unchanged. This resolved excessive repeated I/O.
- Thirteen core architecture, data-loader, loss, metric and training-loop modules were compared
  against the pre-edit AST: identical except the new path import.

## Executed checks

| Check | Result |
|---|---|
| Four training entry points, `--help` | Passed |
| Exp-1 seed31415 and Exp-2 seed48271 preflight | Passed, full prepared-data manifests and weight hashes |
| Classification model construction with both initialization weights | Both models loaded; actual 9/17-channel samples produced finite two-class outputs |
| 2D/3D formal checkpoint loading | Strict loading and finite synthetic forward outputs passed |
| Both three-seed summaries | Completed from retained formal runs |
| Both ROC and confusion-matrix pipelines | Completed; Exp-2 AUC 0.8735928017 ± 0.0218863413 |
| False-positive case tables, case trace and Grad-CAM | Completed |
| 3D protocol checks | Batch invariance, empty cases, missing-class behavior, depth normalization, scheduler and CSV writer passed |
| 3D formal result audit | DSC, mIoU and volume-mean PPV reconstructed |
| Threshold sensitivity | All 141 test volumes evaluated; 0.50 baseline within existing tolerance |
| Fig-13 predictions, volume/slice selection and layout | Completed |
| Channel mapping and subtraction candidates | Completed; execution success does not imply candidate subtraction formulas were confirmed |
| RGB intensity and prediction analysis | Completed; hard predictions unchanged for the audited volume |
| Source-slice and zero-padding checks | Completed |
| Separate 2D reference evaluator | Optional dependencies installed in isolated audit directory; synthetic identical masks returned precision/recall/F1=1 |
| Python syntax | All 48 release Python files passed |
| Historical result integrity | All 217 files matched their pre-release SHA-256 values |

The threshold=0.50 re-inference differed from retained aggregate metrics by at most
1.1910355e-6 in fraction units, within the existing baseline check's tolerance.

No full training rerun or installation on a separate physical computer was performed.
The checks above initially used the available local environments. An additional release test
installed both requirement sets in fresh Python 3.11.5 virtual environments with no access to
system site packages, restored all twelve ZIP assets into a separate directory, and ran sixteen
checks successfully: two dependency checks, two import/GPU-availability checks, four training
entry-point help checks, two classification preflights, two three-seed summaries, two ROC pipelines,
the 3D protocol checks and the 3D formal-result audit. All restored asset files passed SHA-256
verification. Exp-2 AUC remained 0.873593 ± 0.021886, and 3D volume-mean PPV remained 0.6041986659.

Machine-readable results and logs are in [logs/](logs/); complete fresh-install package versions
are in `../environment/*.lock.txt`. These describe release validation, not historical training.
The optional reference evaluator was tested separately as noted above; it was not part of the
two new base-environment installations.

Historical snapshots and log paths remain unchanged as evidence. They can differ from the portable
release in path handling and auxiliary fields, including the historical 3D global auxiliary PPV.

## Archival asset log

`logs/asset_restore_validation.log` records an earlier local validation, including archive names that are no longer current public assets. It is archival provenance, not a current download list. Only the three model ZIPs and SHA256SUMS.txt listed in the historical rc1 release are model downloads. The earlier source was v1.0.0; the current fixed review source is v1.0.1-review. See [Reproducibility & Verification](../REVIEW_NAVIGATION.md) for the current scope and prerequisites; this does not update or repeat the historical checks recorded here.
