# Mixed-origin training and project protocol modifications

The canonical Exp-1/Exp-2 trainers and six historical classification training snapshots are substantially modified original-author implementations. Their complete bodies remain excluded. The four canonical/historical segmentation trainers also remain excluded. This record describes project changes without copying upstream implementation bodies or claiming a redistribution license.

## Structural comparison

Pinned original classification source: [BreastDM Classification task/train.py](https://github.com/smallboy-code/Breast-cancer-dataset/blob/592561c9fc617a345fd6e881eb326fcc521263a2/Classification%20task/train.py).

| Function | Original lines | Canonical Exp-1 lines | Canonical Exp-2 lines | Finding |
|---|---|---|---|---|
| split_data | 69–96 | 94–121 | 98–125 | Exact AST/text match |
| save_dict | 117–122 | 257–262 | 261–266 | Exact AST/text match |
| plot_confusion_matrix | 128–162 | 268–302 | 272–306 | Exact AST/text match |
| train / val | See per-function evidence | See evidence | See evidence | Modified loops with inherited training structure |
| save_training_history / find_best_threshold / evaluate_test_set | No corresponding named function in compared original trainer | Present | Present | Project-specific additions within a mixed-origin complete file |

These findings do not assign an independent license to the mixed file. The [per-file structural review](round2b_review.json) records canonical and historical comparisons, including function ranges and similarities. Segmentation trainers retain upstream-related model-construction, argument, transform and training structure; their project additions do not convert the entire implementation into project-created source.

## Preserved reproduction-specific changes

The public [configuration record](classification_protocol.json) extracts literal facts from the unchanged `exp1_provenance.py` and `exp2_provenance.py`. It is documentation, not a new training implementation. The public modules preserve protocol validation, split/count checks, file fingerprints, completion checks and multi-seed verification. Runtime paths, weights and data are obtained locally, not embedded in this configuration.

- Exp-1 seeds: 31415, 27182, 16180; Exp-2 seeds: 48271, 59317, 84629.
- Recorded 100-epoch, batch-32 SGD protocol and learning-rate schedule; original loss, input preparation and augmentation are unchanged.
- Strict validation-AUC checkpoint selection, earlier checkpoint retained on ties, test excluded from selection.
- Flattened two-column probabilities and one-hot labels for AUC; argmax hard classification; explicitly recorded precision conventions.
- Study-specific split separation, dataset shape checks, history export, run isolation and completed-run evidence.
- The existing `local_generation.py` wrapper calls the restored private trainer's existing evaluator; it does not replace its algorithm. Missing private trainer/model source is reported before running.

The restored classification metric module is a small project-specific NumPy/scikit-learn wrapper. It validates probabilities and labels, derives confusion counts and calls library ROC/AUC/precision functions; no complete matching original-author implementation or substantial external block was identified in the audited corpus. Canonical and snapshot versions retain their baseline bytes. The 3D metric wrapper is likewise public project-specific protocol code, but its imported generic Dice/distributed/training helpers remain separately excluded.

## Scope and limitations

No training body is republished as a patch, and no replacement model or evaluator is introduced. A user with only the public repository and the original-author download cannot recover every exact adapted trainer: a matching privately held adaptation remains necessary. All setup hashes and exclusions are in [source_inventory.json](source_inventory.json). Model files keep their existing third-party or unresolved classification; licensed ViT, 2D U-Net and EarlyStopping notices remain unchanged.

本记录只保存本项目的协议配置和修改说明，不重新公开原作者混合训练实现。配置不参与执行，不改变模型、训练、评价或随机种子；完整复现仍需规定的本地输入及对应私有改编源码。
