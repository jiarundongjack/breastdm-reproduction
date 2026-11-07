# Provenance & Documentation

[Project overview](../README.md) · [Results](../result/README.md) · [Reproducibility & Verification](../REVIEW_NAVIGATION.md)

These records document source origins, formal-run mapping, hashes and historical organization decisions. Historical inventories retain the paths and meanings of their original snapshots. A later documentation update does not rewrite those records or move a fixed source tag.

## Primary records

| Record | Role |
|---|---|
| [release_manifest.json](release_manifest.json) | Fixed review commit, historical source tag and separate model-asset release |
| [manuscript_results.json](manuscript_results.json) | Canonical formal experiments and file-level output dispositions from the result-alignment work |
| [classification_protocol.json](classification_protocol.json) | Classification protocol record |
| [TRAINING_MODIFICATIONS.md](TRAINING_MODIFICATIONS.md) | Study-specific protocol facts and adaptations without republishing excluded training implementations |
| [source_inventory.json](source_inventory.json) | File-level source origins and redistribution classification |
| [workflow_sources.json](workflow_sources.json) | Source requirements for individual workflows |
| [round2b_review.json](round2b_review.json) | Source-recovery and provenance-review evidence |

## Historical path and hash evidence

The [figure-code cleanup inventory](figure_code_cleanup.json) and [before/after SHA-256 record](figure_code_sha256_before_after.json) describe a prior code-organization pass. Use them as path/disposition overlays on older inventories; do not interpret old paths as proof that excluded files are currently distributed. This documentation pass does not repeat those removals or authorize further deletion.

Formal run configurations, logs, completion records and retained `code_snapshot/` files are reachable through the [eight-run index](../REVIEW_NAVIGATION.md#eight-formal-runs). Snapshot presence and functional similarity alone do not establish byte identity with every source used at training time.

## Attribution and resource boundaries

- [THIRD_PARTY_CODE.md](../THIRD_PARTY_CODE.md): source setup, attribution and redistribution boundaries.
- [License records](licenses/) and [model-branch license guide](../weight%20download%20and%20restore/Classification%20model%20branch%20licenses/README.md): existing component-specific notices.
- [Model archive manifest](../weight%20download%20and%20restore/ASSETS.json) and [model-file manifest](../weight%20download%20and%20restore/FILES.json): archive and member checksums; not source or dataset inventories.
- [Historical release checks](../reproduction%20checks/README.md): dated, scoped checks rather than blanket proof of complete reproduction.
- [Verification levels and limitations](../REVIEW_NAVIGATION.md#verification-levels): distinction between statistics, model reevaluation and reconstruction from raw data.

The public documentation does not grant new redistribution rights for original data or unresolved source. Original resource links are in the [download guide](../weight%20download%20and%20restore/DOWNLOADS.md).
