"""Recompute classification statistics from saved predictions, without weights."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

SEEDS = {"exp1": (31415, 27182, 16180), "exp2": (48271, 59317, 84629)}
METRICS = ("Accuracy", "Sensitivity", "Specificity", "Precision_Malignant", "Precision_Weighted", "AUC")


def rank_auc(labels, scores):
    """Mann-Whitney AUC, awarding half credit to tied scores."""
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    preceding_negatives = 0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        group_positive = sum(label for _, label in pairs[i:j])
        group_negative = j - i - group_positive
        wins += group_positive * (preceding_negatives + group_negative / 2)
        preceding_negatives += group_negative
        i = j
    return wins / (positives * negatives)


def compute(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"True Label", "Predicted Label", "Benign Probability", "Malignant Probability"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing required columns: {path}")
    tp = tn = fp = fn = 0
    labels, scores = [], []
    for line, row in enumerate(reader, 2):
        try:
            true, pred = int(row["True Label"]), int(row["Predicted Label"])
            benign, malignant = float(row["Benign Probability"]), float(row["Malignant Probability"])
            if true not in (0, 1) or pred not in (0, 1):
                raise ValueError("labels must be 0 or 1")
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in (benign, malignant)):
                raise ValueError("invalid probability")
            if abs(benign + malignant - 1) > 1e-6:
                raise ValueError("probabilities must sum to one")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}, line {line}: {exc}") from exc
        tp += true == 1 and pred == 1
        tn += true == 0 and pred == 0
        fp += true == 0 and pred == 1
        fn += true == 1 and pred == 0
        labels.extend((1 - true, true))
        scores.extend((benign, malignant))
    n = tp + tn + fp + fn
    if not n:
        raise ValueError(f"No prediction rows: {path}")
    def ratio(a, b):
        return a / b if b else float("nan")
    weighted = ((tp + fn) * (tp / (tp + fp) if tp + fp else 0)
                + (tn + fp) * (tn / (tn + fn) if tn + fn else 0)) / n
    return {
        "Accuracy": (tp + tn) / n, "Sensitivity": ratio(tp, tp + fn),
        "Specificity": ratio(tn, tn + fp), "Precision_Malignant": ratio(tp, tp + fp),
        "Precision_Weighted": weighted, "AUC": rank_auc(labels, scores),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "N": n,
    }, hashlib.sha256(raw).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=SEEDS, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parents[2] / "result")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    rows, inputs = [], []
    for seed in SEEDS[args.experiment]:
        path = args.runs_root / f"fusion_{args.experiment}_100ep_seed{seed}_final_multiseed" / "test_predictions_two_class.csv"
        metrics, digest = compute(path)
        rows.append({"Experiment": args.experiment, "Seed": seed, **metrics})
        inputs.append({"path": str(path.resolve()), "sha256": digest})
    summary = []
    for metric in METRICS:
        values = [row[metric] for row in rows]
        valid = all(math.isfinite(value) for value in values)
        summary.append({"Metric": metric, "Runs": len(values),
                        "Mean": statistics.mean(values) if valid else float("nan"),
                        "SD": statistics.stdev(values) if valid else float("nan")})
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "mean_sd.csv", summary)
    report = {"experiment": args.experiment, "inputs": inputs,
              "scope": "Recomputation from saved predictions only; not full experimental provenance verification.",
              "auc": "Flattened two-column probabilities and one-hot labels (benign=0, malignant=1).",
              "sd": "Sample standard deviation (ddof=1).",
              "undefined": "Undefined ratios are NaN; summary is NaN if any run is undefined. Weighted precision uses zero for undefined class precision."}
    (output / "input_record.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved prediction statistics: {output}")


if __name__ == "__main__":
    main()
