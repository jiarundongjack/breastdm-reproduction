"""Verify saved classification metrics and report both AUC definitions.

Derived from project summarize_predictions.py at b26a3def1542dbaa0a1ec8997f6c5c96d34ce856.
No inference, training, or checkpoint selection is performed.
"""
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
    parser.add_argument('--experiment', choices=SEEDS, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output must be a new directory')
    results, inputs = [], []
    metric_names = tuple(k for k in METRICS if k != 'AUC') + ('AUC_flattened_two_column', 'AUROC_malignant_class')
    for seed in SEEDS[args.experiment]:
        directory=args.runs_root/f'fusion_{args.experiment}_100ep_seed{seed}_final_multiseed'
        path=directory/'test_predictions_two_class.csv'
        metrics, digest=compute(path)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('Input changed during read')
        saved=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        y=[int(r['True Label']) for r in saved]
        probabilities=[float(r['Malignant Probability']) for r in saved]
        if len({r['Index'] for r in saved})!=len(saved): raise ValueError('Duplicate sample index')
        if len(y)!=403 or y.count(1)!=289 or y.count(0)!=114:
            raise ValueError('This verifier expects the formal 403-sample BreastDM classification test set')
        auc=rank_auc(y,probabilities)
        positives=[p for p,t in zip(probabilities,y) if t==1]
        negatives=[p for p,t in zip(probabilities,y) if t==0]
        independent=sum((a>b)+0.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives))
        if abs(auc-independent)>1e-14: raise ValueError('Independent AUROC check failed')
        original_path=directory/'author_style_metrics.csv'
        original_raw=original_path.read_bytes()
        original_rows=list(csv.DictReader(io.StringIO(original_raw.decode('utf-8-sig'))))
        if len(original_rows)!=1: raise ValueError('Expected one official metrics row')
        original=original_rows[0]
        errors={k:abs(metrics[k]-float(original[k])) for k in METRICS}
        if not all(math.isfinite(e) and e<1e-12 for e in errors.values()):
            raise ValueError(f'Official metric mismatch: {errors}')
        for p,b in [(path,raw),(original_path,original_raw)]:
            inputs.append({'path':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest()})
        flat=metrics.pop('AUC')
        results.append({'experiment':args.experiment,'seed':seed,**metrics,
                        'AUC_flattened_two_column':flat,'AUROC_malignant_class':auc,
                        'independent_pairwise_AUROC':independent,'absolute_errors':errors})
    summary={k:{'mean':statistics.mean(r[k] for r in results),
                'sample_SD_ddof_1':statistics.stdev(r[k] for r in results)} for k in metric_names}
    for item in inputs:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Input changed during verification')
    report={'experiment':args.experiment,'per_run':results,'summary':summary,'inputs':inputs,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_hashes_unchanged':True,
            'method':'Malignant=1; benign=0; malignant-class AUROC uses malignant probability. Flattened AUC uses both one-hot columns. Ties receive half credit. Sample SD across three seeds; not a confidence interval.',
            'scope':'Saved predictions from existing selected checkpoints. No training, inference or reselection. Does not establish published-result provenance.'}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
