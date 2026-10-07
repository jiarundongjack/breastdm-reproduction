"""Shared formal metrics: malignant=1, benign=0, rates in [0,1].

AUC-based selection is our reproduction rule based on the released branch;
the manuscript does not identify the branch used for its published results.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_score

METRIC_FIELDS = ("Accuracy", "Sensitivity", "Specificity", "Precision", "AUC")
RESULT_FIELDS = ("Experiment", "Seed", *METRIC_FIELDS)
TEST_METRIC_FIELDS = (*METRIC_FIELDS, "Precision_Malignant", "Precision_Weighted")
TEST_RESULT_FIELDS = ("Experiment", "Seed", *TEST_METRIC_FIELDS)
EXP1_TEST_METRIC_FIELDS = (*TEST_METRIC_FIELDS, "TP", "TN", "FP", "FN", "N")
EXP1_TEST_RESULT_FIELDS = ("Experiment", "Seed", *EXP1_TEST_METRIC_FIELDS)


def classification_roc(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.shape != (len(labels), 2):
        raise ValueError("AUC requires labels [N] and original softmax probabilities [N,2].")
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected nonempty binary labels (benign=0, malignant=1).")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Invalid softmax probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Softmax rows must sum to one.")
    one_hot = np.eye(2, dtype=int)[labels.astype(int)]
    return roc_curve(one_hot.flatten(), probabilities.flatten())


def classification_auc(labels, probabilities):
    fpr, tpr, _ = classification_roc(labels, probabilities)
    return float(auc(fpr, tpr))


def classification_metrics(labels, probabilities, predictions):
    """Global sample confusion counts, plus flattened two-class AUC.

    Require hard predictions from logits argmax; never infer them from probabilities.
    Undefined ratios return NaN (not an invented zero); an empty set is rejected.
    """
    score = classification_auc(labels, probabilities)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    if predictions.shape != labels.shape or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Predictions must be binary and aligned with labels.")
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    return {
        "Accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "Sensitivity": ratio(tp, tp + fn),
        "Specificity": ratio(tn, tn + fp),
        "Precision": ratio(tp, tp + fp),
        "AUC": score,
    }


def classification_test_metrics(labels, probabilities, predictions, *, include_counts=False):
    """Evaluate the full Test set; retain legacy Precision as an exact alias.

    Neither precision is a mean of batch metrics. Validation keeps using
    classification_metrics without changes to its AUC or selection behavior.
    """
    metrics = classification_metrics(labels, probabilities, predictions)
    metrics["Precision_Malignant"] = metrics["Precision"]
    metrics["Precision_Weighted"] = float(precision_score(
        labels, predictions, average="weighted", zero_division=0
    ))
    if include_counts:
        labels, predictions = np.asarray(labels), np.asarray(predictions)
        metrics.update(
            TP=int(np.sum((labels == 1) & (predictions == 1))),
            TN=int(np.sum((labels == 0) & (predictions == 0))),
            FP=int(np.sum((labels == 0) & (predictions == 1))),
            FN=int(np.sum((labels == 1) & (predictions == 0))),
            N=len(labels),
        )
    return metrics


def metrics_from_frame(frame):
    return classification_metrics(
        frame["True Label"].to_numpy(),
        frame[["Benign Probability", "Malignant Probability"]].to_numpy(),
        frame["Predicted Label"].to_numpy(),
    )


def read_auc_predictions(path):
    """Prefer newly evaluated raw columns; never reconstruct benign probability."""
    import pandas as pd
    path = Path(path)
    raw = path.with_name("test_predictions_two_class.csv")
    selected = raw if raw.exists() else path
    frame = pd.read_csv(selected, float_precision="round_trip")
    required = ["Benign Probability", "Malignant Probability"]
    if not all(column in frame for column in required):
        raise ValueError(f"{selected}: raw two-class probabilities missing; evaluate the existing checkpoint.")
    return frame
