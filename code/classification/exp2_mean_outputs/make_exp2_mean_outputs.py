from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

root = (RESULT)
parser = argparse.ArgumentParser(description="Plot formal Exp-2 results using flattened two-class ROC.")
parser.add_argument("--output-dir", default=str(OUTPUT / "exp2_roc"), help="New directory for corrected figures and CSV files")
args = parser.parse_args()
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=False)

seeds = [48271, 59317, 84629]

metric_rows = []
cm_rows = []
interp_tprs = []
all_fpr = np.linspace(0, 1, 500)

plt.figure(figsize=(6.5, 5.5))

for seed in seeds:
    run_dir = os.path.join(root, f"fusion_exp2_100ep_seed{seed}_final_multiseed")

    metric_path = os.path.join(run_dir, "author_style_metrics.csv")
    pred_path = os.path.join(run_dir, "test_predictions_two_class.csv")

    mdf = pd.read_csv(metric_path)
    pdf = pd.read_csv(pred_path)

    y_true = pdf["True Label"].astype(int).to_numpy()
    y_pred = pdf["Predicted Label"].astype(int).to_numpy()
    probabilities = pdf[
        ["Benign Probability", "Malignant Probability"]
    ].to_numpy(dtype=float)

    one_hot = np.eye(2, dtype=int)[y_true]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    fpr, tpr, _ = roc_curve(
        one_hot.flatten(),
        probabilities.flatten()
    )
    roc_auc = auc(fpr, tpr)

    saved_auc = float(mdf.loc[0, "AUC"])
    if not np.isclose(roc_auc, saved_auc, atol=1e-12, rtol=0):
        raise ValueError(
            f"Seed {seed}: recalculated AUC={roc_auc}, "
            f"saved AUC={saved_auc}"
        )

    interp_tpr = np.interp(all_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

    plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f"Seed {seed} (AUC={roc_auc:.4f})")

    metric_rows.append({
        "Seed": seed,
        "Accuracy": float(mdf.loc[0, "Accuracy"]),
        "Sensitivity": float(mdf.loc[0, "Sensitivity"]),
        "Specificity": float(mdf.loc[0, "Specificity"]),
        "Precision_Malignant": float(mdf.loc[0, "Precision_Malignant"]),
        "Precision_Weighted": float(mdf.loc[0, "Precision_Weighted"]),
        "AUC": float(roc_auc)
    })

    cm_rows.append({
        "Seed": seed,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0

auc_vals = [x["AUC"] for x in metric_rows]
mean_auc = float(np.mean(auc_vals))
std_auc = float(np.std(auc_vals, ddof=1))

plt.plot(all_fpr, mean_tpr, lw=3, label=f"Mean ROC (AUC={mean_auc:.4f} \u00b1 {std_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", lw=1.2, label="Random classifier")
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve of Exp-2 Across Three Independent Runs")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "exp2_mean_roc.png"), dpi=300)
plt.close()

cm_df = pd.DataFrame(cm_rows)
mean_cm = cm_df[["TN", "FP", "FN", "TP"]].mean()

mean_cm_matrix = np.array([
    [mean_cm["TN"], mean_cm["FP"]],
    [mean_cm["FN"], mean_cm["TP"]]
], dtype=float)

pd.DataFrame(metric_rows).to_csv(os.path.join(output_dir, "exp2_auc_summary.csv"), index=False)
cm_df.to_csv(os.path.join(output_dir, "exp2_confusion_per_seed.csv"), index=False)
pd.DataFrame(mean_cm_matrix,
             index=["True Benign", "True Malignant"],
             columns=["Predicted Benign", "Predicted Malignant"]).to_csv(
    os.path.join(output_dir, "exp2_mean_confusion_matrix.csv")
)

fig, ax = plt.subplots(figsize=(6, 5.2))
im = ax.imshow(mean_cm_matrix, cmap="Greens")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted Benign", "Predicted Malignant"])
ax.set_yticklabels(["True Benign", "True Malignant"])
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Mean Confusion Matrix of Exp-2")

for i in range(2):
    for j in range(2):
        val = mean_cm_matrix[i, j]
        text_str = f"{val:.1f}"
        ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=12)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "exp2_mean_confusion_matrix.png"), dpi=300)
plt.close()

report_lines = []
report_lines.append(f"Mean AUC = {mean_auc:.6f} \u00b1 {std_auc:.6f}")
report_lines.append("")
report_lines.append("Mean confusion matrix:")
report_lines.append(f"TN = {mean_cm['TN']:.4f}")
report_lines.append(f"FP = {mean_cm['FP']:.4f}")
report_lines.append(f"FN = {mean_cm['FN']:.4f}")
report_lines.append(f"TP = {mean_cm['TP']:.4f}")

with open(os.path.join(output_dir, "exp2_mean_outputs_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Done.")
print("Saved:", os.path.join(output_dir, "exp2_mean_roc.png"))
print("Saved:", os.path.join(output_dir, "exp2_auc_summary.csv"))
print("Saved:", os.path.join(output_dir, "exp2_confusion_per_seed.csv"))
print("Saved:", os.path.join(output_dir, "exp2_mean_confusion_matrix.csv"))
print("Saved:", os.path.join(output_dir, "exp2_mean_confusion_matrix.png"))
print("Saved:", os.path.join(output_dir, "exp2_mean_outputs_report.txt"))
print("")
print(f"Mean AUC = {mean_auc:.6f} \u00b1 {std_auc:.6f}")
print("")
print("Mean confusion matrix:")
print(pd.DataFrame(mean_cm_matrix,
                   index=["True Benign", "True Malignant"],
                   columns=["Predicted Benign", "Predicted Malignant"]))
