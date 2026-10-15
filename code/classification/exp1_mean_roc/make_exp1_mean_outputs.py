from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

root = str(RESULT)

run_dirs = [
    "fusion_exp1_100ep_seed16180_final_multiseed",
    "fusion_exp1_100ep_seed27182_final_multiseed",
    "fusion_exp1_100ep_seed31415_final_multiseed",
]

common_fpr = np.linspace(0, 1, 1001)

roc_rows = []
cm_rows = []
interp_tprs = []

plt.figure(figsize=(7, 6))

for run_dir in run_dirs:
    seed = run_dir.split("seed")[1].split("_")[0]
    csv_path = os.path.join(root, run_dir, "test_predictions_two_class.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # ===== 分类标签 =====
    y_true = df["True Label"].astype(int).to_numpy()
    y_pred = df["Predicted Label"].astype(int).to_numpy()

    # ===== 概率 =====
    p_m = df["Malignant Probability"].astype(float).to_numpy()
    p_b = df["Benign Probability"].astype(float).to_numpy()

    # ===== author-released-code-compatible flattened AUC / ROC =====
    # one-hot labels: benign=0, malignant=1
    y_onehot = np.column_stack([(y_true == 0).astype(int), (y_true == 1).astype(int)])
    p_two = np.column_stack([p_b, p_m])

    y_flat = y_onehot.reshape(-1)
    p_flat = p_two.reshape(-1)

    fpr, tpr, _ = roc_curve(y_flat, p_flat)
    roc_auc = auc(fpr, tpr)

    interp_tpr = np.interp(common_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    interp_tprs.append(interp_tpr)

    roc_rows.append({
        "Seed": int(seed),
        "AUC": roc_auc
    })

    plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f"Seed {seed} (AUC={roc_auc:.4f})")

    # ===== confusion matrix =====
    # label 0 = benign, 1 = malignant
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cm_rows.append({
        "Seed": int(seed),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "Precision_Malignant": tp / (tp + fp) if (tp + fp) > 0 else np.nan
    })

# ===== mean ROC =====
mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[0] = 0.0
mean_tpr[-1] = 1.0

auc_values = [x["AUC"] for x in roc_rows]
mean_auc = float(np.mean(auc_values))
std_auc = float(np.std(auc_values, ddof=1))

plt.plot(common_fpr, mean_tpr, lw=3, label=f"Mean ROC (AUC={mean_auc:.4f} ± {std_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", lw=1.2, label="Random classifier")

plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve of Exp-1 Across Three Independent Runs")
plt.legend(loc="lower right")
plt.tight_layout()

roc_fig_path = os.path.join(OUTPUT, "exp1_mean_roc.png")
plt.savefig(roc_fig_path, dpi=300)
plt.close()

# ===== mean confusion matrix =====
cm_df = pd.DataFrame(cm_rows).sort_values("Seed")
mean_tn = cm_df["TN"].mean()
mean_fp = cm_df["FP"].mean()
mean_fn = cm_df["FN"].mean()
mean_tp = cm_df["TP"].mean()

mean_cm_df = pd.DataFrame({
    "Predicted Benign": [mean_tn, mean_fn],
    "Predicted Malignant": [mean_fp, mean_tp]
}, index=["True Benign", "True Malignant"])

# ===== save outputs =====
pd.DataFrame(roc_rows).sort_values("Seed").to_csv(os.path.join(OUTPUT, "exp1_auc_summary.csv"), index=False)
cm_df.to_csv(os.path.join(OUTPUT, "exp1_confusion_per_seed.csv"), index=False)
mean_cm_df.to_csv(os.path.join(OUTPUT, "exp1_mean_confusion_matrix.csv"))

# also save a simple txt report
report_path = os.path.join(OUTPUT, "exp1_mean_outputs_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Exp-1 Mean ROC and Mean Confusion Matrix Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write("AUC per seed:\n")
    for row in sorted(roc_rows, key=lambda x: x["Seed"]):
        f.write(f"Seed {row['Seed']}: AUC = {row['AUC']:.6f}\n")
    f.write(f"\nMean AUC = {mean_auc:.6f}\n")
    f.write(f"SD AUC   = {std_auc:.6f}\n\n")

    f.write("Confusion matrix per seed (rows=True, cols=Pred):\n")
    for _, row in cm_df.iterrows():
        f.write(
            f"Seed {int(row['Seed'])}: "
            f"TN={int(row['TN'])}, FP={int(row['FP'])}, FN={int(row['FN'])}, TP={int(row['TP'])}\n"
        )

    f.write("\nMean confusion matrix:\n")
    f.write(mean_cm_df.to_string())
    f.write("\n")

print("Done.")
print("Saved:", roc_fig_path)
print("Saved:", os.path.join(OUTPUT, "exp1_auc_summary.csv"))
print("Saved:", os.path.join(OUTPUT, "exp1_confusion_per_seed.csv"))
print("Saved:", os.path.join(OUTPUT, "exp1_mean_confusion_matrix.csv"))
print("Saved:", report_path)

print("\nMean AUC = {:.6f} ± {:.6f}".format(mean_auc, std_auc))
print("\nMean confusion matrix:")
print(mean_cm_df)
