from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

root = (RESULT)
seeds = [48271, 59317, 84629]
prefix = "fusion_exp2_100ep_seed{}_final_multiseed"
csv_name = "test_predictions_two_class.csv"

cms = []

for seed in seeds:
    csv_path = os.path.join(root, prefix.format(seed), csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    y_true = df["True Label"].astype(int)
    y_pred = df["Predicted Label"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cms.append(cm)

cms = np.array(cms, dtype=float)

# element-wise mean confusion matrix
mean_cm = cms.mean(axis=0)

# rounded integer matrix for display
disp_cm = np.rint(mean_cm).astype(int)

# save exact mean and displayed integer version
mean_df = pd.DataFrame(
    mean_cm,
    index=["True Benign", "True Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
)
disp_df = pd.DataFrame(
    disp_cm,
    index=["True Benign", "True Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
)

mean_csv = os.path.join(OUTPUT, "exp2_mean_confusion_matrix_exact.csv")
disp_csv = os.path.join(OUTPUT, "exp2_mean_confusion_matrix_integer.csv")
mean_df.to_csv(mean_csv)
disp_df.to_csv(disp_csv)

# plot
fig, ax = plt.subplots(figsize=(5.8, 4.8))
im = ax.imshow(disp_cm, cmap="OrRd")

# colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Count")

# ticks and labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted Benign", "Predicted Malignant"])
ax.set_yticklabels(["True Benign", "True Malignant"])

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Mean Confusion Matrix of Exp-2")

# annotate integer values
max_val = disp_cm.max()
for i in range(disp_cm.shape[0]):
    for j in range(disp_cm.shape[1]):
        val = disp_cm[i, j]
        color = "white" if val > max_val * 0.55 else "black"
        ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=12)

plt.tight_layout()

fig_path = os.path.join(OUTPUT, "exp2_mean_confusion_matrix_integer.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()

# save note
note_path = os.path.join(OUTPUT, "exp2_mean_confusion_matrix_note.txt")
with open(note_path, "w", encoding="utf-8") as f:
    f.write("Displayed confusion-matrix entries are rounded to the nearest integers ")
    f.write("from the element-wise arithmetic mean across the three independent runs.\n")
    f.write("\nExact mean confusion matrix:\n")
    f.write(mean_df.to_string())
    f.write("\n\nDisplayed integer confusion matrix:\n")
    f.write(disp_df.to_string())

print("Saved figure:", fig_path)
print("Saved exact mean CSV:", mean_csv)
print("Saved integer CSV:", disp_csv)
print("Saved note:", note_path)
print("\nExact mean confusion matrix:")
print(mean_df)
print("\nDisplayed integer confusion matrix:")
print(disp_df)