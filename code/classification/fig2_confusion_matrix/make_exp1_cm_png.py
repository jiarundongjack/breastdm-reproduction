from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = OUTPUT / 'exp1_mean_confusion_matrix.csv'
if not csv_path.exists():
    csv_path = PAPER / 'exp1_figures/exp1_mean_confusion_matrix.csv'
out_path = (OUTPUT / 'exp1_mean_confusion_matrix.png')

df = pd.read_csv(csv_path, index_col=0)
mat = df.apply(pd.to_numeric, errors="coerce").values

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(mat, cmap="Blues")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Count")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Predicted Benign", "Predicted Malignant"], fontsize=12)

ax.set_yticks([0, 1])
ax.set_yticklabels(["True Benign", "True Malignant"], fontsize=12)

ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)
ax.set_title("Mean Confusion Matrix of Exp-1", fontsize=18, pad=12)

threshold = mat.max() / 2.0
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(
            j, i, f"{mat[i, j]:.0f}",
            ha="center", va="center",
            color="white" if mat[i, j] > threshold else "black",
            fontsize=20
        )

ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)
