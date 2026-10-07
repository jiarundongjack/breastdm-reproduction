import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

run = os.getcwd()

cand_csv = os.path.join(run, "top20_overseg_candidates.csv")
pred_csv = os.path.join(run, "test_predictions.csv")

out_dir = os.path.join(run, "top3_overseg_figures")
os.makedirs(out_dir, exist_ok=True)

# ========= helper functions =========
def load_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
    else:
        arr = np.array(Image.open(path))
    return np.array(arr)

def normalize_to_gray(img):
    img = np.array(img, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., 0]
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img)
    return img

def pick_slice(img, gt, pred):
    """
    If 3D, choose the slice with the largest GT area.
    If GT is empty, choose largest pred area.
    Otherwise choose center slice.
    """
    if gt.ndim == 2:
        return img, gt, pred, None

    # make sure shapes are compatible
    zdim = gt.shape[0]

    gt_area = gt.reshape(zdim, -1).sum(axis=1)
    pred_area = pred.reshape(zdim, -1).sum(axis=1)

    if gt_area.max() > 0:
        z = int(np.argmax(gt_area))
    elif pred_area.max() > 0:
        z = int(np.argmax(pred_area))
    else:
        z = zdim // 2

    img2 = img[z] if img.ndim == 3 else img
    gt2 = gt[z]
    pred2 = pred[z]

    return img2, gt2, pred2, z

def binarize(arr):
    arr = np.array(arr)
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if arr.max() <= 1:
        return (arr > 0.5).astype(np.uint8)
    return (arr > 0).astype(np.uint8)

def contour_overlay(ax, img, gt, pred):
    ax.imshow(img, cmap="gray")
    if gt.sum() > 0:
        ax.contour(gt, levels=[0.5], colors=["lime"], linewidths=2)
    if pred.sum() > 0:
        ax.contour(pred, levels=[0.5], colors=["red"], linewidths=2)
    ax.set_xticks([])
    ax.set_yticks([])

def try_find_pred_path(row):
    possible_cols = [
        "pred_path", "prediction_path", "pred_mask_path", "pred_npy_path",
        "prediction_npy", "pred_file", "prediction_file"
    ]
    for c in possible_cols:
        if c in row.index and pd.notna(row[c]):
            p = str(row[c])
            if os.path.exists(p):
                return p

    # If not found, search for any path-like field that exists and contains 'pred'
    for c in row.index:
        val = row[c]
        if isinstance(val, str) and ("pred" in c.lower() or "pred" in val.lower()):
            if os.path.exists(val):
                return val

    return None

def get_row_by_index(df, idx):
    if "index" in df.columns:
        hit = df[df["index"] == idx]
        if len(hit) > 0:
            return hit.iloc[0]
    return None

# ========= read csv =========
if not os.path.exists(cand_csv):
    raise FileNotFoundError(f"Not found: {cand_csv}")
if not os.path.exists(pred_csv):
    raise FileNotFoundError(f"Not found: {pred_csv}")

cand = pd.read_csv(cand_csv)
pred_df = pd.read_csv(pred_csv)

print("Loaded candidates:", cand_csv)
print("Loaded predictions:", pred_csv)
print("Candidate columns:", list(cand.columns))
print("Prediction columns:", list(pred_df.columns))

required = ["index", "patient", "sequence", "filename", "dice", "iou", "ppv", "overseg_score"]
for c in required:
    if c not in cand.columns:
        raise RuntimeError(f"top20_overseg_candidates.csv 缺少关键列: {c}")

# 综合排序：先过分割指数高，再PPV低，再Dice低
cand = cand.sort_values(
    by=["overseg_score", "ppv", "dice"],
    ascending=[False, True, True]
).reset_index(drop=True)

top3 = cand.head(3).copy()
top3_save = os.path.join(out_dir, "selected_top3_overseg_cases.csv")
top3.to_csv(top3_save, index=False)
print("\nSelected top 3 over-segmentation cases:")
print(top3[["index", "patient", "sequence", "filename", "dice", "iou", "ppv", "overseg_score"]].to_string(index=False))
print("\nSaved:", top3_save)

# ========= make per-case figures =========
combined_fig, combined_axes = plt.subplots(3, 4, figsize=(18, 13))
combined_fig.subplots_adjust(hspace=0.45, wspace=0.08)
panel_titles = ["MRI", "Ground Truth", "Prediction", "Overlay"]

for rank, (_, row) in enumerate(top3.iterrows(), start=1):
    idx = row["index"]
    pred_row = get_row_by_index(pred_df, idx)

    if pred_row is None:
        raise RuntimeError(f"test_predictions.csv 里找不到 index={idx} 的记录。")

    image_path = None
    mask_path = None
    for c in ["image_path", "img_path", "image_file"]:
        if c in pred_row.index and pd.notna(pred_row[c]) and os.path.exists(str(pred_row[c])):
            image_path = str(pred_row[c])
            break

    for c in ["mask_path", "gt_path", "label_path", "mask_file"]:
        if c in pred_row.index and pd.notna(pred_row[c]) and os.path.exists(str(pred_row[c])):
            mask_path = str(pred_row[c])
            break

    pred_path = try_find_pred_path(pred_row)

    if image_path is None:
        raise RuntimeError(f"index={idx} 找不到 image_path。")
    if mask_path is None:
        raise RuntimeError(f"index={idx} 找不到 mask_path。")
    if pred_path is None:
        raise RuntimeError(
            f"index={idx} 找不到 prediction 路径。\n"
            f"请把 test_predictions.csv 的列名截图给我，或者把包含 prediction 的那一列名字补到脚本里。"
        )

    img = load_array(image_path)
    gt = load_array(mask_path)
    pred = load_array(pred_path)

    gt = binarize(gt)
    pred = binarize(pred)

    img2, gt2, pred2, z = pick_slice(img, gt, pred)
    img2 = normalize_to_gray(img2)
    gt2 = binarize(gt2)
    pred2 = binarize(pred2)

    # case-level figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    fig.subplots_adjust(top=0.80, wspace=0.05)

    axes[0].imshow(img2, cmap="gray")
    axes[1].imshow(gt2, cmap="gray")
    axes[2].imshow(pred2, cmap="gray")
    contour_overlay(axes[3], img2, gt2, pred2)

    for j in range(4):
        axes[j].set_title(panel_titles[j], fontsize=14)
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    info1 = f"Case {rank} | index={idx} | {row['patient']} | {row['sequence']} | {row['filename']}"
    info2 = (
        f"DSC={row['dice']:.4f} | IoU={row['iou']:.4f} | PPV={row['ppv']:.4f} | "
        f"OverSeg={row['overseg_score']:.4f}"
    )
    if z is not None:
        info3 = f"Displayed slice: z={z}"
    else:
        info3 = "Displayed slice: 2D input"

    fig.suptitle(f"{info1}\n{info2}\n{info3}", fontsize=13)

    legend_handles = [
        Patch(facecolor='none', edgecolor='lime', label='Ground Truth contour'),
        Patch(facecolor='none', edgecolor='red',  label='Prediction contour')
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11)

    single_save = os.path.join(out_dir, f"case{rank}_index{idx}_overseg.png")
    fig.savefig(single_save, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", single_save)

    # combined figure row
    caxes = combined_axes[rank - 1]
    caxes[0].imshow(img2, cmap="gray")
    caxes[1].imshow(gt2, cmap="gray")
    caxes[2].imshow(pred2, cmap="gray")
    contour_overlay(caxes[3], img2, gt2, pred2)

    for j in range(4):
        caxes[j].set_title(panel_titles[j], fontsize=12)
        caxes[j].set_xticks([])
        caxes[j].set_yticks([])

    row_label = (
        f"Case {rank}\nindex={idx}\n{row['patient']}\n"
        f"DSC={row['dice']:.3f}, IoU={row['iou']:.3f}\nPPV={row['ppv']:.3f}, OverSeg={row['overseg_score']:.3f}"
    )
    caxes[0].set_ylabel(row_label, fontsize=11, rotation=0, labelpad=55, va="center")

combined_fig.suptitle(
    "Top 3 Over-Segmentation Failure Cases (3D U-Net)\n"
    "Ground Truth contour = green, Prediction contour = red",
    fontsize=15
)
combined_save = os.path.join(out_dir, "top3_overseg_combined.png")
combined_fig.savefig(combined_save, dpi=300, bbox_inches="tight")
plt.close(combined_fig)

print("\nDone.")
print("Combined figure saved:", combined_save)
print("Output folder:", out_dir)
