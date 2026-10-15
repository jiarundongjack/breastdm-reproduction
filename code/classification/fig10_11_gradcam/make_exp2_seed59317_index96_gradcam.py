from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import sys
import inspect
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = (REPO)

RUN_DIR = os.path.join(
    RESULT,
    "fusion_exp2_100ep_seed59317_final_multiseed"
)

SNAPSHOT_DIR = os.path.join(RUN_DIR, "code_snapshot")

DATA_ROOT = (EXP2_DATA)

CHECKPOINT = os.path.join(
    RUN_DIR,
    "best_fusion_model.pth"
)

PREDICTION_CSV = os.path.join(
    RUN_DIR,
    "test_predictions_two_class.csv"
)

OUTPUT_DIR = os.path.join(
    OUTPUT,
    "exp2_seed59317_index96_gradcam"
)

INDEX = 96

# Display channel.
# Channel 8 is the first channel of the verified 9-channel block
# Exp-2[:, :, 8:17] that corresponds to the Exp-1 input block.
DISPLAY_CHANNEL = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def load_module_from_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of these files exists:\n" + "\n".join(paths)
    )


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())

    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)

    return (x - mn) / (mx - mn)


# ============================================================
# 1. Check formal prediction for Index 96
# ============================================================

df = pd.read_csv(PREDICTION_CSV)

row = df.loc[df["Index"] == INDEX]

if len(row) != 1:
    raise RuntimeError(
        f"Expected exactly one row for Index {INDEX}, found {len(row)}"
    )

row = row.iloc[0]

csv_true = int(row["True Label"])
csv_pred = int(row["Predicted Label"])
csv_p_m = float(row["Malignant Probability"])
csv_p_b = float(row["Benign Probability"])

print("=" * 70)
print("FORMAL CSV RECORD")
print("=" * 70)
print("Index:", INDEX)
print("True Label:", csv_true)
print("Predicted Label:", csv_pred)
print("Benign Probability:", csv_p_b)
print("Malignant Probability:", csv_p_m)

if csv_true != 0 or csv_pred != 1:
    raise RuntimeError(
        "Index 96 is not the expected Benign -> Malignant false positive."
    )


# ============================================================
# 2. Import exact formal Exp-2 model implementation
# ============================================================

sys.path.insert(0, SNAPSHOT_DIR)
sys.path.insert(0, ROOT)

fusion_file = first_existing([
    os.path.join(SNAPSHOT_DIR, "fusionModels_exp2.py"),
    os.path.join(RUNTIME, "fusionModels_exp2.py"),
])

print("\nModel source:")
print(fusion_file)

fusion_module = load_module_from_file(
    "formal_fusionModels_exp2_gradcam",
    fusion_file
)

FusionM = fusion_module.FusionM

sig = inspect.signature(FusionM)
kwargs = {}

if "num_classes" in sig.parameters:
    kwargs["num_classes"] = 2

# Do not reload pretrained weights here.
# The formal best checkpoint already contains trained weights.
if "load_vit" in sig.parameters:
    kwargs["load_vit"] = False

if "load_se" in sig.parameters:
    kwargs["load_se"] = False

print("FusionM constructor arguments:", kwargs)

model = FusionM(**kwargs)


# ============================================================
# 3. Load seed 59317 formal best checkpoint
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Checkpoint:", CHECKPOINT)

state = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

# Robustly unwrap state dict if necessary
if isinstance(state, dict):
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model_state_dict" in state and isinstance(
        state["model_state_dict"], dict
    ):
        state = state["model_state_dict"]

# Remove DataParallel prefix if present
clean_state = {}

for k, v in state.items():
    if k.startswith("module."):
        clean_state[k[7:]] = v
    else:
        clean_state[k] = v

load_result = model.load_state_dict(
    clean_state,
    strict=True
)

print("Checkpoint loaded successfully.")

model = model.to(device)
model.eval()


# ============================================================
# 4. Import formal Exp-2 dataset class
# ============================================================

loader_file = first_existing([
    os.path.join(SNAPSHOT_DIR, "data_loader_exp2.py"),
    os.path.join(RUNTIME, "data_loader_exp2.py"),
])

print("\nDataset source:")
print(loader_file)

loader_module = load_module_from_file(
    "formal_data_loader_exp2_gradcam",
    loader_file
)

BreastDMNpyDataset = loader_module.BreastDMNpyDataset

test_dataset = BreastDMNpyDataset(
    root=DATA_ROOT,
    split="test",
    augment=False
)

print("Test dataset length:", len(test_dataset))

if INDEX >= len(test_dataset):
    raise IndexError(
        f"Index {INDEX} exceeds dataset length {len(test_dataset)}"
    )

item = test_dataset[INDEX]

if not isinstance(item, (tuple, list)) or len(item) < 2:
    raise RuntimeError(
        "Unexpected dataset return format."
    )

x = item[0]
true_label = int(item[1])

print("\nDataset Index:", INDEX)
print("Dataset true label:", true_label)
print("Input tensor shape:", tuple(x.shape))

if true_label != csv_true:
    raise RuntimeError(
        f"Dataset/CSV label mismatch: dataset={true_label}, csv={csv_true}"
    )

if x.ndim != 3 or x.shape[0] != 17:
    raise RuntimeError(
        f"Expected [17,H,W] input, got {tuple(x.shape)}"
    )

x = x.unsqueeze(0).float().to(device)


# ============================================================
# 5. Grad-CAM hook
#
# Hook the fused 1024-channel spatial representation that enters
# the model's global average pooling layer.
# ============================================================

storage = {}


def avgpool_hook(module, inputs, output):

    feature = inputs[0]

    storage["activation"] = feature

    def save_gradient(grad):
        storage["gradient"] = grad

    feature.register_hook(save_gradient)


if not hasattr(model, "avgpool"):
    raise AttributeError(
        "Formal FusionM does not contain model.avgpool."
    )

handle = model.avgpool.register_forward_hook(
    avgpool_hook
)


# ============================================================
# 6. Forward pass and verify prediction
# ============================================================

model.zero_grad(set_to_none=True)

logits = model(x)

probabilities = torch.softmax(
    logits,
    dim=1
)

pred_label = int(
    torch.argmax(probabilities, dim=1).item()
)

p_b = float(probabilities[0, 0].item())
p_m = float(probabilities[0, 1].item())

print("\n" + "=" * 70)
print("RECOMPUTED FORMAL MODEL PREDICTION")
print("=" * 70)
print("Predicted Label:", pred_label)
print("Benign Probability:", p_b)
print("Malignant Probability:", p_m)

if pred_label != csv_pred:
    raise RuntimeError(
        f"Prediction mismatch: rerun={pred_label}, csv={csv_pred}"
    )

if abs(p_m - csv_p_m) > 5e-5:
    raise RuntimeError(
        "Malignant probability does not reproduce the formal CSV.\n"
        f"Rerun={p_m:.8f}, CSV={csv_p_m:.8f}\n"
        "Stop here rather than generating a potentially mismatched Grad-CAM."
    )

print(
    "Prediction matches formal CSV."
)


# ============================================================
# 7. Backpropagate malignant logit
# ============================================================

malignant_score = logits[0, 1]

malignant_score.backward()

handle.remove()

if "activation" not in storage:
    raise RuntimeError(
        "No feature activation captured."
    )

if "gradient" not in storage:
    raise RuntimeError(
        "No feature gradient captured."
    )

activation = storage["activation"].detach()[0]
gradient = storage["gradient"].detach()[0]

print("Captured fused feature shape:",
      tuple(activation.shape))

print("Captured gradient shape:",
      tuple(gradient.shape))


# ============================================================
# 8. Standard Grad-CAM
# ============================================================

weights = gradient.mean(
    dim=(1, 2),
    keepdim=True
)

cam = torch.sum(
    weights * activation,
    dim=0
)

cam = torch.relu(cam)

cam = cam.unsqueeze(0).unsqueeze(0)

cam = F.interpolate(
    cam,
    size=x.shape[-2:],
    mode="bilinear",
    align_corners=False
)

cam = cam[0, 0].cpu().numpy()

cam = normalize01(cam)


# ============================================================
# 9. Exact model input image used for display
# ============================================================

display_img = (
    x[0, DISPLAY_CHANNEL]
    .detach()
    .cpu()
    .numpy()
)

display_img = normalize01(display_img)


# ============================================================
# 10. Save raw input image
# ============================================================

fig = plt.figure(figsize=(5, 5))

plt.imshow(
    display_img,
    cmap="gray"
)

plt.axis("off")

plt.tight_layout(pad=0)

input_path = os.path.join(
    OUTPUT_DIR,
    "a_index96_benign_input.png"
)

plt.savefig(
    input_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

plt.close(fig)


# ============================================================
# 11. Save Grad-CAM overlay
# ============================================================

fig = plt.figure(figsize=(5, 5))

plt.imshow(
    display_img,
    cmap="gray"
)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)

plt.axis("off")

plt.tight_layout(pad=0)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "b_index96_gradcam_malignant.png"
)

plt.savefig(
    overlay_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

plt.close(fig)



# ============================================================
# 12. Save final two-panel manuscript figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.4, 4.25)
)

# -------------------------
# (a) Original benign input
# -------------------------

axes[0].imshow(
    display_img,
    cmap="gray"
)

axes[0].axis("off")

axes[0].text(
    0.5,
    -0.055,
    "(a) Benign input",
    transform=axes[0].transAxes,
    ha="center",
    va="top",
    fontsize=9.5
)


# -------------------------
# (b) Grad-CAM
# -------------------------

axes[1].imshow(
    display_img,
    cmap="gray"
)

axes[1].imshow(
    cam,
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)

axes[1].axis("off")

axes[1].text(
    0.5,
    -0.055,
    f"(b) Grad-CAM for malignant prediction, P(Malignant) = {p_m:.6f}",
    transform=axes[1].transAxes,
    ha="center",
    va="top",
    fontsize=8.5
)


# Keep both panels aligned
plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.98,
    bottom=0.12,
    wspace=0.08
)

pair_path = os.path.join(
    OUTPUT_DIR,
    "exp2_seed59317_index96_false_positive_gradcam_pair_v2.png"
)

plt.savefig(
    pair_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.close(fig)

print("\nSaved revised two-panel figure:")
print(pair_path)


# ============================================================
# 13. Save all 17 input channels for visual inspection
# ============================================================

fig, axes = plt.subplots(
    4,
    5,
    figsize=(12, 10)
)

axes = axes.flatten()

for ch in range(17):

    img_ch = (
        x[0, ch]
        .detach()
        .cpu()
        .numpy()
    )

    img_ch = normalize01(img_ch)

    axes[ch].imshow(
        img_ch,
        cmap="gray"
    )

    axes[ch].set_title(
        f"Channel {ch}",
        fontsize=9
    )

    axes[ch].axis("off")

for j in range(17, len(axes)):
    axes[j].axis("off")

plt.tight_layout()

channels_path = os.path.join(
    OUTPUT_DIR,
    "index96_all17_input_channels.png"
)

plt.savefig(
    channels_path,
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. Save provenance metadata
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_metadata.txt"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("Experiment: Exp-2\n")
    f.write("Seed: 59317\n")
    f.write(f"Index: {INDEX}\n")
    f.write(f"True label: {true_label} (Benign)\n")
    f.write(f"Predicted label: {pred_label} (Malignant)\n")
    f.write(f"Benign probability: {p_b:.10f}\n")
    f.write(f"Malignant probability: {p_m:.10f}\n")
    f.write(f"CSV malignant probability: {csv_p_m:.10f}\n")
    f.write(f"Display channel: {DISPLAY_CHANNEL}\n")
    f.write(f"Input shape: {tuple(x.shape)}\n")
    f.write(f"Fused feature shape: {tuple(activation.shape)}\n")
    f.write("Grad-CAM target: malignant logit\n")
    f.write(
        "Grad-CAM feature: fused spatial feature map immediately "
        "before adaptive global average pooling.\n"
    )
    f.write(f"Checkpoint: {CHECKPOINT}\n")
    f.write(f"Model source: {fusion_file}\n")
    f.write(f"Dataset source: {loader_file}\n")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("Saved input:")
print(input_path)

print("\nSaved Grad-CAM:")
print(overlay_path)

print("\nSaved final two-panel figure:")
print(pair_path)

print("\nSaved all-channel preview:")
print(channels_path)

print("\nSaved metadata:")
print(metadata_path)
