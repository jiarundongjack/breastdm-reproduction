from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
from pathlib import Path
import numpy as np
from PIL import Image

SEG2D = Path((SEG2_DATA))
SEG3D = Path((SEG3_DATA))

split = "train"
patient = "BreaDM-Ma-1904"
sequence = "SUB2"

src_dir = SEG2D / split / "images" / patient / sequence
vol_path = SEG3D / split / "images" / patient / f"{sequence}.npy"

print("=" * 80)
print("RGB INTENSITY DISCREPANCY AUDIT")
print("=" * 80)

volume = np.load(vol_path).astype(np.float32)

files = sorted(
    [p for p in src_dir.iterdir()
     if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
)

print(f"\nPatient   : {patient}")
print(f"Sequence  : {sequence}")
print(f"2D slices : {len(files)}")
print(f"3D shape  : {volume.shape}")

print("\n" + "=" * 80)
print("SLICE-BY-SLICE COMPARISON")
print("=" * 80)

for i, path in enumerate(files):

    img = Image.open(path)

    print(f"\n[{i}] {path.name}")
    print(f"Original mode : {img.mode}")
    print(f"Original size : {img.size}")

    # Inspect RGB channel consistency
    if img.mode == "RGB":
        rgb = np.asarray(img).astype(np.float32)

        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        rg = np.max(np.abs(r - g))
        rb = np.max(np.abs(r - b))
        gb = np.max(np.abs(g - b))

        print(f"Max |R-G| : {rg:.6f}")
        print(f"Max |R-B| : {rb:.6f}")
        print(f"Max |G-B| : {gb:.6f}")

        if rg == 0 and rb == 0 and gb == 0:
            print("RGB channels identical: YES")
        else:
            print("RGB channels identical: NO")

    # Convert exactly as our dataset pipeline does
    gray = np.asarray(
        img.convert("L"),
        dtype=np.float32
    )

    if i < volume.shape[2]:

        slice3d = volume[:, :, i]

        if gray.shape == slice3d.shape:

            diff = np.abs(gray - slice3d)

            print(f"MAE vs 3D slice : {diff.mean():.6f}")
            print(f"Max difference  : {diff.max():.6f}")
            print(
                f"Different pixels : "
                f"{np.count_nonzero(diff)} / {diff.size}"
            )

        else:
            print(
                f"Shape mismatch: 2D={gray.shape}, "
                f"3D={slice3d.shape}"
            )

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)