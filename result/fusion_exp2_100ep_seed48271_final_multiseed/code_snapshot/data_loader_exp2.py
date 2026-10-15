from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BreastDMNpyDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root) / split
        self.augment = augment

        # Benign = 0, Malignant = 1
        self.class_to_idx = {
            "Benign": 0,
            "Malignant": 1
        }

        self.samples = []

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            npy_files = sorted(class_dir.rglob("*.npy"))

            for npy_path in npy_files:
                self.samples.append((npy_path, label))

        self.targets = [label for _, label in self.samples]

        print(
            f"{split} dataset loaded: "
            f"{len(self.samples)} npy files, "
            f"augment={self.augment}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        npy_path, label = self.samples[index]

        # -------------------------------------------------
        # 1. Load npy
        # Original shape: H × W × 17
        # -------------------------------------------------
        image = np.load(npy_path)

        if image.ndim != 3 or image.shape[-1] != 17:
            raise ValueError(
                f"Invalid image shape: {npy_path} -> {image.shape}"
            )

        # H × W × 17
        # ->
        # 17 × H × W
        image = torch.from_numpy(
            image
        ).permute(2, 0, 1).float()

        # uint8: 0~255 -> 0~1
        image = image / 255.0

        # -------------------------------------------------
        # 2. Resize to paper Exp-2 input: 17 × 96 × 96
        # -------------------------------------------------
        image = F.interpolate(
            image.unsqueeze(0),
            size=(96, 96),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # -------------------------------------------------
        # 3. Data augmentation
        # Only enabled for training set
        # -------------------------------------------------
        if self.augment:

            # ---------- Flipping ----------
            if torch.rand(1).item() < 0.5:
                image = torch.flip(
                    image,
                    dims=[2]
                )

            

            # ---------- Scaling ----------
            # Random scale: 0.9 ~ 1.1
            scale = 0.9 + 0.2 * torch.rand(1).item()
            new_size = int(round(96 * scale))

            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_size, new_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            # Scale > 1: center crop back to 96 × 96
            if new_size > 96:
                start = (new_size - 96) // 2

                image = image[
                    :,
                    start:start + 96,
                    start:start + 96
                ]

            # Scale < 1: pad back to 96 × 96
            elif new_size < 96:
                pad_total = 96 - new_size

                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

                image = F.pad(
                    image,
                    (
                        pad_left,
                        pad_right,
                        pad_left,
                        pad_right
                    )
                )

            # ---------- Clipping / random crop ----------
            # Randomly keep 90% ~ 100% of the image
            crop_ratio = 0.9 + 0.1 * torch.rand(1).item()
            crop_size = int(round(96 * crop_ratio))

            max_offset = 96 - crop_size

            if max_offset > 0:
                top = torch.randint(
                    0,
                    max_offset + 1,
                    (1,)
                ).item()

                left = torch.randint(
                    0,
                    max_offset + 1,
                    (1,)
                ).item()

                image = image[
                    :,
                    top:top + crop_size,
                    left:left + crop_size
                ]

                # Resize crop back to 96 × 96
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(96, 96),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

        # -------------------------------------------------
        # 4. Label
        # -------------------------------------------------
        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label