from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"split must be 'train', 'val' or 'test', but got: {split}"
            )

        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        self.image_root = self.root / split / "images"
        self.mask_root = self.root / split / "labels"

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image folder not found: {self.image_root}"
            )

        if not self.mask_root.exists():
            raise FileNotFoundError(
                f"Mask folder not found: {self.mask_root}"
            )

        self.images = []
        self.masks = []

        # 按 patient / sequence / image 递归读取
        for image_path in sorted(self.image_root.rglob("*.jpg")):

            relative_path = image_path.relative_to(self.image_root)

            mask_path = (
                self.mask_root
                / relative_path.parent
                / f"{image_path.stem}.png"
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image:\n"
                    f"{image_path}\n"
                    f"Expected mask:\n"
                    f"{mask_path}"
                )

            self.images.append(image_path)
            self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No JPG images found in: {self.image_root}"
            )

        print(
            f"[{split}] loaded: "
            f"{len(self.images)} image-mask pairs"
        )

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        # MRI统一读取为单通道灰度
        image = Image.open(image_path).convert("L")

        # Mask读取为灰度
        mask = Image.open(mask_path).convert("L")

        # Mask严格转换为 0 / 1
        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self):
        return len(self.images)