from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
from pathlib import Path

import torch

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torch.nn.functional as F

def breastdm_zscore_3d(image):
    """
    image: torch.Tensor [1, D, H, W]

    对有效 slice 做：
    0.1% - 99.9% clipping
    + z-score normalization

    全零 padding slice 保持为 0。
    """
    image = image.clone()

    # 找出真正有图像内容的 slice
    valid_slices = torch.count_nonzero(
        image[0], dim=(1, 2)
    ) > 0

    if not torch.any(valid_slices):
        return image

    valid_data = image[0, valid_slices]

    # 论文：bottom 0.1% / top 0.1%
    low = torch.quantile(valid_data, 0.001)
    high = torch.quantile(valid_data, 0.999)

    valid_data = torch.clamp(valid_data, min=low, max=high)

    mean = valid_data.mean()
    std = valid_data.std()

    if std > 0:
        valid_data = (valid_data - mean) / std
    else:
        valid_data = valid_data - mean

    # 只写回真实 slice
    image[0, valid_slices] = valid_data

    # padding slice 从始至终保持 0
    image[0, ~valid_slices] = 0

    return image

def standardize_depth(image, mask, depth=8):
    """Reproduction rule: central slices or symmetric zero padding; HWD arrays."""
    if image.ndim != 3 or image.shape != mask.shape or image.shape[2] == 0:
        raise ValueError("Expected matching, nonempty HWD image and mask")
    count = image.shape[2]
    if count > depth:
        start = (count - depth) // 2
        return image[:, :, start:start + depth], mask[:, :, start:start + depth]
    if count < depth:
        before = (depth - count) // 2
        padding = ((0, 0), (0, 0), (before, depth - count - before))
        return np.pad(image, padding), np.pad(mask, padding)
    return image, mask


class DriveDataset(Dataset):
    def __init__(self, root, split="train", transforms=None, *, input_size, depth):
        self.input_size = input_size
        self.depth = depth
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
        for image_path in sorted(self.image_root.rglob("*.npy")):
            relative_path = image_path.relative_to(self.image_root)

            mask_path = self.mask_root / relative_path

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
                f"No NPY volumes found in: {self.image_root}"
            )

        print(
            f"[{split}] loaded: "
            f"{len(self.images)} image-mask pairs"
        )

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        # 读取作者发布的 3D volume
        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path)

        # image / mask 必须严格同尺寸
        if image.shape != mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch:\n"
                f"{image_path}: {image.shape}\n"
                f"{mask_path}: {mask.shape}"
            )

        image, mask = standardize_depth(image, mask, depth=self.depth)

        # 当前作者数据格式为 [H, W, D]
        # 转换为 PyTorch 3D 格式 [D, H, W]
        image = np.transpose(image, (2, 0, 1)).copy()
        mask = np.transpose(mask, (2, 0, 1)).copy()

        # mask: 0/255 -> 0/1
        mask = (mask > 0).astype(np.int64)

        # image: [1, D, H, W]
        # mask : [D, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)

        # 只调整空间尺寸 H, W：369×369 -> 224×224
        # image 当前 [1, D, H, W]，把 D 当作独立通道，因此不会混合不同 slice
        image = F.interpolate(
            image,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False
        )

        # mask 使用 nearest，避免产生非 0/1 标签
        mask = F.interpolate(
            mask.unsqueeze(0).float(),
            size=(self.input_size, self.input_size),
            mode="nearest"
        ).squeeze(0).long()

        image = breastdm_zscore_3d(image)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self):
        return len(self.images)
