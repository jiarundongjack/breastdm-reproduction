from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import sys
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path((REPO))
RUN = RUN3
SNAPSHOT = RUN / "code_snapshot"
DATA_ROOT = Path((SEG3_DATA))
CKPT = RUN / "best_model.pth"

# 强制使用最终正式 run 保存下来的代码快照
sys.path.insert(0, str(SNAPSHOT))

from my_dataset3d import DriveDataset
from src.unet3d import UNet
from train3d import get_transform


THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
EPS = 1e-6


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", CKPT)
    print("Data root:", DATA_ROOT)

    # 与正式 Test 完全相同的数据构造
    test_dataset = DriveDataset(
        DATA_ROOT,
        split="test",
        transforms=get_transform(
            train=False,
            input_size=224
        ),
        input_size=224,
        depth=8
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )

    print("Test volumes:", len(test_dataset))

    # 最终正式模型：1 input channel, 2 output classes
    model = UNet(
        in_channels=1,
        num_classes=2,
        base_c=32
    ).to(device)

    checkpoint = torch.load(
        CKPT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    stats = {}

    for t in THRESHOLDS:
        stats[t] = {
            "dsc_sum": 0.0,
            "ppv_sum": 0.0,
            "volumes": 0,
            "zero_pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    with torch.no_grad():
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device).long()

            logits = model(image)["out"]

            # Tumor softmax probability
            tumor_prob = torch.softmax(logits, dim=1)[:, 1]

            valid = target != 255

            for threshold in THRESHOLDS:

                # 使用 > 而不是 >=
                # threshold=0.50 与 two-class argmax 的 tie handling 一致
                pred = (tumor_prob > threshold).long()

                p = pred[valid]
                g = target[valid]

                tp = int(((p == 1) & (g == 1)).sum().item())
                fp = int(((p == 1) & (g == 0)).sum().item())
                fn = int(((p == 0) & (g == 1)).sum().item())
                tn = int(((p == 0) & (g == 0)).sum().item())

                # 与正式 3D DSC 一致：
                # tumor-only, per-volume, epsilon=1e-6
                dsc = (2.0 * tp + EPS) / (
                    2.0 * tp + fp + fn + EPS
                )

                # 与正式3D分割结果的PPV一致：
                # per-volume PPV，空 prediction 记 0
                if tp + fp > 0:
                    ppv = tp / (tp + fp)
                else:
                    ppv = 0.0
                    stats[threshold]["zero_pred"] += 1

                stats[threshold]["dsc_sum"] += dsc
                stats[threshold]["ppv_sum"] += ppv
                stats[threshold]["volumes"] += 1

                stats[threshold]["tp"] += tp
                stats[threshold]["fp"] += fp
                stats[threshold]["fn"] += fn
                stats[threshold]["tn"] += tn

            if (idx + 1) % 20 == 0 or (idx + 1) == len(test_dataset):
                print(
                    f"Processed {idx + 1}/{len(test_dataset)} volumes"
                )

    rows = []

    print("\n========== THRESHOLD SENSITIVITY ==========")

    for threshold in THRESHOLDS:
        s = stats[threshold]

        n = s["volumes"]
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        tn = s["tn"]

        dsc = s["dsc_sum"] / n
        ppv_volume = s["ppv_sum"] / n

        # Global voxel confusion matrix -> background/tumor IoU
        tumor_den = tp + fp + fn
        bg_den = tn + fp + fn

        tumor_iou = (
            tp / tumor_den
            if tumor_den > 0
            else float("nan")
        )

        background_iou = (
            tn / bg_den
            if bg_den > 0
            else float("nan")
        )

        miou = (tumor_iou + background_iou) / 2.0

        gt_tumor = tp + fn
        pred_tumor = tp + fp

        pred_gt_ratio = (
            pred_tumor / gt_tumor
            if gt_tumor > 0
            else float("nan")
        )

        row = {
            "threshold": threshold,
            "DSC": dsc,
            "mIoU": miou,
            "PPV_volume_mean": ppv_volume,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Pred_GT_ratio": pred_gt_ratio,
            "zero_prediction_volumes": s["zero_pred"],
            "num_volumes": n,
        }

        rows.append(row)

        print(
            f"threshold={threshold:.2f} | "
            f"DSC={dsc*100:.2f}% | "
            f"mIoU={miou*100:.2f}% | "
            f"PPV(volume)={ppv_volume*100:.2f}% | "
            f"FP={fp} | "
            f"Pred/GT={pred_gt_ratio:.4f} | "
            f"zero_pred={s['zero_pred']}"
        )

    # ---------------------------------------------------------
    # 验证 threshold=0.50 是否复现正式 argmax 结果
    # ---------------------------------------------------------
    official_file = RUN / "test_metrics.csv"

    with open(official_file, "r", newline="", encoding="utf-8") as f:
        official = next(csv.DictReader(f))

    baseline = rows[0]

    official_dsc = float(official["test_dice"])
    official_miou = float(official["test_miou"])
    official_ppv = float(official["PPV_volume_mean"])

    print("\n========== BASELINE CHECK ==========")
    print(
        "Official argmax: "
        f"DSC={official_dsc:.10f}, "
        f"mIoU={official_miou:.10f}, "
        f"PPV={official_ppv:.10f}"
    )

    print(
        "Threshold 0.50: "
        f"DSC={baseline['DSC']:.10f}, "
        f"mIoU={baseline['mIoU']:.10f}, "
        f"PPV={baseline['PPV_volume_mean']:.10f}"
    )

    max_diff = max(
        abs(baseline["DSC"] - official_dsc),
        abs(baseline["mIoU"] - official_miou),
        abs(baseline["PPV_volume_mean"] - official_ppv),
    )

    print(f"Maximum baseline difference = {max_diff:.12g}")

    if max_diff > 1e-5:
        raise RuntimeError(
            "Threshold=0.50 does not reproduce the formal argmax "
            "baseline. Do NOT interpret the sensitivity results yet."
        )

    print(
        "PASS: threshold=0.50 reproduces the formal argmax baseline."
    )

    # 保存独立分析文件，不覆盖正式结果
    output_csv = OUTPUT / "threshold_sensitivity_volume_ppv.csv"

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved to:")
    print(output_csv)


if __name__ == "__main__":
    main()
