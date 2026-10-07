from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import time
import datetime
import csv
import random
import numpy as np
import shutil
import sys
import json
import matplotlib.pyplot as plt

import torch

from src import UNet
from train_utils import train_one_epoch, evaluate
from my_dataset import DriveDataset
import transforms as T

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.BreastDMZScore(),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, crop_size,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(crop_size),
            T.ToTensor(),
            T.BreastDMZScore(),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), input_size=224):
    base_size = input_size
    crop_size = input_size

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(crop_size,mean=mean, std=std)


def create_model(aux,num_classes,pretrain=False):
    model = UNet(in_channels=1,num_classes=num_classes,base_c=32)

    if pretrain:
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def save_training_curves(history_file, run_dir):
    epochs = []
    train_loss = []
    val_dice = []
    val_miou = []
    val_ppv = []
    lr_list = []

    with open(history_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_dice.append(float(row["val_dice"]))
            val_miou.append(float(row["val_miou"]))
            val_ppv.append(float(row["val_ppv"]))
            lr_list.append(float(row["lr"]))

    # 1. Training Loss
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # 2. Validation Dice
    plt.figure()
    plt.plot(epochs, val_dice)
    plt.xlabel("Epoch")
    plt.ylabel("DSC")
    plt.title("Validation DSC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dice_curve.png"), dpi=300)
    plt.close()

    # 3. Validation mIoU
    plt.figure()
    plt.plot(epochs, val_miou)
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "miou_curve.png"), dpi=300)
    plt.close()

    # 4. Validation PPV
    plt.figure()
    plt.plot(epochs, val_ppv)
    plt.xlabel("Epoch")
    plt.ylabel("PPV")
    plt.title("Validation PPV Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "ppv_curve.png"), dpi=300)
    plt.close()

    # 5. Learning Rate
    plt.figure()
    plt.plot(epochs, lr_list)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "lr_curve.png"), dpi=300)
    plt.close()

def save_test_predictions(model, test_loader, test_dataset, device, run_dir):
    model.eval()

    predictions_file = os.path.join(run_dir, "test_predictions.csv")
    records = []

    with torch.no_grad():
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)

            output = model(image)["out"]
            pred = output.argmax(dim=1)

            # 忽略 padding 区域 255
            valid = target != 255

            gt_pos = (target == 1) & valid
            pred_pos = (pred == 1) & valid

            tp = (gt_pos & pred_pos).sum().item()
            fp = ((~gt_pos) & pred_pos & valid).sum().item()
            fn = (gt_pos & (~pred_pos) & valid).sum().item()

            dice_den = 2 * tp + fp + fn
            iou_den = tp + fp + fn
            ppv_den = tp + fp

            dice = (2 * tp / dice_den) if dice_den > 0 else 1.0
            iou = (tp / iou_den) if iou_den > 0 else 1.0
            ppv = (tp / ppv_den) if ppv_den > 0 else 0.0

            image_path = test_dataset.images[idx]
            mask_path = test_dataset.masks[idx]

            relative_path = image_path.relative_to(test_dataset.image_root)

            patient = relative_path.parts[0]
            sequence = relative_path.parts[1]
            filename = relative_path.name

            records.append([
                idx,
                str(image_path),
                str(mask_path),
                patient,
                sequence,
                filename,
                dice,
                iou,
                ppv,
                int(gt_pos.sum().item()),
                int(pred_pos.sum().item())
            ])

    with open(predictions_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "index",
            "image_path",
            "mask_path",
            "patient",
            "sequence",
            "filename",
            "dice",
            "iou",
            "ppv",
            "gt_positive_pixels",
            "pred_positive_pixels"
        ])

        writer.writerows(records)

    print("Test predictions saved to:", predictions_file)

    return records

def save_representative_examples(
        model,
        test_dataset,
        test_records,
        device,
        run_dir
):
    model.eval()

    # 按单张Dice排序
    records_sorted = sorted(
        test_records,
        key=lambda x: x[6]   # 第7列就是dice
    )

    failure_record = records_sorted[0]
    success_record = records_sorted[-1]

    cases = [
        ("representative_success.png", success_record, "Successful Segmentation"),
        ("representative_failure.png", failure_record, "Failed Segmentation")
    ]

    for save_name, record, title_name in cases:

        idx = record[0]
        dice = record[6]
        iou = record[7]
        ppv = record[8]

        # 重新取得这一张已经完成224x224预处理的MRI和Mask
        image, target = test_dataset[idx]

        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_input)["out"]
            pred = output.argmax(dim=1)[0].cpu().numpy()

        # MRI用于显示
        image_show = image.squeeze(0).cpu().numpy()

        # z-score图像重新映射到0~1，仅用于可视化
        image_min = image_show.min()
        image_max = image_show.max()

        image_show = (
            (image_show - image_min) /
            (image_max - image_min + 1e-8)
        )

        target_show = target.cpu().numpy().copy()

        # 255是ignore区域，显示时当作背景
        target_show[target_show == 255] = 0

        # ===== 绘图 =====
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # MRI
        axes[0].imshow(image_show, cmap="gray")
        axes[0].set_title("MRI")
        axes[0].axis("off")

        # Ground Truth
        axes[1].imshow(target_show, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Prediction
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        # Overlay
        axes[3].imshow(image_show, cmap="gray")

        axes[3].contour(
            target_show,
            levels=[0.5],
            linewidths=2
        )

        axes[3].contour(
            pred,
            levels=[0.5],
            linewidths=1
        )

        axes[3].set_title("Overlay")
        axes[3].axis("off")

        fig.suptitle(
            f"{title_name}\n"
            f"DSC={dice:.4f} | IoU={iou:.4f} | PPV={ppv:.4f}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(run_dir, save_name),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print("Representative success/failure figures saved.")

def validate_output_directory(run_dir):
    """Read-only guard: never allow a nonempty output directory, including resume."""
    if os.path.exists(run_dir):
        if not os.path.isdir(run_dir) or os.listdir(run_dir):
            raise FileExistsError(
                f"Output directory already exists and is nonempty (or is a file): {run_dir}. "
                "Choose a new output directory; existing results will not be overwritten."
            )


def main(args):

    # 固定随机种子，保证实验可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    requested_device = args.device
    cuda_available = torch.cuda.is_available()
    print("Requested device:", requested_device)
    print("CUDA available:", cuda_available)
    if requested_device == "cuda" and not cuda_available:
        print("Actual device: unavailable")
        print("GPU name: unavailable")
        raise RuntimeError("CUDA was explicitly requested but is unavailable; CPU fallback is disabled.")
    device = torch.device(requested_device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "N/A (CPU)"
    batch_size = args.batch_size
    run_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    validate_output_directory(run_dir)
    args.data_path = os.path.abspath(os.path.expanduser(args.data_path))
    args.output_dir = run_dir
    criterion_text = "Validation DSC strictly increases; ties keep best; no secondary metric; Test excluded"
    configuration = {
        "Data root": args.data_path,
        "Train split": os.path.join(args.data_path, "train"),
        "Validation split": os.path.join(args.data_path, "val"),
        "Test split": os.path.join(args.data_path, "test"),
        "Output directory": run_dir,
        "Seed": args.seed,
        "Epochs": args.epochs,
        "Batch size": args.batch_size,
        "Initial LR": args.lr,
        "Optimizer": "Adam",
        "LR patience": args.lr_patience,
        "LR factor": args.lr_factor,
        "Input size": f"1x{args.input_size}x{args.input_size}",
        "Num workers": args.num_workers,
        "Requested device": requested_device,
        "CUDA available": cuda_available,
        "Actual device": str(device),
        "GPU name": gpu_name,
        "Best checkpoint criterion": criterion_text,
        "Scheduler": "ReduceLROnPlateau(training mean loss)",
        "PyTorch scheduler patience": args.lr_patience - 1,
        "LR threshold": 0.0,
        "Preprocessing": "0.1%-99.9% clipping + z-score",
    }
    for name, value in configuration.items():
        print(f"{name}: {value}")
    # Reconstruct an executable PowerShell invocation; preserve exact argv separately.
    invocation = "& " + " ".join("'" + str(v).replace("'", "''") + "'"
                                  for v in [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])
    os.makedirs(run_dir, exist_ok=True)
    config_file = os.path.join(run_dir, "run_config.txt")
    with open(config_file, "x", encoding="utf-8") as f:
        f.write(f"PowerShell invocation (reconstructed): {invocation}\n")
        f.write("CLI argv: " + json.dumps(sys.argv, ensure_ascii=False) + "\n")
        f.write("Working directory: " + os.getcwd() + "\n")
        f.write("Parsed arguments: " + json.dumps(vars(args), ensure_ascii=False) + "\n")
        for name, value in configuration.items():
            f.write(f"{name}: {value}\n")

    # 保存本次实验所使用的代码快照
    project_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_dir = os.path.join(run_dir,"code_snapshot_resume" if args.resume else "code_snapshot")

    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_files = [
        ("release_paths.py", "release_paths.py"),
        ("src/__init__.py", "src/__init__.py"),
        ("train_utils/__init__.py", "train_utils/__init__.py"),
        ("train.py", "train.py"),
        ("my_dataset.py", "my_dataset.py"),
        ("transforms.py", "transforms.py"),
        (os.path.join("src", "unet.py"), os.path.join("src", "unet.py")),
        (
            os.path.join("train_utils", "train_and_eval.py"),
            os.path.join("train_utils", "train_and_eval.py")
        ),
        (
            os.path.join("train_utils", "distributed_utils.py"),
            os.path.join("train_utils", "distributed_utils.py")
        ),
        (
            os.path.join("train_utils", "dice_coefficient_loss.py"),
            os.path.join("train_utils", "dice_coefficient_loss.py")
        ),
    ]

    for src_rel, dst_rel in snapshot_files:
        src_path = os.path.join(project_dir, src_rel)
        dst_path = os.path.join(snapshot_dir, dst_rel)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    print("Code snapshot saved to:", snapshot_dir)

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = os.path.join(run_dir, "training_log.txt")

    history_file = os.path.join(run_dir, "training_history.csv")

    # 第一次创建CSV时写入表头
    if not os.path.exists(history_file):
        with open(history_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_dice",
                "val_miou",
                "val_ppv",
                "lr"
            ])

    train_dataset = DriveDataset(
        args.data_path,
        split="train",
        transforms=get_transform(train=True, mean=mean, std=std, input_size=args.input_size)
    )

    val_dataset = DriveDataset(
        args.data_path,
        split="val",
        transforms=get_transform(train=False, mean=mean, std=std, input_size=args.input_size)
    )

    test_dataset = DriveDataset(
        args.data_path,
        split="test",
        transforms=get_transform(train=False, mean=mean, std=std, input_size=args.input_size)
    )

    num_workers = args.num_workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)
                                               
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             shuffle = False,
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              pin_memory=True)

    model = create_model(aux=args.aux,num_classes=num_classes)
    model.to(device)

    # params_to_optimize = [
    #     {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
    #     {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    # ]
    #
    # if args.aux:
    #     params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10}) # FCN增加
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 每个epoch结束后，根据训练平均loss更新学习率
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        # PyTorch reduces after num_bad_epochs > patience. CLI counts bad epochs.
        patience=args.lr_patience - 1,
        threshold=0.0
    )

    if args.resume:
        checkpoint = torch.load(args.resume,map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.0

    if args.resume and os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if rows:
            best_dice = max(float(row["val_dice"]) for row in rows)
            print(f"Resume best Dice from history: {best_dice:.4f}")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        print_freq=args.print_freq, scaler=scaler)

        lr_scheduler.step(mean_loss)

        confmat,dice,miou,ppv=evaluate(model,val_loader,device=device,num_classes=num_classes,header="Validation:")
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        print(f"mIoU: {miou:.3f}")
        print(f"PPV: {ppv:.3f}")

        # 保存每个epoch的训练历史
        with open(history_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(mean_loss),
                float(dice),
                float(miou),
                float(ppv),
                float(optimizer.param_groups[0]["lr"])
            ])

        # write into txt
        with open(results_file, mode='a') as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch + 1}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {optimizer.param_groups[0]['lr']:.6f}\n" \
                         f"dice coefficient: {dice:.4f}\n" \
                         f"mIoU: {miou:.4f}\n" \
                         f"PPV: {ppv:.4f}\n"
            f.write(train_info + val_info + "\n\n")

        # 判断当前epoch是否刷新最佳Dice
        is_best = dice > best_dice

        if is_best:
            best_dice = dice

        # 保存完整checkpoint
        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args
        }

        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # 每个epoch都覆盖保存最新checkpoint
        torch.save(
            save_file,
            os.path.join(run_dir, "last_checkpoint.pth")
        )

        # 只有刷新最佳Dice时才覆盖best model
        if is_best:
            torch.save(
                save_file,
                os.path.join(run_dir, "best_model.pth")
            )

    # 全部epoch训练完成后，保存训练曲线
    save_training_curves(history_file, run_dir)
    print("Training curves saved to:", run_dir)

    # ===== 最终Test评估：只使用best model =====
    best_model_path = os.path.join(run_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model not found: {best_model_path}"
        )

    # 重新加载验证集Dice最优的模型
    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(best_checkpoint["model"])

    best_epoch = best_checkpoint["epoch"] + 1

    print("\n===== FINAL TEST =====")
    print(f"Loaded best model from epoch: {best_epoch}")

    # 独立test集只在训练全部结束后评估一次
    test_confmat, test_dice, test_miou, test_ppv, test_ppv_details = evaluate(
        model,
        test_loader,
        device=device,
        num_classes=num_classes,
        return_ppv_details=True,
        header="Test:"
    )

    print(test_confmat)
    print(f"Test DSC:  {test_dice:.4f}")
    print(f"Test mIoU: {test_miou:.4f}")
    print(f"Test PPV:  {test_ppv:.4f}")
    for name, value in test_ppv_details.items():
        print(f"Test {name}: {value}")

    # 保存最终test指标
    test_metrics_file = os.path.join(run_dir, "test_metrics.csv")

    with open(test_metrics_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "best_epoch",
            "test_dice",
            "test_miou",
            "test_ppv",  # Legacy global PPV field retained unchanged.
            *test_ppv_details.keys()
        ])

        writer.writerow([
            best_epoch,
            float(test_dice),
            float(test_miou),
            float(test_ppv),
            *test_ppv_details.values()
        ])

    print("Test metrics saved to:", test_metrics_file)

    # 保存test集中每一张MRI的预测指标
    test_records = save_test_predictions(
        model,
        test_loader,
        test_dataset,
        device,
        run_dir
    )

    # 保存代表性的成功 / 失败分割案例
    save_representative_examples(
        model,
        test_dataset,
        test_records,
        device,
        run_dir
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-root", "--data-path", dest="data_path", required=True,
                        help="dataset root containing train/val/test (data-path is an alias)")
    parser.add_argument("--output-dir", required=True, help="new or empty output directory")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--input-size", required=True, type=int)
    parser.add_argument("--lr-patience", required=True, type=int,
                        help="number of consecutive non-improving training-loss epochs before LR reduction")
    parser.add_argument("--lr-factor", required=True, type=float)

    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss") # fcn增加
    parser.add_argument("--device", required=True, choices=["cuda", "cpu"], help="training device; no CPU fallback")
    parser.add_argument("-b", "--batch-size", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', required=True, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    if args.resume:
        parser.error("This entry requires a fresh output directory; resuming interrupted runs is disabled.")
    if args.start_epoch != 0:
        parser.error("A fresh run requires --start-epoch 0.")
    for name in ["epochs", "batch_size", "input_size", "lr_patience"]:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if not 0 < args.lr < float("inf") or not 0 < args.lr_factor < 1:
        parser.error("lr must be finite and positive; lr-factor must be between 0 and 1")
    if not args.output_dir.strip() or not args.data_path.strip():
        parser.error("data-root and output-dir must not be empty")
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
