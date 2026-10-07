import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes, *, return_ppv_details=False, header="Test:"):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # Opt-in for 2D Test; preserve the legacy four-value API for validation/3D.
    author_ppv_sum = 0.0
    author_included = author_excluded = author_tp_zero = author_tn_zero = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            if return_ppv_details:
                # Independent author seg_binary PPV: per-image filtering and mean.
                # This name identifies the released evaluator, not Table 2 provenance.
                for pred_i, gt_i in zip(output.argmax(1), target):
                    tp_i = int(((pred_i == 1) & (gt_i == 1)).sum().item())
                    tn_i = int(((pred_i == 0) & (gt_i == 0)).sum().item())
                    fp_i = int(((pred_i == 1) & (gt_i == 0)).sum().item())
                    fn_i = int(((pred_i == 0) & (gt_i == 1)).sum().item())
                    author_tp_zero += int(tp_i == 0)
                    author_tn_zero += int(tn_i == 0)
                    if tp_i != 0 and tn_i != 0:
                        author_ppv_sum += tp_i / (tp_i + fp_i)
                        author_included += 1
                    else:
                        author_excluded += 1

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

        # ===== mIoU =====
        # compute() 返回：
        # acc_global, 每类别accuracy, 每类别IoU
        _, _, iu = confmat.compute()

        miou = iu.mean().item()

        # ===== PPV =====
        # confusion matrix:
        # 行 = Ground Truth
        # 列 = Prediction
        # class 0 = background
        # class 1 = tumor
        h = confmat.mat.float()

        tp = h[1, 1]
        fp = h[0, 1]

        if (tp + fp).item() > 0:
            ppv = (tp / (tp + fp)).item()
        else:
            ppv = 0.0

        if return_ppv_details:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                totals = torch.tensor([author_ppv_sum, author_included, author_excluded,
                                       author_tp_zero, author_tn_zero], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(totals)
                author_ppv_sum, author_included, author_excluded, author_tp_zero, author_tn_zero = totals.tolist()
            # The released seg_binary also returns zero when num_evals == 0.
            author_ppv = author_ppv_sum / author_included if author_included != 0 else 0.0
            ppv_details = {
                'PPV_global': ppv,
                'PPV_author_separate_evaluator': author_ppv,
                'author_ppv_included_images': int(author_included),
                'author_ppv_excluded_images': int(author_excluded),
                'author_ppv_tp_zero_images': int(author_tp_zero),
                'author_ppv_tn_zero_images': int(author_tn_zero),
            }
            return confmat, dice.value.item(), miou, ppv, ppv_details
        return confmat, dice.value.item(), miou, ppv


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None
    # print(len(data_loader))
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        if not torch.isfinite(loss):
            print("\n===== NON-FINITE LOSS DETECTED =====")
            print("epoch =", epoch)
            print("loss =", loss)
            print("image finite =", torch.isfinite(image).all().item())
            print("image min =", image.min().item())
            print("image max =", image.max().item())
            print("target unique =", torch.unique(target).detach().cpu().tolist())

            for name, value in output.items():
                print(
                    name,
                    "finite =", torch.isfinite(value).all().item(),
                    "min =", value.min().item(),
                    "max =", value.max().item()
                )

            raise RuntimeError("Non-finite loss detected. Training stopped before backward/optimizer.step.")

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            bad_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    bad_grads.append(name)

            if bad_grads:
                print("\n===== NON-FINITE GRADIENT DETECTED =====")
                print("epoch =", epoch)
                print("bad gradients =", bad_grads)
                raise RuntimeError(
                    "Non-finite gradient detected. Training stopped before optimizer.step."
                )

            optimizer.step()

        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
