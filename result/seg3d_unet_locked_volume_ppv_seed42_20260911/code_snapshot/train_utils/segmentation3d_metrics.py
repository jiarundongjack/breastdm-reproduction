"""Locked reproduction rules; not a claim about the unreleased author 3D evaluator."""
import warnings
import torch
from . import distributed_utils as utils
from .dice_coefficient_loss import dice_coeff

METRIC_PROTOCOL = {
    "prediction": "two-class logits argmax; background=0, tumor=1",
    "DSC": "tumor-only per-volume Dice, arithmetic mean over all volumes",
    "DSC_epsilon": 1e-6,
    "DSC_empty_both": 1.0,
    "mIoU": "global voxel confusion matrix, mean of background and tumor IoU",
    "missing_class": "0/0 remains NaN; direct class mean, no filtering or epsilon",
    "PPV": "PPV_volume_mean; all volumes included; empty prediction contributes zero",
    "PPV_global_aux": "global voxel TP/(TP+FP); zero if denominator is zero",
    "ignore_label": 255,
    "padding": "GT=0 padding voxels participate; no volume filtering",
    "best_checkpoint": "strictly increasing Validation per-volume mean DSC; no tie-break",
    "lr_scheduler": "ReduceLROnPlateau(training loss), patience=9: reduction on 10th bad epoch",
    "depth": "8; central crop if longer, symmetric zero padding if shorter",
}


class VolumeMetrics:
    def __init__(self):
        self.confmat = utils.ConfusionMatrix(2)
        self.dice_sum = 0.0
        self.ppv_sum = 0.0
        self.num_volumes = 0
        self.zero_predictions = 0

    def update(self, logits, target):
        if logits.ndim != 5 or logits.shape[1] != 2 or target.shape != logits[:, 0].shape:
            raise ValueError("Expected logits [N,2,D,H,W] and target [N,D,H,W]")
        pred = logits.argmax(dim=1)
        self.confmat.update(target.flatten(), pred.flatten())
        for p, g in zip(pred, target):
            valid = g != 255
            if not bool(((g == 0) | (g == 1) | (g == 255)).all()):
                raise ValueError("3D target must contain only 0, 1, or ignore label 255")
            # Existing Dice primitive, applied once per volume: independent of batch grouping.
            self.dice_sum += dice_coeff(p.unsqueeze(0).float(), g.unsqueeze(0).float(),
                                        ignore_index=255, epsilon=1e-6).item()
            tp = int(((p == 1) & (g == 1) & valid).sum().item())
            fp = int(((p == 1) & (g == 0) & valid).sum().item())
            self.ppv_sum += tp / (tp + fp) if tp + fp > 0 else 0.0
            self.zero_predictions += int(tp + fp == 0)
            self.num_volumes += 1

    def compute(self):
        if self.num_volumes == 0:
            raise ValueError("Cannot evaluate an empty volume set")
        self.confmat.reduce_from_all_processes()
        totals = torch.tensor([self.dice_sum, self.ppv_sum, self.num_volumes,
                               self.zero_predictions], dtype=torch.float64,
                              device=self.confmat.mat.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        dice_sum, ppv_sum, count, zero = totals.tolist()
        _, _, iu = self.confmat.compute()
        if not bool(torch.isfinite(iu).all()):
            warnings.warn("Undefined class IoU retained: mIoU uses direct mean without filtering.")
        h = self.confmat.mat.float()
        tp, fp = h[1, 1], h[0, 1]
        ppv_global = (tp / (tp + fp)).item() if (tp + fp).item() > 0 else 0.0
        details = {
            "PPV": ppv_sum / count,
            "PPV_volume_mean": ppv_sum / count,
            "PPV_global_aux": ppv_global,
            "ppv_num_volumes": int(count),
            "ppv_zero_prediction_volumes": int(zero),
        }
        return self.confmat, dice_sum / count, iu.mean().item(), details["PPV"], details


def evaluate(model, data_loader, device, num_classes, *, header="Test:"):
    if num_classes != 2:
        raise ValueError("Locked 3D protocol requires two classes")
    model.eval()
    metrics = VolumeMetrics()
    logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            metrics.update(model(image)["out"], target)
    return metrics.compute()
