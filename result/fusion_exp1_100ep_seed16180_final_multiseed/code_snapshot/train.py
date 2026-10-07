from __future__ import print_function
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import Models
import time
from torch.utils import model_zoo
from sklearn import metrics
import copy
import csv
from data_loader import BreastDMNpyDataset
import numpy as np
from classification_auc import classification_metrics, classification_roc, METRIC_FIELDS, RESULT_FIELDS
from classification_auc import classification_test_metrics, EXP1_TEST_METRIC_FIELDS as TEST_METRIC_FIELDS, EXP1_TEST_RESULT_FIELDS as TEST_RESULT_FIELDS
from exp1_provenance import validate_args, begin_run, finish_run

def youden(tpr, fpr, thresholds):
    youden_index = tpr - fpr
    index = np.argmax(youden_index)
    optimal_threshold = thresholds[index]
    return index, optimal_threshold

import pandas as pd
import os
from shutil import rmtree, copytree, copyfile
import random
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

import fusionModels
from tools import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--model', type=str, default='resnet50', help='coco.data file path')
parser.add_argument('--gpu', type=str, default='0', help='coco.data file path')
parser.add_argument('--num_class', type=int, default=2, help='coco.data file path')
parser.add_argument('--random_seed', type=int, default=42, help='random seed')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='coco.data file path')
parser.add_argument('--task_name', type=str, default='breast-cancer-dataset', help='coco.data file path')
parser.add_argument(
    '--path',
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "processed_dataset"
    ),
    help='processed JPG dataset path'
)
parser.add_argument('--auto_split', type=str, default='0', help='coco.data file path')


parser.add_argument("--output-dir", required=True, help="Empty isolated final Exp-1 run directory")
parser.add_argument("--preflight-only", action="store_true", help="Check evidence in temporary storage; no training or inference")
arg = parser.parse_args()
validate_args(arg)
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
# Training settings
batch_size = arg.batch_size
epochs = 100
lr = 0.01
momentum = 0.9
no_cuda = False
seed = int(arg.random_seed)
log_interval = 10
l2_decay = 0.01
random_seed = seed
split_train_ratio = arg.split_train_ratio
path = arg.path

source_name = "train"
target_name = "val"
test_name = "test"

# test_name = 'test'

cuda = not no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")
print("Training device:", device)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


def split_data():
    for name in [source_name, target_name]:
        if os.path.exists(os.path.join(path, name)):
            rmtree(os.path.join(path, name))
            os.makedirs(os.path.join(path, name))
        else:
            os.makedirs(os.path.join(path, name))  ##创建train文件夹和validation文件夹

    tmp = os.listdir(path)  ##列出path路径下的所有文件夹

    tmp = [i for i in tmp if i not in [source_name, target_name]]
    for properties in tmp:
        files = os.listdir(os.path.join(path, properties))  ##列出path/properties下的文件 也就是images
        random.seed(random_seed)  ##随机种子
        random.shuffle(files)  ##依据随机种子进行images的打乱
        for file in files[: int(len(files) * split_train_ratio)]:  ##从第1张images到80%的len的images分为trainset
            if not os.path.exists(os.path.join(path, 'train', properties)):
                os.makedirs(os.path.join(path, 'train', properties))
            copyfile(os.path.join(path, properties, file),
                     os.path.join(path, 'train', properties, file)
                     )  ## 将properties中的80%转到train文件夹下
        for file in files[int(len(files) * split_train_ratio):]:
            if not os.path.exists(os.path.join(path, 'val', properties)):
                os.makedirs(os.path.join(path, 'val', properties))
            copyfile(os.path.join(path, properties, file),
                     os.path.join(path, 'val', properties, file)
                     )  ## 将properties中的20%转到val文件夹下
    print('complete data split')


if arg.auto_split == '1':
    split_data()
else:
    pass

num_classes = 2

# =========================================================
# BreastDM official .npy dataset
# structure:
# processed_dataset/train|val|test/Benign|Malignant/patient/*.npy
# =========================================================

train_dataset = BreastDMNpyDataset(
    root=path,
    split="train",augment=True
)

val_dataset = BreastDMNpyDataset(
    root=path,
    split="val",augment=False
)

test_dataset = BreastDMNpyDataset(
    root=path,
    split="test",augment=False
)

# Benign = 0, Malignant = 1
expected_classes = {
    "Benign": 0,
    "Malignant": 1
}

assert train_dataset.class_to_idx == expected_classes
assert val_dataset.class_to_idx == expected_classes
assert test_dataset.class_to_idx == expected_classes





source_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=cuda
)

target_val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=cuda
)

target_test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=cuda
)


len_source_dataset = len(train_dataset)
len_target_dataset = len(val_dataset)
len_test_dataset = len(test_dataset)
len_source_loader = len(source_loader)


# 根据新的官方训练集重新计算 class weights
class_counts = np.bincount(
    np.array(train_dataset.targets),
    minlength=num_classes
)

if np.any(class_counts == 0):
    raise ValueError(
        f"某个类别没有训练样本: {class_counts}"
    )

class_weight_values = (
    len(train_dataset) /
    (num_classes * class_counts)
)

CLASS_WEIGHTS = torch.tensor(
    class_weight_values,
    dtype=torch.float32
)


print("Dataset root:", path)
print("Class mapping:", train_dataset.class_to_idx)

print(
    "Train:",
    len(train_dataset),
    "B/M:",
    class_counts.tolist()
)

print(
    "Val:",
    len(val_dataset),
    "B/M:",
    np.bincount(
        val_dataset.targets,
        minlength=2
    ).tolist()
)

print(
    "Test:",
    len(test_dataset),
    "B/M:",
    np.bincount(
        test_dataset.targets,
        minlength=2
    ).tolist()
)

print(
    "Class weights:",
    CLASS_WEIGHTS.tolist()
)




def save_dict(model):
    dict = model.module.state_dict() if type(
        model) is nn.parallel.DistributedDataParallel else model.state_dict()
    if not os.path.exists('model/{}'.format(arg.task_name)):
        os.makedirs('model/{}'.format(arg.task_name))
    torch.save(dict, 'model/{}/{}.pth'.format(arg.task_name, arg.model))





def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    classes = ['benign', 'malignant']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


def train(epoch, model, optimizer):
    model.train()

    

    # 保留原来的学习率规则
    learning_rate = max(
        lr * (0.1 ** (epoch // 10)),
        1e-5
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    device = next(model.parameters()).device
    class_weights = CLASS_WEIGHTS.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    for data, label in source_loader:
        data = data.float().to(device)
        label = label.long().to(device)

        optimizer.zero_grad()

        pred = model(data)

        loss = F.nll_loss(
            F.log_softmax(pred, dim=1),
            label
        )

        loss.backward()
        optimizer.step()

        batch_size_now = data.size(0)

        running_loss += loss.item() * batch_size_now
        predicted = pred.argmax(dim=1)

        correct += predicted.eq(label).sum().item()
        total += batch_size_now

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total

    print(
        f"Epoch {epoch}: "
        f"Train Loss={train_loss:.4f}, "
        f"Train Accuracy={train_accuracy:.2f}%, "
        f"LR={learning_rate}"
    )

    return train_loss, train_accuracy



def val(model):
    model.eval()

    device = next(model.parameters()).device

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for data, target in target_val_loader:
            data = data.float().to(device)
            target = target.long().to(device)

            output = model(data)

            loss = F.nll_loss(
                F.log_softmax(output, dim=1),
                target,
                reduction="sum"
            )

            total_loss += loss.item()

            probability = F.softmax(output, dim=1)
            predicted = torch.argmax(output, dim=1)
            all_predictions.extend(predicted.cpu().numpy().tolist())

            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            all_labels.extend(
                target.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probability.cpu().numpy().tolist()
            )

    val_loss = total_loss / total
    val_metrics = classification_metrics(all_labels, all_probabilities, all_predictions)
    val_accuracy = 100.0 * val_metrics["Accuracy"]
    val_auc = val_metrics["AUC"]

    print(
        f"Epoch Validation: "
        f"Val Loss={val_loss:.4f}, "
        f"Val Accuracy={val_accuracy:.2f}%, "
        f"Val AUC={val_auc:.4f}"
    )

    return val_loss, val_metrics



def save_training_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始数据，防止以后无法重新画图
    csv_path = os.path.join(
        output_dir,
        "training_history.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Val Loss",
            "Train Accuracy",
            "Val Accuracy",
            "Val AUC",
            "Best Epoch",
            "Best Val AUC"
        ])

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["train_accuracy"][i],
                history["val_accuracy"][i],
                history["AUC"][i],
                history["best_epoch"][i],
                history["best_val_auc"][i]
            ])

    epochs_x = history["epoch"]

    # Train Loss vs Val Loss
    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs_x,
        history["train_loss"],
        marker="o",
        label="Train Loss"
    )
    plt.plot(
        epochs_x,
        history["val_loss"],
        marker="o",
        label="Val Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs. Validation Loss")
    plt.xticks(epochs_x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "loss_curve.png"),
        dpi=300
    )
    plt.close()

    # Train Accuracy vs Val Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(
        epochs_x,
        history["train_accuracy"],
        marker="o",
        label="Train Accuracy"
    )
    plt.plot(
        epochs_x,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train Accuracy vs. Validation Accuracy")
    plt.xticks(epochs_x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "accuracy_curve.png"),
        dpi=300
    )
    plt.close()

def find_best_threshold(model):
    model.eval()
    device = next(model.parameters()).device

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in target_val_loader:
            data = data.float().to(device)
            target = target.long().to(device)

            output = model(data)
            probability = F.softmax(output, dim=1)[:, 1]

            all_labels.extend(
                target.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probability.cpu().numpy().tolist()
            )

    fpr, tpr, thresholds = roc_curve(
        all_labels,
        all_probabilities
    )

    # Youden index: sensitivity + specificity - 1
    valid = (thresholds >= 0.40) & (thresholds <= 0.70)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        best_threshold = 0.5
    else:
        best_index = valid_indices[
            np.argmax((tpr - fpr)[valid_indices])
        ]
        best_threshold = thresholds[best_index]

    print(
        f"Best validation threshold: "
        f"{best_threshold:.4f}"
    )

    return float(best_threshold)

def evaluate_test_set(model, output_dir):
    """
    测试集在训练结束并恢复最佳 flattened Validation AUC 模型后评估。
    """

    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_two_class_probabilities = []

    with torch.no_grad():
        for data, target in target_test_loader:
            data = data.float().to(device)
            target = target.long().to(device)

            output = model(data)

            loss = F.nll_loss(
                F.log_softmax(output, dim=1),
                target,
                reduction="sum"
            )

            total_loss += loss.item()

            probabilities = F.softmax(output, dim=1)
            all_two_class_probabilities.extend(probabilities.cpu().numpy().tolist())
            probability = probabilities[:, 1]
            predicted = torch.argmax(output, dim=1)
            
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            all_labels.extend(
                target.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predicted.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probability.cpu().numpy().tolist()
            )

    if total != 403:
        raise RuntimeError(f"Expected 403 Test samples, got {total}")
    test_loss = total_loss / total
    test_metrics = classification_test_metrics(all_labels, all_two_class_probabilities, all_predictions, include_counts=True)
    test_accuracy = 100.0 * test_metrics["Accuracy"]

    # ROC与AUC
    fpr, tpr, _ = classification_roc(
        all_labels,
        all_two_class_probabilities
    )

    test_auc = test_metrics["AUC"]

    # PR与Average Precision
    precision, recall, _ = precision_recall_curve(
        all_labels,
        all_probabilities
    )

    average_precision = average_precision_score(
        all_labels,
        all_probabilities
    )

    print("\n========== Final Test Results ==========")
    print(f"Test Loss: {test_loss:.4f}")
    for metric in TEST_METRIC_FIELDS:
        print(f"{metric}: {test_metrics[metric]:.6f}")
    print("========================================")

    # ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC={test_auc:.4f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "test_roc_curve.png"),
        dpi=300
    )
    plt.close()

    # PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=(
            f"PR Curve "
            f"(AP={average_precision:.4f})"
        )
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "test_pr_curve.png"),
        dpi=300
    )
    plt.close()

    # 保存每张测试图片对应的预测数据
    prediction_path = os.path.join(
        output_dir,
        "test_predictions_two_class.csv"
    )

    with open(
        prediction_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Index",
            "True Label",
            "Predicted Label",
            "Malignant Probability",
            "Benign Probability"
        ])

        for index, (
            true_label,
            predicted_label,
            probability
        ) in enumerate(
            zip(
                all_labels,
                all_predictions,
                all_probabilities
            )
        ):
            writer.writerow([
                index,
                true_label,
                predicted_label,
                probability,
                all_two_class_probabilities[index][0]
            ])

    metric_path = os.path.join(output_dir, "author_style_metrics.csv")
    with open(metric_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=TEST_RESULT_FIELDS)
        writer.writeheader()
        writer.writerow({"Experiment": "Exp-1", "Seed": seed, **test_metrics})

    return (
        test_loss,
        test_accuracy,
        test_auc,
        average_precision
    )


def test(model, output_dir):
    """Compatibility entry point: use the same formal test evaluator."""
    return evaluate_test_set(model, output_dir)


if __name__ == '__main__':
    evidence = begin_run(arg, (train_dataset, val_dataset, test_dataset))
    if arg.preflight_only:
        raise SystemExit(0)
    if arg.model == 'resnet101':
        model = Models.Resnet101(num_classes=num_classes)
    if arg.model == 'resnext101':
        model = Models.Resnext101(num_classes=num_classes)
    if arg.model == 'densenet201':
        model = Models.Densnet201(num_classes=num_classes)
    if arg.model == 'resnet50':
        model = Models.Resnet50(num_classes=num_classes)
    if arg.model == 'densenet169':
        model = Models.Densenet169(num_classes=num_classes)
    if arg.model == 'vgg16':
        model = Models.vgg16(num_classes=num_classes)
    if arg.model == 'senet101':
        model = Models.Senet101(num_classes=num_classes)
    if arg.model == 'resnet18':
        model = Models.Resnet18(num_classes=num_classes)
    if arg.model == 'mynet':
        model = Models.MyNet(num_classes=num_classes)
    if arg.model == 'senet50':
        model = Models.Senet50(num_classes=num_classes)
    if arg.model == 'resnet152':
        model = Models.Resnet152(num_classes=num_classes)
    if arg.model == 'fusion':
        model = fusionModels.FusionM(
            num_classes=num_classes,
            load_vit=True,
            load_se=True
        )
        model = model.to(device)

    # =========================================================
    # Run21: freeze pretrained backbones
    # 只训练 fusion layers + classifier
    # =========================================================

    

    

    correct = 0
    # model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel,2)
    # model_path = r'./model/resnet50-19c8e357.pth'
    # assert os.path.exists(model_path), "file {} does not exist.".format(model_path)
    # pre_state_dict = torch.load(model_path)
    # new_state_dict = {}
    # for k,v in model.state_dict().items():
    #     if k in pre_state_dict.keys() and k!='conv1.weight' and k not in ['fc.weight','fc.bias']:
    #         new_state_dict[k] = pre_state_dict[k]
    # model.load_state_dict(new_state_dict,False)

    # print(model)



    #model = torch.nn.DataParallel(model, device_ids=list(range(len(arg.gpu.split(',')))))
    #model
    output_dir = arg.output_dir

    # 优化器只建立一次，避免每个Epoch重置momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=l2_decay
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "AUC": [],
        "best_epoch": [],
        "best_val_auc": []
    }

    best_val_auc = -1.0
    best_epoch = 0
    best_model_state = None

    # 严格训练100个Epoch
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(
            epoch,
            model,
            optimizer
        )

        val_loss, val_metrics = val(model)
        val_accuracy = 100.0 * val_metrics["Accuracy"]
        val_auc = val_metrics["AUC"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["AUC"].append(val_auc)

        # Reproduction rule: released AUC branch, not a manuscript-specified criterion.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_model_state,
                os.path.join(
                    output_dir,
                    "best_fusion_model.pth"
                )
            )

            print(
                f"Best model updated at Epoch {epoch}, "
                f"Val AUC={val_auc:.4f}"
            )

        history["best_epoch"].append(best_epoch)
        history["best_val_auc"].append(best_val_auc)

    # Save epoch history and training curves
    save_training_history(
        history,
        output_dir
    )

    # 恢复验证集表现最好的模型
    if best_model_state is None:
        raise RuntimeError(
            "没有成功保存最佳模型"
        )

    model.load_state_dict(torch.load(
        os.path.join(output_dir, "best_fusion_model.pth"), map_location=device, weights_only=True
    ), strict=True)

    print(
        f"\nUsing best model from Epoch {best_epoch}, "
        f"Best Val AUC={best_val_auc:.4f}"
    )

    # 使用 logits argmax 评估 test set
    evaluate_test_set(
        model,
        output_dir
    )

    print("\nAll results saved to:")
    print(os.path.abspath(output_dir))
    finish_run(arg, (train_dataset, val_dataset, test_dataset), evidence)
