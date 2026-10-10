# Classification input preparation / 分类输入整理

This script organizes existing 9-channel and 17-channel `.npy` arrays into the input layout used in this reproduction. It copies retained arrays byte-for-byte, preserving their relative paths and existing train/validation/test assignments. It does not generate channels from DICOM or JPG images, compute subtraction images, normalize intensities, resize arrays, or create new splits.

本脚本将已有的 9 通道、17 通道 `.npy` 数组整理为本复现实验的输入目录。保留文件逐字节复制，相对路径及已有训练、验证、测试划分不变。它不从 DICOM 或 JPG 生成通道，不计算减影，不归一化、不缩放数组，也不重新划分数据。

## Inputs / 输入

For original resource access, see [DOWNLOADS.md](../../../pretrained%20and%20trained%20weights/DOWNLOADS.md). Select the existing 9-channel and 17-channel directories as `--source9` and `--source17`. The locally inspected directories were named `cls/img9Se` and `cls/img17Se`. Check the downloaded contents: the current upstream download layout has not been verified, and this script requires the structure below.

原始资源入口见 [DOWNLOADS.md](../../../pretrained%20and%20trained%20weights/DOWNLOADS.md)。用 `--source9` 和 `--source17` 指定已有的九通道、十七通道目录。本地核对的目录名为 `cls/img9Se`、`cls/img17Se`。请先检查实际下载内容：尚未核实上游当前下载包结构，脚本要求以下输入结构。

```text
img9Se/ or img17Se/
  train/Benign/<patient>/*.npy
  train/Malignant/<patient>/*.npy
  val/Benign/<patient>/*.npy
  val/Malignant/<patient>/*.npy
  test/Benign/<patient>/*.npy
  test/Malignant/<patient>/*.npy
```

## Study-specific selection / 本研究的选择规则

From the 9-channel input only, omit `.npy` files under `test/Malignant/` for these six patient directories:

仅对九通道输入，跳过 `test/Malignant/` 下这六个患者目录中的 `.npy` 文件：

```text
BreaDM-Ma-1802
BreaDM-Ma-1803
BreaDM-Ma-1804
BreaDM-Ma-1806
BreaDM-Ma-1807
BreaDM-Ma-1808
```

This rule reconstructs the file selection observed in the saved formal inputs. In the locally inspected 9-channel source, these directories contain 43 files (13, 9, 4, 5, 8, 4 respectively). It is not a general duplicate-removal algorithm or a claim about why those patients were excluded historically. The 17-channel input is retained as supplied. An already filtered 9-channel input is also accepted if all checks pass.

该规则重建的是已保存正式输入中的文件选择。本地九通道源目录中，这六个目录分别含 13、9、4、5、8、4 个文件，共 43 个。这不是通用去重算法，也不据此推断历史排除原因。十七通道输入原样保留；已经筛选过的九通道输入通过全部检查后也可使用。

The retained paths must match across both inputs, with these counts per experiment:

两组保留文件的相对路径必须一致，每个实验的文件数如下：

| Split / 划分 | Benign / 良性 | Malignant / 恶性 | Total / 合计 |
|---|---:|---:|---:|
| train | 327 | 875 | 1202 |
| val | 24 | 93 | 117 |
| test | 114 | 289 | 403 |
| Total / 总计 | 465 | 1257 | 1722 |

Each pair must be `uint8`, shaped `H×W×9` and `H×W×17`, with the same spatial dimensions; the final nine channels of the 17-channel array must equal the 9-channel array pixel-for-pixel. These checks match the inspected study inputs; they do not establish full provenance or the original raw-image conversion procedure.

每对数组须为 `uint8`、形状分别为 `H×W×9` 和 `H×W×17`，空间尺寸一致；十七通道数组的后九通道须与九通道数组逐像素一致。这些检查对应已核对的研究输入，不构成完整来源验证或原始影像转换流程的确认。

## Commands / 命令

Requires Python and NumPy (available in the classification environment). Run from the repository root; replace the two source paths with your actual directories. First validate without copying:

需要 Python 和 NumPy（分类环境中已包含）。从仓库根目录运行，将两个源路径替换为实际位置。先检查、不复制：

```powershell
python code/preparation/classification/prepare_classification.py --source9 "D:/downloaded/cls/img9Se" --source17 "D:/downloaded/cls/img17Se" --output9 "data/classification_exp1/processed_dataset" --output17 "data/classification_exp2/img17Se" --dry-run
```

After validation, use the same command without `--dry-run` to copy the files:

检查通过后，去掉 `--dry-run`，执行复制：

```powershell
python code/preparation/classification/prepare_classification.py --source9 "D:/downloaded/cls/img9Se" --source17 "D:/downloaded/cls/img17Se" --output9 "data/classification_exp1/processed_dataset" --output17 "data/classification_exp2/img17Se"
```

Both output directories must be new and separate from the sources and from each other. No source files are changed. The script validates all arrays before copying, checks copied file hashes, and writes `preparation_record.json` inside each output. An interrupted copy may leave a partial output; use new output paths for a retry. The existing training loaders perform division by 255 and resizing to 96×96 at load time, so this preparation step must not apply them again.

两个输出目录都必须尚不存在，且与源目录及彼此不重叠、不互相嵌套。源文件不修改。脚本先检查全部数组，再复制并核对文件哈希，在各输出目录写入 `preparation_record.json`。复制中断时可能留下部分输出，重试时请使用新输出路径。已有训练读取器在读取时除以 255 并缩放到 96×96，整理阶段不重复执行这些处理。

This script runs directly, not through `code/run.py`. Only the script and documentation are provided here; obtain the input arrays through the original resources.

本脚本直接运行，不通过 `code/run.py` 调用。此处提供脚本与说明，输入数组通过原始资源获取。
