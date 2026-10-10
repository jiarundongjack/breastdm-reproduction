# Segmentation data preparation / 分割数据整理

These scripts implement the existing patient-directory selection and copying steps used in this reproduction. They accept explicit source and output paths and use only the Python standard library. Python 3.11 is the version used for this project.

这两个脚本实现本复现使用的患者目录筛选和复制步骤，可通过参数指定输入、输出路径，仅使用 Python 标准库。本项目使用 Python 3.11。

## Inputs / 输入

Obtain original code and data access information from the [original authors' repository](https://github.com/smallboy-code/Breast-cancer-dataset). Follow the original providers' access and use requirements.

原始代码及数据获取信息见[原作者仓库](https://github.com/smallboy-code/Breast-cancer-dataset)，请遵守原提供方的访问和使用要求。

The scripts require **existing split segmentation directories** with this structure. Select the appropriate 2D or 3D input directory as `--source`:

脚本需要**已划分的分割数据目录**，结构如下。通过 `--source` 指定相应的二维或三维输入目录：

```text
source/
  train/
    images/<patient_directory>/...
    labels/<patient_directory>/...
  val/
    images/<patient_directory>/...
    labels/<patient_directory>/...
  test/
    images/<patient_directory>/...
    labels/<patient_directory>/...
```

The study's local input folders were named `seg` (2D) and `seg3D` (3D). These names describe the inputs used in the study; they do not establish that every upstream download has this layout. Confirm the contents after extraction. If your download does not contain these split directories, these scripts alone cannot prepare it for training.

本研究使用的本地输入目录名为 `seg`（二维）和 `seg3D`（三维）。这些名称描述本研究的实际输入，不代表原始下载包一定采用相同结构。请在解压后核对内容；如果下载物不包含上述已划分目录，仅凭这两个脚本无法完成训练数据准备。

The scripts do not convert DICOM, generate volumes, resample images, create masks, generate classification channels, or create new train/validation/test splits. They check patient-directory pairing, not voxel alignment or file-level image/mask correspondence.

脚本不负责 DICOM 转换、体积生成、图像重采样、掩膜生成、分类通道构建或重新划分训练/验证/测试集。检查范围是患者目录配对，不包含体素对齐或逐文件图像与掩膜对应关系。

## Selection rules / 筛选规则

| Script / 脚本 | Rule / 规则 |
|---|---|
| `make_seg2d_clean.py` | Copy paired patient directories; exclude `BreaDM-Be-1801`, `BreaDM-Be-1803` and `BreaDM-Be-1804` from **test only**, following the study's existing duplicate-patient exclusion list. / 复制配对的患者目录，按本研究既有重复病例排除清单，仅从**测试集**排除上述三个患者。 |
| `make_seg3d_clean.py` | Select patients present in each split's `images` directory and copy their corresponding `labels`; label-only patient directories are not copied. / 以各划分 `images` 中的患者目录为准，复制对应 `labels`；仅存在于标签端的患者目录不复制。 |

The 2D exclusion list is fixed for this study, not a general duplicate-detection algorithm. Both scripts preserve file contents, patient-directory names and the existing splits.

二维排除清单是本研究的固定规则，不是通用重复检测算法。两个脚本均保留文件内容、患者目录名称及原有划分。

## Commands / 运行命令

Run from the reproduction repository root. Replace `D:/downloaded_BreastDM/seg` and `D:/downloaded_BreastDM/seg3D` with your actual input directories. Paths containing spaces must be quoted.

在复现仓库根目录执行。将示例中的两个输入路径替换为实际目录；包含空格的路径需加引号。

First check inputs without copying / 先检查输入，不复制文件：

```powershell
python code/preparation/make_seg2d_clean.py --source "D:/downloaded_BreastDM/seg" --output "data/segmentation_2d/seg2D_clean_v2" --dry-run
python code/preparation/make_seg3d_clean.py --source "D:/downloaded_BreastDM/seg3D" --output "data/segmentation_3d/seg3D_clean" --dry-run
```

Then copy the selected data / 检查通过后复制所选数据：

```powershell
python code/preparation/make_seg2d_clean.py --source "D:/downloaded_BreastDM/seg" --output "data/segmentation_2d/seg2D_clean_v2"
python code/preparation/make_seg3d_clean.py --source "D:/downloaded_BreastDM/seg3D" --output "data/segmentation_3d/seg3D_clean"
```

Use these standalone commands directly; these scripts are not tasks registered with `code/run.py`. The output locations match the segmentation input paths in the main [README](../../README.md). Model dependencies and training requirements are described there separately.

直接使用上述独立命令；这两个脚本不是 `code/run.py` 中注册的任务。输出位置对应[首页 README](../../README.md)中的分割输入路径，模型依赖及训练要求另见该文档。

## Output handling / 输出处理

Source data are read and copied, never edited. The output directory must not already exist and must not be nested inside the source. All required split directories and selected patient pairs are checked before copying starts. If copying is interrupted by a filesystem error, the incomplete destination is left for inspection; rerunning refuses to overwrite it. A successful dry run checks directory structure only, not sufficient disk space or dataset validity.

脚本只读取并复制源数据，不修改源数据。输出目录必须不存在，且不能位于源目录内部。复制前检查所有必需划分目录及所选患者配对。如果复制因文件系统错误中断，不完整的输出保留供检查，再次运行不会覆盖它。试运行通过仅表示目录检查通过，不代表磁盘空间或数据内容已验证。
