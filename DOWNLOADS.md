# Resource access / 资源获取

## Original code and data / 原始代码与数据

The original BreastDM code and data access information are provided in the authors' repository:

[Original BreastDM repository](https://github.com/smallboy-code/Breast-cancer-dataset)

BreastDM 原始代码及数据获取信息由原作者仓库提供，请访问上述链接。

Please obtain the data through the Google Drive links in the original repository's README. Follow the original providers' access and use requirements. If a link is unavailable or access is restricted, contact the original providers.

请通过原作者仓库 README 中提供的 Google Drive 链接获取数据，并遵守原提供方的访问和使用要求。如链接失效或访问受限，请联系原提供方。

## Reproduction environment and instructions / 复现环境与说明

Environment specifications are provided in [environment/](environment/). Reproduction commands and task descriptions are provided in [README.md](README.md).

环境配置见 [environment/](environment/)，复现命令与任务说明见 [README.md](README.md)。

The experiments use task-specific prepared inputs. Files downloaded from the original source should not be assumed to match the prepared directory structure automatically.

本研究的实验使用按任务整理的输入数据。从原始入口下载的文件不一定直接对应本研究使用的预处理目录结构。

## Archive restoration / 归档恢复

`restore_assets.py` is a restoration utility for the reproduction-specific ZIP archives listed in `ASSETS.json`. It does not download or prepare the original BreastDM data and should not be run on the original authors' downloaded archives.

`restore_assets.py` 用于恢复 `ASSETS.json` 中列出的本复现专用 ZIP 归档。它不负责下载或预处理原始 BreastDM 数据，不应直接用于原作者提供的下载包。

The model archives listed in [ASSETS.json](ASSETS.json) are available from the [rc1 release](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1). `FILES.json.gz` records checksums for the 12 model files inside those three archives; it is not a source-code inventory or a dataset manifest.

[ASSETS.json](ASSETS.json) 所列模型压缩包见 [rc1 发布页](https://github.com/jiarundongjack/breastdm-reproduction/releases/tag/v1.0.0-rc1)。`FILES.json.gz` 记录这三个压缩包内 12 个模型文件的校验信息，不是源码或数据集清单。

Download all three listed ZIP files into one local directory, then run from the repository root:

将清单中的三个 ZIP 文件下载到同一本地目录，在仓库根目录执行：

```powershell
python restore_assets.py --asset-dir "D:/downloaded_model_assets" --verify-only
python restore_assets.py --asset-dir "D:/downloaded_model_assets"
```

Replace the example path with your download directory. The first command verifies the archives and member checksums; the second restores the model files. Saved-prediction statistics do not require these model archives; see [ANALYSIS.md](ANALYSIS.md).

将示例路径替换为实际下载目录。第一条命令校验压缩包及内部文件，第二条恢复模型文件。仅凭预测记录重算统计不需要这些模型压缩包，见 [ANALYSIS.md](ANALYSIS.md)。

## Sources and redistribution / 来源与再分发

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for original sources and redistribution notes.

原始来源及再分发说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
