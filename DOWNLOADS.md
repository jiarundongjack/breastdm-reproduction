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

## Sources and redistribution / 来源与再分发

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for original sources and redistribution notes.

原始来源及再分发说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
