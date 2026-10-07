# Release assets / 发布附件

The complete package uses one GitHub repository with two storage locations:
source, documentation and small results in the repository; data and model weights
in assets attached to the matching Release. This local candidate has not been published.
整套材料使用同一个GitHub仓库：源码、说明和小型结果放主页；数据及模型放对应Release的附件。当前候选版尚未发布。

GitHub's ordinary file limit is 100 MiB (browser uploads: 25 MiB). Release assets must
each be smaller than 2 GiB. These limits were checked on 2026-10-07:
[file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[Release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
普通Git文件和Release附件的限制不同；本包采用Release附件承载大文件。

## Restore / 恢复完整目录

1. Download the repository at the chosen version / 下载指定版本的仓库。
2. Download every ZIP listed in `ASSETS.json` from that same Release into one local folder.
   下载同一Release中清单列出的全部ZIP附件，放在一个本地文件夹。
3. Run from the repository root / 在仓库根目录执行：

```powershell
python restore_assets.py --asset-dir "D:/downloaded_breastdm_assets"
```

On macOS/Linux, replace the example path with your downloaded-assets directory.
其他系统使用对应下载目录路径。
The script checks SHA-256 and each extracted file, and refuses to overwrite differing files.
脚本检查压缩包及解压文件的SHA-256，拒绝覆盖内容不同的已有文件。
All archive paths already include `data/`, `code/pretrained/`, or `result/`.
Do not create an extra dataset-directory layer when extracting manually.
压缩包内部已包含上述目录，手动解压时不要再套一层文件夹。

To verify without extracting / 只校验不解压：

```powershell
python restore_assets.py --asset-dir "D:/downloaded_breastdm_assets" --verify-only
```

See `README.md` for environment and execution instructions after restoration.
恢复完成后，按`README.md`安装环境并运行。
