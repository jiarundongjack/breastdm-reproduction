# Third-party sources and licensing / 第三方来源与许可说明

Evidence reviewed on 2026-10-07. This document records source information; it does not grant rights on behalf of third parties.
本说明记录2026-10-07核查到的来源信息，不替第三方授予许可。

## BreastDM data and adapted code / 数据与改编代码

Source: [Original BreastDM repository](https://github.com/smallboy-code/Breast-cancer-dataset).

The reviewed repository README, file tree and public issue discussions did not establish explicit permission to redistribute the BreastDM datasets or a repository-wide license covering all adapted code. Licenses found in individual subprojects apply only within their respective scope. Public download availability alone does not establish redistribution permission. Separate written permission may exist; the original data archive's internal terms and the full paper were not completely verified.

已检查的README、文件树及公开问题回复未能确认BreastDM数据的再分发许可，也未找到覆盖全部改编代码的全仓库许可。个别子项目的许可证仅在各自范围内适用。未找到许可不等于作者禁止；可公开下载也不等于已经获得重新上传的授权。原始数据压缩包内部条款及论文全文尚未完整核验。

The reproduction repository is currently private pending clarification of public redistribution permissions for the data and adapted code. Historical source snapshots in `result/*/code_snapshot` retain the same upstream provenance considerations.

当前复现仓库保持私有，数据及改编代码的公开再分发范围待确认；历史代码快照也涉及相同的上游来源。

## ViT pretrained weights / ViT预训练权重

File: `jx_vit_base_patch16_224_in21k-e5005f0a.pth`.

The [timm ViT configuration](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) maps the historical filename to `vit_base_patch16_224.orig_in21k`. Its [maintainer model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k) and the [Google original model card](https://huggingface.co/google/vit-base-patch16-224-in21k) identify Apache-2.0. These are provenance and licensing evidence; they do not establish byte identity between the historical file and modern model downloads. The historical experiment weight is retained without replacement.

已找到该模型的Apache-2.0标注及旧文件名对应证据。现代版本对分类头有调整，因此不能声称新旧下载文件逐字节相同；实验使用的旧权重不作替换。

## SE-ResNet50 pretrained weights / SE-ResNet50预训练权重

File: `se_resnet50-ce0d4300.pth`.

The [timm SENet configuration](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/senet.py) for `legacy_seresnet50.in1k` directly references this filename and inherits an Apache-2.0 license field. The [Cadene implementation source](https://github.com/Cadene/pretrained-models.pytorch/blob/master/LICENSE.txt) separately uses BSD-3-Clause.

timm中对应模型的配置直接引用该权重文件，并标注Apache-2.0；Cadene实现源码另按BSD-3-Clause许可。两者不可混为同一许可范围。

## License texts and attribution / 许可文本与署名

- [Apache-2.0 text](third_party/licenses/timm_APACHE_2.0.txt)
- [Cadene BSD-3-Clause text](third_party/licenses/cadene_BSD_3_clause.txt)

Retain applicable copyright, attribution and NOTICE information and identify modifications as required by the respective licenses. See [Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0) and the [timm licensing notes](https://github.com/huggingface/pytorch-image-models#licenses). A model license does not grant redistribution rights to its training dataset.

应按各自许可保留适用版权、署名与NOTICE，并标明修改。模型许可不等于其训练数据的再分发许可。

No blanket MIT or Apache license is assigned to the entire reproduction package. The pretrained-model evidence above does not resolve the separate BreastDM data and adapted-code permissions.

本复现包未统一套用MIT或Apache许可。两份预训练模型的上述许可证据，不替代BreastDM数据和改编代码的独立授权。
