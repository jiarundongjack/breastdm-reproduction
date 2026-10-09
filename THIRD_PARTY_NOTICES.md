# Original resources and redistribution notes / 原始资源与再分发说明

This document identifies the original sources of third-party materials and explains the distribution approach used for this reproduction.
本说明列明第三方材料的原始来源，并说明本复现项目采用的发布方式。

## BreastDM data and adapted code / 数据与改编代码

Redistribution terms for the BreastDM data and adapted code have not been clearly established. The public version of this reproduction will therefore direct users to the original resources rather than redistribute third-party materials with unclear permissions.

BreastDM 数据及改编代码的再分发条款尚未明确。因此，本复现的公开版本将引导使用者从原始资源获取相关材料，不重新分发许可不明确的第三方材料。

### Original resources / 原始资源入口

Original code and data access information:

[Original BreastDM repository](https://github.com/smallboy-code/Breast-cancer-dataset)

Please obtain the data through the download links provided in the original repository's README and follow the original providers' access and use requirements.

原始代码及数据获取信息见上述原作者仓库。请通过该仓库 README 中提供的下载链接获取数据，并遵守原提供方的访问和使用要求。

### Reproduction materials / 复现材料

The public version will provide our own preparation and analysis scripts, environment specifications, and execution instructions. Any dependencies on original code or data will be identified, with instructions for obtaining them from their original sources.

公开版本将提供我们自己的准备与分析脚本、环境配置及运行说明。对于原始代码或数据的依赖，将明确列出，并说明如何从原始来源获取。


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
