# Classification model branch licenses / 分类模型分支的上游许可说明

## Upstream code licenses / 上游代码许可

The ViT branch uses code derived from timm (Apache-2.0). The SE-ResNet50 branch uses Cadene's pretrainedmodels implementation (BSD-3-Clause). Code licenses and pretrained-weight licensing information are listed separately.

ViT 分支使用源自 timm 的代码（Apache-2.0）；SE-ResNet50 分支使用 Cadene 的 pretrainedmodels 实现（BSD-3-Clause）。代码许可与预训练权重的许可信息分别说明。

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

These licenses apply only to the corresponding upstream components; they do not provide a blanket license for the entire repository or its datasets.

上述许可仅适用于对应的上游组件，不构成对整个仓库或数据集的统一授权。
