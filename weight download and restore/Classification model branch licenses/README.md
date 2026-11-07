# Classification model branch licenses

## Upstream code licenses

The ViT implementation directly matches the WZMIAOMIAO GPL-3.0 implementation at a verified licensed 2022 snapshot, with a depth-7 factory modification. Its header also credits timm (Apache-2.0); that antecedent does not make the whole derivative Apache-only. The SE-ResNet50 branch uses Cadene's pretrainedmodels implementation (BSD-3-Clause). Code licenses and pretrained-weight licensing information are listed separately.

## ViT pretrained weights

File: `jx_vit_base_patch16_224_in21k-e5005f0a.pth`.

The [timm ViT configuration](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) maps the historical filename to `vit_base_patch16_224.orig_in21k`. Its [maintainer model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k) and the [Google original model card](https://huggingface.co/google/vit-base-patch16-224-in21k) identify Apache-2.0. These are provenance and licensing evidence; they do not establish byte identity between the historical file and modern model downloads. The historical experiment weight is retained without replacement.

The modern model version has a modified classification head; its download must not be assumed to be byte-identical to the historical experiment weight.

## SE-ResNet50 pretrained weights

File: `se_resnet50-ce0d4300.pth`.

The [timm SENet configuration](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/senet.py) for `legacy_seresnet50.in1k` directly references this filename and inherits an Apache-2.0 license field. The [Cadene implementation source](https://github.com/Cadene/pretrained-models.pytorch/blob/master/LICENSE.txt) separately uses BSD-3-Clause.

## License texts and attribution

- [Apache-2.0 text](timm_APACHE_2.0.txt)
- [Cadene BSD-3-Clause text](cadene_BSD_3_clause.txt)

Retain applicable copyright, attribution and NOTICE information and identify modifications as required by the respective licenses. See [Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0) and the [timm licensing notes](https://github.com/huggingface/pytorch-image-models#licenses). A model license does not grant redistribution rights to its training dataset.

These licenses apply only to the corresponding upstream components; they do not provide a blanket license for the entire repository or its datasets.

The authoritative file-level source mapping and complete additional GPL/MIT notices are in [THIRD_PARTY_CODE.md](../../THIRD_PARTY_CODE.md). This page separates dependency code from weight information; it does not license BreastDM authors’ implementations, unresolved adaptations, data or experimental outputs. This round did not change or re-audit model weight files.
