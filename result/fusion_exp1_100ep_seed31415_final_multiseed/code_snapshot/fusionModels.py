from functools import partial

import pretrainedmodels.models as premodels
from torch import nn
import torch.nn.functional as F
from VIT_model import *
import torch
import os


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch,x_otherBranch):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else:  # default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch

        return z

class VisionTransformer_base(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=2,
                 embed_dim=768, depth=7, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_base, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 768] -> [N, 196, 768] -> [N, 768, 196] -> [N, 768, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
        # return x_r
class FusionM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        load_vit=True,
        load_se=True
    ):
        super(FusionM, self).__init__()

        # 以 fusionModels.py 所在位置为基准，避免工作目录变化导致路径错误
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.vit_path = os.path.join(
            base_dir,
            "model",
            "jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        )

        self.se_path = os.path.join(
            base_dir,
            "model",
            "se_resnet50-ce0d4300.pth"
        )

        # -------------------------------------------------
        # 1. 建立并加载 SE-ResNet50 预训练权重
        # -------------------------------------------------
        model_se = premodels.se_resnet50(pretrained=None)

        model_se.layer0.conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        if load_se:
            self._load_local_pretrained(
                model=model_se,
                weight_path=self.se_path,
                model_name="SE-ResNet50"
            )

        # -------------------------------------------------
        # 2. 建立并加载 ViT 预训练权重
        # -------------------------------------------------
        self.vit = VisionTransformer_base(
            img_size=96,
            in_c=9
        )

        if load_vit:
            self._load_local_pretrained(
                model=self.vit,
                weight_path=self.vit_path,
                model_name="ViT",
                ignored_keys=("head.weight", "head.bias")
            )

        # -------------------------------------------------
        # 3. 提取 SE-ResNet50 特征层
        # -------------------------------------------------
        self.layer0 = model_se.layer0
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        self.layer3 = model_se.layer3

        self.Nlblock = NLBlockND(in_channels=512)

        self.fcuup = FCUUp(
            inplanes=768,
            outplanes=512,
            up_stride=2
        )

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    @staticmethod
    def _convert_hf_vit_weights(hf_state, model):
        """把当前HuggingFace格式ViT权重转换为本项目ViT格式。"""

        converted = {}

        # 1. Patch Embedding和位置编码
        basic_mapping = {
            "embeddings.cls_token":
                "cls_token",

            "embeddings.position_embeddings":
                "pos_embed",

            "embeddings.patch_embeddings.projection.weight":
                "patch_embed.proj.weight",

            "embeddings.patch_embeddings.projection.bias":
                "patch_embed.proj.bias",
        }

        for old_key, new_key in basic_mapping.items():
            if old_key in hf_state:
                converted[new_key] = hf_state[old_key]

        # 2. 只加载当前模型使用的前7层Transformer
        depth = len(model.blocks)

        for i in range(depth):
            old_prefix = f"encoder.layer.{i}"
            new_prefix = f"blocks.{i}"

            layer_mapping = {
                f"{old_prefix}.layernorm_before.weight":
                    f"{new_prefix}.norm1.weight",

                f"{old_prefix}.layernorm_before.bias":
                    f"{new_prefix}.norm1.bias",

                f"{old_prefix}.attention.output.dense.weight":
                    f"{new_prefix}.attn.proj.weight",

                f"{old_prefix}.attention.output.dense.bias":
                    f"{new_prefix}.attn.proj.bias",

                f"{old_prefix}.layernorm_after.weight":
                    f"{new_prefix}.norm2.weight",

                f"{old_prefix}.layernorm_after.bias":
                    f"{new_prefix}.norm2.bias",

                f"{old_prefix}.intermediate.dense.weight":
                    f"{new_prefix}.mlp.fc1.weight",

                f"{old_prefix}.intermediate.dense.bias":
                    f"{new_prefix}.mlp.fc1.bias",

                f"{old_prefix}.output.dense.weight":
                    f"{new_prefix}.mlp.fc2.weight",

                f"{old_prefix}.output.dense.bias":
                    f"{new_prefix}.mlp.fc2.bias",
            }

            for old_key, new_key in layer_mapping.items():
                if old_key in hf_state:
                    converted[new_key] = hf_state[old_key]

            # 合并Q、K、V
            q_weight = hf_state.get(
                f"{old_prefix}.attention.attention.query.weight"
            )
            k_weight = hf_state.get(
                f"{old_prefix}.attention.attention.key.weight"
            )
            v_weight = hf_state.get(
                f"{old_prefix}.attention.attention.value.weight"
            )

            q_bias = hf_state.get(
                f"{old_prefix}.attention.attention.query.bias"
            )
            k_bias = hf_state.get(
                f"{old_prefix}.attention.attention.key.bias"
            )
            v_bias = hf_state.get(
                f"{old_prefix}.attention.attention.value.bias"
            )

            if (
                q_weight is not None
                and k_weight is not None
                and v_weight is not None
            ):
                converted[f"{new_prefix}.attn.qkv.weight"] = torch.cat(
                    [q_weight, k_weight, v_weight],
                    dim=0
                )

            if (
                q_bias is not None
                and k_bias is not None
                and v_bias is not None
            ):
                converted[f"{new_prefix}.attn.qkv.bias"] = torch.cat(
                    [q_bias, k_bias, v_bias],
                    dim=0
                )

        # 3. 最后的LayerNorm
        if "layernorm.weight" in hf_state:
            converted["norm.weight"] = hf_state["layernorm.weight"]

        if "layernorm.bias" in hf_state:
            converted["norm.bias"] = hf_state["layernorm.bias"]

        return converted


    @staticmethod
    def _load_local_pretrained(
        model,
        weight_path,
        model_name,
        ignored_keys=()
    ):
        """加载本地预训练权重。"""

        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"{model_name}预训练权重不存在：{weight_path}"
            )

        checkpoint = torch.load(
            weight_path,
            map_location="cpu"
        )

        # 兼容外层model/state_dict
        if isinstance(checkpoint, dict):
            if (
                "model" in checkpoint
                and isinstance(checkpoint["model"], dict)
            ):
                checkpoint = checkpoint["model"]

            elif (
                "state_dict" in checkpoint
                and isinstance(checkpoint["state_dict"], dict)
            ):
                checkpoint = checkpoint["state_dict"]

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"{model_name}权重文件格式错误"
            )

        # 清除module.前缀
        cleaned_checkpoint = {}

        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_checkpoint[key] = value

        # 检测当前HuggingFace格式ViT
        if (
            model_name == "ViT"
            and "embeddings.cls_token" in cleaned_checkpoint
        ):
            print("Detected HuggingFace ViT checkpoint.")
            print("Converting ViT weight keys...")

            cleaned_checkpoint = (
                FusionM._convert_hf_vit_weights(
                    cleaned_checkpoint,
                    model
                )
            )

        # 去掉不需要的分类头
        for key in ignored_keys:
            cleaned_checkpoint.pop(key, None)

        model_state = model.state_dict()

        # -------------------------------------------------
        # 适配论文 Exp-1：3通道预训练权重 -> 9通道
        # -------------------------------------------------

        if model_name == "SE-ResNet50":
            key = "layer0.conv1.weight"

            if key in cleaned_checkpoint:
                w = cleaned_checkpoint[key]

                if w.shape[1] == 3 and model.state_dict()[key].shape[1] == 9:
                    cleaned_checkpoint[key] = w.repeat(1, 3, 1, 1) / 3.0
                    print("SE-ResNet50 conv1 adapted: 3 channels -> 9 channels")


        if model_name == "ViT":
            key = "patch_embed.proj.weight"

            if key in cleaned_checkpoint:
                w = cleaned_checkpoint[key]

                if w.shape[1] == 3 and model.state_dict()[key].shape[1] == 9:
                    cleaned_checkpoint[key] = w.repeat(1, 3, 1, 1) / 3.0
                    print("ViT patch embedding adapted: 3 channels -> 9 channels")

            # 224×224 的位置编码 14×14
            # -> 96×96 的位置编码 6×6
            if "pos_embed" in cleaned_checkpoint:

                pos_embed = cleaned_checkpoint["pos_embed"]

                if pos_embed.shape[1] == 197:

                    cls_pos = pos_embed[:, :1, :]
                    patch_pos = pos_embed[:, 1:, :]

                    patch_pos = patch_pos.reshape(
                        1, 14, 14, -1
                    ).permute(0, 3, 1, 2)

                    patch_pos = F.interpolate(
                        patch_pos,
                        size=(6, 6),
                        mode="bicubic",
                        align_corners=False
                    )

                    patch_pos = patch_pos.permute(
                        0, 2, 3, 1
                    ).reshape(1, 36, -1)

                    cleaned_checkpoint["pos_embed"] = torch.cat(
                        [cls_pos, patch_pos],
                        dim=1
                    )

                    print("ViT position embedding adapted: 197 tokens -> 37 tokens")

        # 只加载名称和形状都匹配的权重
        matched_weights = {
            key: value
            for key, value in cleaned_checkpoint.items()
            if (
                key in model_state
                and torch.is_tensor(value)
                and value.shape == model_state[key].shape
            )
        }

        if len(matched_weights) == 0:
            raise RuntimeError(
                f"{model_name}没有任何权重成功匹配"
            )

        load_result = model.load_state_dict(
            matched_weights,
            strict=False
        )

        print("======================================")
        print(f"{model_name} pretrained weight check")
        print("Weight file:", weight_path)
        print(
            "Matched parameters:",
            len(matched_weights),
            "/",
            len(model_state)
        )
        print(
            "Missing parameters:",
            len(load_result.missing_keys)
        )
        print(f"{model_name} pretrained weights loaded once.")
        print("======================================")

    def forward(self, x):
        # ViT权重已经在__init__中加载，这里只做前向计算
        vit_x = self.vit(x)
        vit_x = self.fcuup(vit_x, 6, 6)

        # SE-ResNet50分支
        x = self.layer0(x)
        x = self.layer1(x)
        se_x = self.layer2(x)

        # 双向Non-Local融合
        x_path1 = self.Nlblock(se_x, vit_x)
        x_path2 = self.Nlblock(vit_x, se_x)

        out = torch.cat((x_path1, x_path2), dim=1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)

        return out


if __name__ == '__main__':
    a = torch.rand(2,3,224,224)
    model = FusionM(num_classes=2)
    out = model(a)
    print(out.shape)
    # print(a[:,0].shape)