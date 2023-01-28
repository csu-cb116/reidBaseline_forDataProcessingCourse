"""
ViP Architecture in PyTorch
Copyright 2021 Shuyang Sun
"""
import math

import torch.nn.init as init
from timm.models.registry import register_model
from timm.models.layers import DropPath

from .vip_layers import *


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class PatchEmbed(nn.Module):
    def __init__(self, stride, has_mask=False, in_ch=0, out_ch=0):
        super(PatchEmbed, self).__init__()
        self.to_token = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch)
        self.proj = nn.Linear(in_ch, out_ch, bias=False)
        self.has_mask = has_mask

    def process_mask(self, x, mask, H, W):
        if mask is None and self.has_mask:
            mask = x.new_zeros((1, 1, H, W))
        if mask is not None:
            H_mask, W_mask = mask.shape[-2:]
            if H_mask != H or W_mask != W:
                mask = F.interpolate(mask, (H, W), mode='nearest')
        return mask

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, 1, H, W] if exists, else None
        Returns:
            out: [B, out_H * out_W, out_C]
            H, W: output height & width
            mask: [B, 1, out_H, out_W] if exists, else None
        """
        out = self.to_token(x)
        B, C, H, W = out.shape
        mask = self.process_mask(out, mask, H, W)
        out = rearrange(out, "b c h w -> b (h w) c").contiguous()  # 改变维度
        out = self.proj(out)  # 做一个线性变化
        return out, H, W, mask


class Encoder(nn.Module):
    def __init__(self, dim, num_parts=64, num_enc_heads=1, drop_path=0.1, act=nn.GELU, has_ffn=True):
        super(Encoder, self).__init__()
        self.num_heads = num_enc_heads
        self.enc_attn = AnyAttention(dim, num_enc_heads)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path else nn.Identity()
        self.reason = SimpleReasoning(num_parts, dim)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=act) if has_ffn else None

    def forward(self, feats, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            feats: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            qpos: [B, N, 1, C]
            kpos: [B, patch_num * patch_size, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        Returns:
            parts: [B, N, C]
        """
        attn_out = self.enc_attn(q=parts, k=feats, v=feats, qpos=qpos, kpos=kpos, mask=mask)  # 计算注意力，并输出
        parts = parts + self.drop_path(attn_out)  # 更新parts
        parts = self.reason(parts)
        if self.enc_ffn is not None:
            parts = parts + self.drop_path(self.enc_ffn(parts))  # 更新parts,
        return parts


class Decoder(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=7, ffn_exp=3, act=nn.GELU, drop_path=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.attn1 = AnyAttention(dim, num_heads)
        self.attn2 = AnyAttention(dim, num_heads)
        self.rel_pos = FullRelPos(patch_size, patch_size, dim // num_heads)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act, norm_layer=Norm)
        self.ffn2 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act, norm_layer=Norm)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, parts=None, part_kpos=None, mask=None, P=0):
        """
        Args:
            x: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
            P: patch_num
        Returns:
            feat: [B, patch_num, patch_size, C]
        """
        dec_mask = None if mask is None else rearrange(mask.squeeze(1), "b h w -> b (h w) 1 1")
        out = self.attn1(q=x, k=parts, v=parts, kpos=part_kpos, mask=dec_mask)
        out = x + self.drop_path(out)
        out = out + self.drop_path(self.ffn1(out))

        out = rearrange(out, "b (p k) c -> (b p) k c", p=P)
        local_out = self.attn2(q=out, k=out, v=out, mask=mask, rel_pos=self.rel_pos)
        out = out + self.drop_path(local_out)
        out = out + self.drop_path(self.ffn2(out))
        return rearrange(out, "(b p) k c -> b p k c", p=P)


class ViPBlock(nn.Module):
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, patch_size=7, num_heads=1, num_enc_heads=1, num_parts=0):
        super(ViPBlock, self).__init__()
        self.encoder = Encoder(dim, num_parts=num_parts, num_enc_heads=num_enc_heads, drop_path=drop_path)
        self.decoder = Decoder(dim, num_heads=num_heads, patch_size=patch_size, ffn_exp=ffn_exp, drop_path=drop_path)

    def forward(self, x, parts=None, part_qpos=None, part_kpos=None, mask=None):
        """
        Args:
            x: [B, patch_num, patch_size, C]
            parts: [B, N, C]
            part_qpos: [B, N, 1, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        Returns:
            feats: [B, patch_num, patch_size, C]
            parts: [B, N, C]
            part_qpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        """
        P = x.shape[1]
        x = rearrange(x, "b p k c -> b (p k) c")
        parts = self.encoder(x, parts=parts, qpos=part_qpos, mask=mask)
        feats = self.decoder(x, parts=parts, part_kpos=part_kpos, mask=mask, P=P)
        return feats, parts, part_qpos, mask


class Stage(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks, patch_size=7, num_heads=1, num_enc_heads=1, stride=1, num_parts=0,
                 last_np=0, last_enc=False, drop_path=0.1, has_mask=None, ffn_exp=3):
        super(Stage, self).__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path for _ in range(num_blocks)]
        self.patch_size = patch_size
        self.rpn_qpos = nn.Parameter(torch.Tensor(1, num_parts, 1, out_ch // num_enc_heads))
        self.rpn_kpos = nn.Parameter(torch.Tensor(1, num_parts, 1, out_ch // num_heads))

        self.proj = PatchEmbed(stride, has_mask=has_mask, in_ch=in_ch, out_ch=out_ch)
        self.proj_token = nn.Sequential(
            nn.Conv1d(last_np, num_parts, 1, bias=False) if last_np != num_parts else nn.Identity(),
            nn.Linear(in_ch, out_ch),
            Norm(out_ch)
        )
        self.proj_norm = Norm(out_ch)

        blocks = [
            ViPBlock(out_ch,
                     patch_size=patch_size,
                     num_heads=num_heads,
                     num_enc_heads=num_enc_heads,
                     num_parts=num_parts,
                     ffn_exp=ffn_exp,
                     drop_path=drop_path[i])
            for i in range(num_blocks)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.last_enc = Encoder(dim=out_ch,
                                num_enc_heads=num_enc_heads,
                                num_parts=num_parts,
                                drop_path=drop_path[-1],
                                has_ffn=False) if last_enc else None
        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.rpn_qpos, a=math.sqrt(5))
        trunc_normal_(self.rpn_qpos, std=.02)
        init.kaiming_uniform_(self.rpn_kpos, a=math.sqrt(5))
        trunc_normal_(self.rpn_kpos, std=.02)

    def to_patch(self, x, patch_size, H, W, mask=None):
        x = rearrange(x, "b (h w) c -> b h w c", h=H)
        pad_l = pad_t = 0
        pad_r = int(math.ceil(W / patch_size)) * patch_size - W
        pad_b = int(math.ceil(H / patch_size)) * patch_size - H
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # 填充
        if mask is not None:
            mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), value=1)
        x = rearrange(x, "b (sh kh) (sw kw) c -> b (sh sw) (kh kw) c", kh=patch_size, kw=patch_size)
        if mask is not None:
            mask = rearrange(mask, "b c (sh kh) (sw kw) -> b c (kh kw) (sh sw)", kh=patch_size, kw=patch_size)
        return x, mask, H + pad_b, W + pad_r

    def forward(self, x, parts=None, mask=None):
        """
        Args:
            x: [B, C, H, W]
            parts: [B, N, C]
            mask: [B, 1, H, W] if exists, else None
        Returns:
            x: [B, out_C, out_H, out_W]
            parts: [B, out_N, out_C]
            mask: [B, 1, out_H, out_W] if exists else None
        """
        x, H, W, mask = self.proj(x, mask=mask)  # vip_layer调用，下一步特征图将进入patch_embedding
        x = self.proj_norm(x)
        if self.proj_token is not None:
            parts = self.proj_token(parts)

        rpn_qpos, rpn_kpos = self.rpn_qpos, self.rpn_kpos  # 位置编码  随机生成的tensor
        rpn_qpos = rpn_qpos.expand(x.shape[0], -1, -1, -1)  # parts 位置编码
        rpn_kpos = rpn_kpos.expand(x.shape[0], -1, -1, -1)  # whole 位置编码

        ori_H, ori_W = H, W
        x, mask, H, W = self.to_patch(x, self.patch_size, H, W, mask)
        for blk in self.blocks:
            # x: [B, K, P, C]
            x, parts, rpn_qpos, mask = blk(x,
                                           parts=parts,
                                           part_qpos=rpn_qpos,
                                           part_kpos=rpn_kpos,
                                           mask=mask)

        dec_mask = None if mask is None else rearrange(mask.squeeze(1), "b h w -> b 1 1 (h w)")
        if self.last_enc is not None:
            x = rearrange(x, "b p k c -> b (p k) c")
            rpn_out = self.last_enc(x, parts=parts, qpos=rpn_qpos, mask=dec_mask)
            return rpn_out, parts, mask
        else:
            x = rearrange(x, "b (sh sw) (kh kw) c -> b c (sh kh) (sw kw)", kh=self.patch_size, sh=H // self.patch_size)
            x = x[:, :, :ori_H, :ori_W]
            return x, parts, mask


class ViP(nn.Module):
    def __init__(self,
                 in_chans=64,
                 inplanes=64,
                 num_layers=(3, 4, 6, 3),
                 num_chs=(256, 512, 1024, 2048),
                 num_strides=(1, 2, 2, 2),
                 num_classes=1000,
                 num_heads=(1, 1, 1, 1),
                 num_parts=(1, 1, 1, 1),
                 patch_sizes=(1, 1, 1, 1),
                 drop_path=0.1,
                 num_enc_heads=(1, 1, 1, 1),
                 act=nn.GELU,
                 # act=nn.ReLU(inplace=True),
                 ffn_exp=3,
                 no_pos_wd=False,
                 has_last_encoder=False,
                 pretrained=False,
                 **ret_args):
        super(ViP, self).__init__()
        self.depth = len(num_layers)
        self.no_pos_wd = no_pos_wd

        self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(inplanes)
        # self.act = act(inplace=True)
        self.act = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rpn_tokens = nn.Parameter(
            torch.Tensor(1, num_parts[0], inplanes))  # 随机初始化一个tensor,维度为（1, num_part[0], inplanes)
        drop_path_ratios = torch.linspace(0, drop_path, sum(num_layers))
        last_chs = [inplanes, *num_chs[:-1]]
        last_nps = [num_parts[0], *num_parts[:-1]]

        for i, n_l in enumerate(num_layers):
            stage_ratios = [drop_path_ratios[sum(num_layers[:i]) + did] for did in range(n_l)]
            setattr(self,
                    "layer_{}".format(i),
                    Stage(last_chs[i],
                          num_chs[i],
                          n_l,
                          stride=num_strides[i],
                          num_heads=num_heads[i],
                          num_enc_heads=num_enc_heads[i],
                          patch_size=patch_sizes[i],
                          drop_path=stage_ratios,
                          ffn_exp=ffn_exp,
                          num_parts=num_parts[i],
                          last_np=last_nps[i],
                          last_enc=has_last_encoder and i == len(num_layers) - 1)
                    )  # setattr函数，为对象设置属性值，这里就相当于定义了self.layer_i=Stage()

        if has_last_encoder:
            self.last_fc = nn.Linear(num_chs[-1], num_classes)
        else:
            self.last_linear = nn.Conv2d(num_chs[-1], num_chs[-1], kernel_size=1, bias=False)
            self.last_norm = nn.BatchNorm2d(num_chs[-1])
            self.pool2 = nn.AdaptiveAvgPool2d(1)
            self.last_fc = nn.Linear(num_chs[-1], num_classes)

        self.has_last_encoder = has_last_encoder
        self._init_weights(pretrained=pretrained)
        self.gap = self.gap = nn.AdaptiveAvgPool2d(1)

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_pattern = ['rel_pos'] if self.no_pos_wd else []
        no_wd_layers = set()
        for name, param in self.named_parameters():
            for skip_name in skip_pattern:
                if skip_name in name:
                    no_wd_layers.add(name)
        return no_wd_layers

    def _init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location=torch.device("cpu"))
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            return

        init.kaiming_uniform_(self.rpn_tokens, a=math.sqrt(5))
        trunc_normal_(self.rpn_tokens, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not torch.sum(m.weight.data == 0).item() == m.num_features:  # zero gamma
                    m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def load_param(self, model_path):
        self.load_state_dict(remove_fc(torch.load(model_path)), strict=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.pool1(out)

        B, _, H, W = out.shape
        rpn_tokens, mask = self.rpn_tokens.expand(x.shape[0], -1, -1), None  # 将随机初始化的tensor进行扩充 rpn_tokens即为parts
        # for i in range(self.depth):
        #     layer = getattr(self, "layer_{}".format(i))  # getattr函数，获取实例的属性 这里可以理解为 layer = self.layer_i
        #     out, rpn_tokens, mask = layer(out, rpn_tokens, mask=mask)  # 进入vip层
        out1, rpn_tokens1, mask1 = self.layer_0(out, rpn_tokens, mask=mask)
        out2, rpn_tokens2, mask2 = self.layer_1(out1, rpn_tokens1, mask=mask1)
        out3, rpn_tokens3, mask3 = self.layer_2(out2, rpn_tokens2, mask=mask2)
        out4, rpn_tokens4, mask4 = self.layer_3(out3, rpn_tokens3, mask=mask3)

        if self.has_last_encoder:
            out = self.act(out4)
            out = out.mean(1)
        else:
            out = self.last_linear(out4)
            out = self.last_norm(out)
            out = self.act(out)
        return out


@register_model
def vip_mobile(pretrained=False, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(48, 96, 192, 384), patch_sizes=[8, 7, 7, 7], num_heads=[1, 2, 4, 8],
                     num_enc_heads=[1, 2, 4, 8], num_parts=[16, 16, 16, 32], num_layers=[1, 1, 1, 1], ffn_exp=3,
                     has_last_encoder=True, drop_path=0., **cfg)
    return ViP(pretrained=pretrained, **model_cfg)


def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    # for key, value in state_dict.items():
    for key, value in list(state_dict.items()):
        if key.startswith('last_fc.'):
            del state_dict[key]
    return state_dict


@register_model
def vip_tiny(has_last_encoder=True, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(64, 128, 256, 512), patch_sizes=[8, 7, 7, 7], num_heads=[1, 2, 4, 8],
                     num_enc_heads=[1, 2, 4, 8], num_parts=[32, 32, 32, 32], num_layers=[1, 1, 2, 1], ffn_exp=3,
                     has_last_encoder=has_last_encoder, drop_path=0.1, **cfg)
    model = ViP(**model_cfg)
    return model


@register_model
def vip_small(has_last_encoder=True, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(96, 192, 384, 768), patch_sizes=[8, 7, 7, 7], num_heads=[3, 6, 12, 24],
                     num_enc_heads=[1, 3, 6, 12], num_parts=[64, 64, 64, 64], num_layers=[1, 1, 3, 1], ffn_exp=3,
                     has_last_encoder=has_last_encoder, drop_path=0.1, **cfg)
    model = ViP(**model_cfg)
    return model


@register_model
def vip_medium(has_last_encoder=True, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(96, 192, 384, 768), patch_sizes=[8, 7, 7, 7], num_heads=[3, 6, 12, 24],
                     num_enc_heads=[1, 3, 6, 12], num_parts=[64, 64, 64, 128], num_layers=[1, 1, 8, 1], ffn_exp=3,
                     has_last_encoder=has_last_encoder, drop_path=0.2, **cfg)
    model = ViP(**model_cfg)
    return model


@register_model
def vip_base(has_last_encoder=True, **cfg):
    model_cfg = dict(inplanes=64, num_chs=(128, 256, 512, 1024), patch_sizes=[8, 7, 7, 7], num_heads=[4, 8, 16, 32],
                     num_enc_heads=[1, 4, 8, 16], num_parts=[64, 64, 128, 128], num_layers=[1, 1, 8, 1], ffn_exp=3,
                     has_last_encoder=has_last_encoder, drop_path=0.3, **cfg)
    model = ViP(**model_cfg)
    return model
