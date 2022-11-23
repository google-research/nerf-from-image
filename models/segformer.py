# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils


class SegDropPath(nn.Module):

    def __init__(self, p: float = 0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        keep_p = 1 - self.p
        r = x.new_empty([x.shape[0]] + [1] * (x.ndim - 1)).bernoulli_(keep_p)
        if keep_p > 0:
            r.div_(keep_p)
        return x * r


class SegDWConv(nn.Module):

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, height, width):
        x = x.transpose(1, 2).view(x.shape[0], x.shape[2], height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class SegMLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegDWConv(hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class SegAttention(nn.Module):

    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()

        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(dim // num_heads)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        bs, n, c = x.shape
        q = self.q(x).reshape(bs, n, self.num_heads,
                              c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(bs, c, height, width)
            x_ = self.sr(x_).reshape(bs, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(bs, -1, 2, self.num_heads,
                                     c // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(bs, -1, 2, self.num_heads,
                                    c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(bs, n, c)
        x = self.proj(x)

        return x


class SegBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4, drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SegAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.drop_path = SegDropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SegMLP(in_features=dim, hidden_features=dim * mlp_ratio)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))

        return x


class SegOverlapPatchEmbed(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768):

        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.height = img_size[0] // patch_size[0]
        self.width = img_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class SegLinearMLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class Segformer(nn.Module):

    def __init__(
        self,
        img_size=128,
        in_channels=3,
        out_features=512,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.1,
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=768,
        init_weights=True,
    ):
        super().__init__()

        for i in range(len(embed_dims)):
            setattr(
                self, f'patch_embed{i+1}',
                SegOverlapPatchEmbed(
                    img_size=img_size // (1 if i == 0 else (2**(i + 1))),
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i]))

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(len(embed_dims)):
            setattr(
                self, f'block{i+1}',
                nn.ModuleList([
                    SegBlock(dim=embed_dims[i],
                             num_heads=num_heads[i],
                             mlp_ratio=mlp_ratios[i],
                             drop_path=dpr[cur + j],
                             sr_ratio=sr_ratios[i]) for j in range(depths[i])
                ]))
            setattr(self, f'norm{i+1}', nn.LayerNorm(embed_dims[i], eps=1e-6))
            cur += depths[i]

        for i in reversed(range(4)):
            setattr(
                self, f'linear_c{i+1}',
                SegLinearMLP(input_dim=embed_dims[i], embed_dim=decoder_dim))
        self.linear_fuse = nn.Conv2d(4 * decoder_dim,
                                     decoder_dim,
                                     kernel_size=1)
        self.linear_pred = nn.Conv2d(decoder_dim, out_features, kernel_size=1)

        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = (m.kernel_size[0] * m.kernel_size[1] *
                       m.out_channels) // m.groups
            m.weight.data.normal_(0, math.sqrt(2. / fan_out))
            if m.bias is not None:
                m.bias.data[:] = 0
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data[:] = 0
        elif isinstance(m, nn.LayerNorm):
            m.bias.data[:] = 0
            m.weight.data[:] = 1

    def forward(self, x):
        bs = x.shape[0]
        features = []
        for i in range(4):
            x, height, width = getattr(self, f'patch_embed{i+1}')(x)
            for blk in getattr(self, f'block{i+1}'):
                x = blk(x, height, width)
            x = getattr(self, f'norm{i+1}')(x)
            x = x.reshape(bs, height, width, -1).permute(0, 3, 1,
                                                         2).contiguous()
            features.append(x)

        all_c = []
        for i in reversed(range(4)):
            c = features[i]
            c = getattr(self, f'linear_c{i+1}')(c).permute(0, 2, 1).reshape(
                bs, -1, c.shape[2], c.shape[3])
            if i > 0:
                c = F.interpolate(c,
                                  size=features[0].shape[2:],
                                  mode='bilinear',
                                  align_corners=False)
            all_c.append(c)

        x = self.linear_fuse(torch.cat(all_c, dim=1))
        x = self.linear_pred(x)
        x = F.interpolate(x,
                          size=features[0].shape[2:],
                          mode='bilinear',
                          align_corners=False)
        return x.float()


def init_segformer(out_features,
                   in_channels=3,
                   pretrained=True,
                   pretrained_model_path=None):

    segformer = Segformer(out_features=out_features,
                          in_channels=in_channels,
                          init_weights=pretrained)
    if pretrained:
        assert pretrained_model_path
        pretrained_filename = 'mit_b5.pth'
        if os.path.basename(pretrained_model_path) != pretrained_filename:
            # Try to look up file in the specified directory
            pretrained_model_path = os.path.join(pretrained_model_path,
                                                 pretrained_filename)
        if not utils.file_exists(pretrained_model_path):
            raise FileNotFoundError(
                'Attempting to load SegFormer pretrained model from '
                f'{pretrained_model_path}, '
                'but it was not found. Please download it from the '
                'official repository and copy it manually.')
        with utils.open_file(pretrained_model_path, 'rb') as f:
            mit_pretrained = torch.load(f, map_location='cpu')
        segformer_state_dict = segformer.state_dict()
        for k, v in mit_pretrained.items():
            if k in segformer_state_dict:
                segformer_state_dict[k].data[:] = v.to(
                    segformer_state_dict[k].device)
    total_params = 0
    for param in segformer.parameters():
        total_params += param.numel()
    print(f'Initialized SegFormer B5 backbone ({total_params} parameters)')
    return segformer
