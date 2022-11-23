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

import torch
import torch.nn.functional as F
from torch import nn


class EfficientResample(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, h, padding, stride, transpose):
        if transpose:
            y = F.conv_transpose2d(x, h, padding=padding, stride=stride)
        else:
            y = F.conv2d(x, h, padding=padding, stride=stride)

        ctx.save_for_backward(h)
        ctx.x_shape = x.shape
        ctx.padding = padding
        ctx.stride = stride
        ctx.transpose = transpose
        return y

    @staticmethod
    def backward(ctx, dy):
        assert not ctx.needs_input_grad[1]
        h, = ctx.saved_tensors

        dx = None
        if ctx.needs_input_grad[0]:
            dx = EfficientResample.apply(dy, h, ctx.padding, ctx.stride,
                                         not ctx.transpose)

        return dx, None, None, None, None


def bilinear_filter():
    h = torch.FloatTensor([1, 3, 3, 1])
    h = h[:, None] * h[None, :]
    h /= h.sum()
    return h


def filter2d(im, kernel, gain=1, transpose=False):
    bs, nc = im.shape[:2]
    if gain != 1:
        kernel = kernel * gain
    if transpose:
        output = EfficientResample.apply(
            im.flatten(0, 1).unsqueeze(1), kernel[None, None, ...], 1, 1, True)
    else:
        output = EfficientResample.apply(
            im.flatten(0, 1).unsqueeze(1), kernel[None, None, ...], 1, 1, False)

    return output.view(bs, nc, output.shape[2], output.shape[3])


def upsample2d(im, kernel):
    bs, nc = im.shape[:2]
    output = EfficientResample.apply(
        im.flatten(0, 1).unsqueeze(1), kernel[None, None, ...] * 4, 1, 2, True)
    return output.view(bs, nc, output.shape[2], output.shape[3])


def downsample2d(im, kernel):
    bs, nc = im.shape[:2]
    output = EfficientResample.apply(
        im.flatten(0, 1).unsqueeze(1), kernel[None, None, ...], 1, 2, False)
    return output.view(bs, nc, output.shape[2], output.shape[3])


def conv_resampled2d(x, w, f=None, up=False, down=False, padding=0):
    assert not (up and down)
    kw = w.shape[-1]

    if kw == 1 and down:
        assert padding == 0
        x = downsample2d(x, f)
        x = F.conv2d(x, w)
        return x

    if down:
        x = filter2d(x, f, transpose=True)
        x = F.conv2d(x, w, stride=2)
        return x

    if up:
        assert padding == 1
        x = F.conv_transpose2d(x, w.transpose(0, 1), stride=2)
        x = filter2d(x, f, gain=4)
        return x

    if not up and not down:
        # No resampling -> use regular convolution
        return F.conv2d(x, w, padding=padding)

    return None


def conv_modulated2d(x,
                     weight,
                     styles,
                     noise=None,
                     up=False,
                     down=False,
                     padding=0,
                     resample_filter=None,
                     demodulate=True):
    bs = x.shape[0]
    w = None
    dcoefs = None
    if demodulate:
        w = weight.unsqueeze(0)
        w = w * styles.reshape(bs, 1, -1, 1, 1)
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()

    x = x * styles.reshape(bs, -1, 1, 1)
    x = conv_resampled2d(x=x,
                         w=weight,
                         f=resample_filter,
                         up=up,
                         down=down,
                         padding=padding)
    if demodulate and noise is not None:
        x = torch.addcmul(noise, x, dcoefs.reshape(bs, -1, 1, 1))
    elif demodulate:
        x = x * dcoefs.reshape(bs, -1, 1, 1)
    elif noise is not None:
        x = x.add_(noise)
    return x


class EqualizedLinear(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 activate=False,
                 lr_multiplier=1,
                 init_bias_one=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate = activate
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels]) / lr_multiplier)
        if bias:
            if init_bias_one:
                self.bias = nn.Parameter(torch.ones(out_channels))
            else:
                self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.weight_gain = lr_multiplier / math.sqrt(in_channels)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        weight = self.weight * self.weight_gain
        b = self.bias * self.bias_gain if self.bias is not None else None

        x = F.linear(x, weight, b)
        if self.activate:
            x = F.leaky_relu(x.mul_(math.sqrt(2)), 0.2, inplace=True)
        return x


class EqualizedConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 activate=False,
                 up=False,
                 down=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate = activate
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', bilinear_filter())
        self.padding = kernel_size // 2
        self.weight_gain = 1 / math.sqrt(in_channels * (kernel_size**2))
        self.act_gain = math.sqrt(2) if activate else 1

        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = nn.Parameter(torch.zeros([out_channels])) if bias else None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias if self.bias is not None else None
        x = conv_resampled2d(x=x,
                             w=w,
                             f=self.resample_filter,
                             up=self.up,
                             down=self.down,
                             padding=self.padding)

        act_gain = self.act_gain * gain
        if b is not None:
            x = x + b.view([-1 if dim == 1 else 1 for dim in range(x.ndim)])
        if act_gain != 1:
            x = x.mul_(act_gain)
        if self.activate:
            x = F.leaky_relu(x, 0.2, inplace=True)
        return x


class MappingNetwork(nn.Module):

    def __init__(self,
                 z_dim,
                 c_dim,
                 w_dim,
                 num_ws,
                 num_layers=8,
                 embed_features=None,
                 layer_features=None,
                 lr_multiplier=0.01,
                 normalize_c=True):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.normalize_c = normalize_c

        if embed_features is None:
            embed_features = w_dim if normalize_c else c_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features
                        ] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0 and normalize_c:
            self.embed = EqualizedLinear(c_dim, embed_features)
        for idx in range(num_layers):
            in_channels = features_list[idx]
            out_channels = features_list[idx + 1]
            layer = EqualizedLinear(in_channels,
                                    out_channels,
                                    activate=True,
                                    lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

    @staticmethod
    def normalize_latent(x, dim=1, eps=1e-8):
        return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

    def forward(self, z, c):
        x = None
        if self.z_dim > 0:
            x = MappingNetwork.normalize_latent(z)
        if self.c_dim > 0:
            if self.normalize_c:
                y = MappingNetwork.normalize_latent(self.embed(c))
            else:
                y = c
            x = torch.cat([x, y], dim=-1) if x is not None else y

        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        if self.num_ws is not None and len(x.shape) == 2:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        return x


class SynthesisLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 resolution,
                 kernel_size=3,
                 up=False,
                 use_noise=True,
                 activate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activate = activate
        self.register_buffer('resample_filter', bilinear_filter())
        self.padding = kernel_size // 2
        self.act_gain = math.sqrt(2) if activate else 1

        self.affine = EqualizedLinear(w_dim, in_channels, init_bias_one=True)
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        if use_noise:
            self.register_buffer('noise_const',
                                 torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', gain=1):
        assert noise_mode in ['random', 'const']
        styles = self.affine(w)

        noise = None
        if (self.use_noise and noise_mode == 'random' and
            (self.training or self.noise_strength != 0)):
            # If we are in inference mode and the model was trained
            # without noise, we can save computation time.
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution],
                device=x.device) * self.noise_strength
        if (self.use_noise and noise_mode == 'const' and
                self.noise_strength != 0):
            noise = self.noise_const * self.noise_strength

        x = conv_modulated2d(x=x,
                             weight=self.weight,
                             styles=styles,
                             noise=noise,
                             up=self.up,
                             padding=self.padding,
                             resample_filter=self.resample_filter)

        act_gain = self.act_gain * gain

        x = x + self.bias.view([-1 if dim == 1 else 1 for dim in range(x.ndim)])
        if act_gain != 1:
            x = x.mul_(act_gain)
        if self.activate:
            x = F.leaky_relu(x, 0.2, inplace=True)
        return x


class OutputLayer(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.affine = EqualizedLinear(w_dim, in_channels, init_bias_one=True)
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / math.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = conv_modulated2d(x=x,
                             weight=self.weight,
                             styles=styles,
                             demodulate=False)

        x = x + self.bias.view([-1 if dim == 1 else 1 for dim in range(x.ndim)])
        return x


class SynthesisBlock(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution,
                 img_channels, is_last, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.register_buffer('resample_filter', bilinear_filter())
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels,
                                        out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        up=True,
                                        **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels,
                                    out_channels,
                                    w_dim=w_dim,
                                    resolution=resolution,
                                    **layer_kwargs)
        self.num_conv += 1

        self.torgb = OutputLayer(out_channels, img_channels, w_dim=w_dim)
        self.num_torgb += 1

    def forward(self, x, img, ws, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))

        if self.in_channels == 0:
            x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            x = self.conv0(x, next(w_iter), **layer_kwargs)
        x = self.conv1(x, next(w_iter), **layer_kwargs)

        if img is not None:
            img = upsample2d(img, self.resample_filter)

        y = self.torgb(x, next(w_iter))
        img = img.add_(y) if img is not None else y

        return x, img


class SynthesisNetwork(nn.Module):

    def __init__(self,
                 w_dim,
                 img_resolution,
                 img_channels,
                 channel_base=32768,
                 channel_max=512,
                 **block_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(math.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions
        }

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels,
                                   out_channels,
                                   w_dim=w_dim,
                                   resolution=res,
                                   img_channels=img_channels,
                                   is_last=is_last,
                                   **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(
                ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


class DiscriminatorBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 tmp_channels,
                 out_channels,
                 resolution,
                 img_channels,
                 activate=True):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels

        if in_channels == 0:
            self.fromrgb = EqualizedConv2d(img_channels,
                                           tmp_channels,
                                           kernel_size=1,
                                           activate=activate)

        self.conv0 = EqualizedConv2d(tmp_channels,
                                     tmp_channels,
                                     kernel_size=3,
                                     activate=activate)
        self.conv1 = EqualizedConv2d(tmp_channels,
                                     out_channels,
                                     kernel_size=3,
                                     activate=activate,
                                     down=True)
        self.skip = EqualizedConv2d(tmp_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    down=True)

    def forward(self, x, img):
        if self.in_channels == 0:
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = None

        y = self.skip(x, gain=math.sqrt(2) / 2)
        x = self.conv0(x)
        x = self.conv1(x, gain=math.sqrt(2) / 2)
        x = y.add_(x)

        return x, img


class DiscriminatorMinibatchStd(nn.Module):

    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        _, nc, h, w = x.shape
        ng = self.group_size
        f = self.num_channels
        nc //= f

        # Follows reference StyleGAN2 implementation
        y = x.reshape(ng, -1, f, nc, h, w)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4])
        y = y.reshape(-1, f, 1, 1)
        y = y.repeat(ng, 1, h, w)
        x = torch.cat([x, y], dim=1)
        return x


class DiscriminatorOutput(nn.Module):

    def __init__(self,
                 in_channels,
                 cmap_dim,
                 resolution,
                 img_channels,
                 mbstd_group_size=4,
                 mbstd_num_channels=1,
                 activate=True):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels

        self.mbstd = (DiscriminatorMinibatchStd(group_size=mbstd_group_size,
                                                num_channels=mbstd_num_channels)
                      if mbstd_num_channels > 0 else None)
        self.conv = EqualizedConv2d(in_channels + mbstd_num_channels,
                                    in_channels,
                                    kernel_size=3,
                                    activate=activate)
        self.fc = EqualizedLinear(in_channels * (resolution**2),
                                  in_channels,
                                  activate=activate)
        self.out = EqualizedLinear(in_channels,
                                   1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, cmap):
        x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) / math.sqrt(self.cmap_dim)
        return x


class DiscriminatorBackbone(nn.Module):

    def __init__(self,
                 c_dim,
                 img_resolution,
                 img_channels,
                 channel_base=32768,
                 channel_max=512,
                 cmap_dim=None,
                 block_kwargs={},
                 mapping_kwargs={}):
        super().__init__()

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(math.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels)
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = DiscriminatorBlock(in_channels,
                                       tmp_channels,
                                       out_channels,
                                       resolution=res,
                                       **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)

        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0,
                                          c_dim=c_dim,
                                          w_dim=cmap_dim,
                                          num_ws=None,
                                          **mapping_kwargs)

        self.b4 = DiscriminatorOutput(channels_dict[4],
                                      cmap_dim=cmap_dim,
                                      resolution=4,
                                      **common_kwargs)

    def forward(self, img, c=None, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim == -1:
            cmap = c
        elif self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, cmap)
        return x
