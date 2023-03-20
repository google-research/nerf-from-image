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
from lib import ops
from models import stylegan


@torch.jit.script
def laplace_pdf(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Evaluates PDF of a Laplace distribution parameterized by beta."""
    return 0.5 * torch.exp(-x.abs() / beta) / beta


@torch.jit.script
def laplace_cdf(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Evaluates CDF of a Laplace distribution parameterized by beta."""
    return 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs() / beta))


@torch.jit.script
def wide_sigmoid_rescaled(x: torch.Tensor) -> torch.Tensor:
    """Computes wide sigmoid proposed in MipNeRF, then rescaled to [-1, 1]."""
    return torch.sigmoid(x) * 2.004 - 1.002


class ConditionalLayerNorm(nn.Module):

    def __init__(self, ch, emb_dim):
        super().__init__()

        self.norm = nn.LayerNorm(ch, elementwise_affine=False)
        self.emb_dim = emb_dim
        self.fc_gamma = stylegan.EqualizedLinear(emb_dim, ch)
        self.fc_beta = stylegan.EqualizedLinear(emb_dim, ch)

    def forward(self, x, z):
        x = self.norm(x)
        beta = self.fc_beta(z)
        while len(beta.shape) < len(x.shape):
            beta = beta.unsqueeze(-2)
        gamma = self.fc_gamma(z)
        while len(gamma.shape) < len(x.shape):
            gamma = gamma.unsqueeze(-2)
        return torch.addcmul(beta, 1 + gamma, x)


class ResidualEncoder(nn.Module):

    def __init__(self, nc, nd, use_instance_norm=False):
        super().__init__()

        if use_instance_norm:
            norm_layer = lambda nc: nn.InstanceNorm2d(nc, affine=True)
            bias = False
        else:
            norm_layer = lambda nc: nn.Identity()
            bias = True

        conv_layer = lambda in_ch, out_ch, ks, padding, stride, bias: stylegan.EqualizedConv2d(
            in_ch, out_ch, ks, bias=bias)

        self.conv1 = conv_layer(nc, 64, 3, padding=1, stride=1, bias=True)
        self.conv2 = conv_layer(64, 128, 3, padding=1, stride=1, bias=True)

        self.conv3 = conv_layer(128, 128, 3, padding=1, stride=1, bias=bias)
        self.norm3 = norm_layer(128)
        self.conv4 = conv_layer(128, 128, 3, padding=1, stride=1, bias=bias)
        self.norm4 = norm_layer(128)

        self.conv5 = conv_layer(128, 256, 3, padding=1, stride=1, bias=bias)
        self.norm5 = norm_layer(256)
        self.conv6 = conv_layer(256, 256, 3, padding=1, stride=1, bias=bias)
        self.norm6 = norm_layer(256)
        self.shortcut = conv_layer(128, 256, 1, padding=0, stride=1, bias=False)

        self.conv7 = conv_layer(256, 256, 3, padding=1, stride=1, bias=bias)
        self.norm7 = norm_layer(256)
        self.conv8 = conv_layer(256, 256, 3, padding=1, stride=1, bias=bias)
        self.norm8 = norm_layer(256)

        self.conv9 = conv_layer(256, 512, 3, padding=1, stride=1, bias=True)
        self.conv10 = conv_layer(512, 512, 3, padding=1, stride=1, bias=True)

        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        scale = math.sqrt(2) / 2

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)  # 128 - 64

        s = x
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.relu(self.norm4(self.conv4(x)))
        x = self.avgpool(x + s) * scale  # 64 - 32

        s = self.shortcut(x)
        x = self.relu(self.norm5(self.conv5(x)))
        x = self.relu(self.norm6(self.conv6(x)))
        x = self.avgpool(x + s) * scale  # 32 - 16

        s = x
        x = self.relu(self.norm7(self.conv7(x)))
        x = self.relu(self.norm8(self.conv8(x)))
        x = self.avgpool(x + s) * scale  # 16 - 8

        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))

        y = x.mean(dim=[2, 3])
        return y


class AttentionMapper(nn.Module):

    def __init__(self, latent_dim, num_values):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = 512
        self.const = nn.Parameter(torch.randn(1, self.hidden_size))

        self.fc1 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm1 = ConditionalLayerNorm(self.hidden_size, self.latent_dim)
        self.fc2 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm2 = ConditionalLayerNorm(self.hidden_size, self.latent_dim)
        self.fc3 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm3 = ConditionalLayerNorm(self.hidden_size, self.latent_dim)
        self.fc4 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm4 = ConditionalLayerNorm(self.hidden_size, self.latent_dim)

        self.fc5 = stylegan.EqualizedLinear(self.hidden_size, self.hidden_size)

        self.num_values = num_values
        self.fc_values = stylegan.EqualizedLinear(self.hidden_size,
                                                  num_values * 3)  # 3 for RGB

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, c):
        scale = math.sqrt(2) / 2

        x = self.const.expand(c.shape[0], -1)

        shortcut = x
        x = self.relu(self.norm1(self.fc1(x), c))
        x = self.relu(self.norm2(self.fc2(x), c))
        x = (x + shortcut).mul_(scale)

        shortcut = x
        x = self.relu(self.norm3(self.fc3(x), c))
        x = self.relu(self.norm4(self.fc4(x), c))
        x = (x + shortcut).mul_(scale)

        x = self.relu(self.fc5(x))

        values = self.fc_values(x).view(-1, self.num_values, 3)
        values = wide_sigmoid_rescaled(values)

        return values


class ViewDirectionMapper(nn.Module):

    def __init__(self, output_size, num_features=32):
        super().__init__()

        self.hidden_size = 64

        self.fc0 = stylegan.EqualizedLinear(3, self.hidden_size)
        self.fc1 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm1 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        self.fc2 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        self.fc3 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm3 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        self.fc4 = stylegan.EqualizedLinear(self.hidden_size,
                                            self.hidden_size,
                                            bias=False)
        self.norm4 = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

        self.fc5 = stylegan.EqualizedLinear(self.hidden_size, self.hidden_size)
        self.fc6 = stylegan.EqualizedLinear(self.hidden_size, num_features)

        self.output = stylegan.EqualizedLinear(num_features, output_size)
        self.output.weight.data.fill_(0.)
        self.output.bias.data.fill_(0.)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, viewdir):
        scale = math.sqrt(2) / 2

        x = self.relu(self.fc0(viewdir))

        shortcut = x
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.relu(self.norm2(self.fc2(x)))
        x = (x + shortcut).mul_(scale)

        shortcut = x
        x = self.relu(self.norm3(self.fc3(x)))
        x = self.relu(self.norm4(self.fc4(x)))
        x = (x + shortcut).mul_(scale)

        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        assert x.shape[-2] == 1, x.shape

        def mapper_closure(features):
            assert x.shape[-1] == features.shape[-1], (x.shape, features.shape)
            features_shape = features.shape
            features = features.view(*x.shape[:-2], -1, x.shape[-1])

            y = self.relu(x + features)
            y = y.view(features_shape)
            y = self.output(y)
            return y

        return mapper_closure


class MappingNetworkWrapper(nn.Module):

    def __init__(self, mapping_network, num_classes=None):
        super().__init__()
        self.backbone = mapping_network
        self.num_classes = num_classes

    def get_average_w(self, label=None, device=None):
        assert (self.num_classes is None) == (label is None)
        with torch.no_grad():
            device = self.backbone.fc0.weight.device
            if label is None:
                n_samples = 10000
                z = torch.randn((n_samples, self.backbone.z_dim), device=device)
                w = self(z)
                w_mean = w.mean(dim=0, keepdim=True)
            else:
                # Conditional case (fewer samples)
                n_samples = 256
                z = torch.randn(
                    (label.shape[0], n_samples, self.backbone.z_dim),
                    device=device)
                w = self(z, label.unsqueeze(1).expand(-1, n_samples, -1))
                w_mean = w.mean(dim=1,
                                keepdim=True).expand(-1, self.backbone.num_ws,
                                                     -1).contiguous()
        return w_mean

    def forward(self, z, c=None):
        return self.backbone(z, c)


class TriplanarDecoder(nn.Module):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.num_input_features = num_input_features

        hidden_dim = 64
        self.net = nn.Sequential(
            stylegan.EqualizedLinear(num_input_features, hidden_dim),
            nn.Softplus(),
            stylegan.EqualizedLinear(hidden_dim, 1 + num_output_features),
        )

    def forward(self, xy, xz, yz, coords, requires_double_backward=False):
        assert xy.shape[1] == self.num_input_features
        assert xz.shape[1] == self.num_input_features
        assert yz.shape[1] == self.num_input_features

        if requires_double_backward:
            # Use custom grid sample with double differentiation support
            e1 = ops.grid_sample2d(xy, coords[..., [0, 1]])
            e2 = ops.grid_sample2d(xz, coords[..., [0, 2]])
            e3 = ops.grid_sample2d(yz, coords[..., [1, 2]])
        else:
            e1 = F.grid_sample(xy,
                               coords[..., [0, 1]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)
            e2 = F.grid_sample(xz,
                               coords[..., [0, 2]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)
            e3 = F.grid_sample(yz,
                               coords[..., [1, 2]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)

        x = (e1 + e2 + e3) / 3
        x = x.view(x.shape[0], self.num_input_features, -1).transpose(-2, -1)
        x = self.net(x)
        return {'features': x[..., 1:], 'density_or_distance': x[..., :1]}


class Generator(nn.Module):

    def __init__(self,
                 latent_dim,
                 scene_range,
                 attention_values=0,
                 use_viewdir=False,
                 use_encoder=False,
                 disable_stylegan_noise=False,
                 use_sdf=False,
                 num_classes=None):
        super().__init__()

        self.scene_range = scene_range
        self.attention_values = attention_values
        self.use_viewdir = use_viewdir
        self.use_sdf = use_sdf
        self.use_encoder = use_encoder
        self.num_classes = num_classes

        c_dim = 512 if num_classes else 0
        z_dim = latent_dim
        w_dim = 512
        if use_encoder:
            self.emb = ResidualEncoder(3, w_dim, use_instance_norm=True)
            c_dim = w_dim

        lr_multiplier = 0.01
        num_ws = 14
        if attention_values > 0:
            num_ws += 1
        self.mapping_network = MappingNetworkWrapper(
            stylegan.MappingNetwork(z_dim=z_dim,
                                    c_dim=c_dim,
                                    w_dim=w_dim,
                                    num_ws=num_ws,
                                    num_layers=2,
                                    lr_multiplier=lr_multiplier,
                                    normalize_c=False), num_classes)
        self.synthesis_network = stylegan.SynthesisNetwork(w_dim=w_dim,
                                                           img_resolution=256,
                                                           img_channels=96)
        if use_viewdir:
            decoder_output_dim = 32
        elif attention_values > 0:
            # We can efficiently implement the attention-based color mapping
            # through a linear layer (matrix multiplication), since the
            # queries are constant.
            decoder_output_dim = attention_values
        else:
            decoder_output_dim = 3
        self.decoder = TriplanarDecoder(32, decoder_output_dim)

        if disable_stylegan_noise:
            for module in self.synthesis_network.modules():
                buffer_keys = [k for k, _ in module.named_buffers()]
                if 'noise_const' in buffer_keys:
                    module.use_noise = False

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if self.use_viewdir:
            self.viewdir_mapper = ViewDirectionMapper(
                attention_values if attention_values > 0 else 3, 32)
        if use_sdf:
            self.beta = nn.Parameter(torch.FloatTensor([0.1]))
            self.alpha = nn.Parameter(torch.FloatTensor([1.]))

        if attention_values > 0:
            self.texture_mapper = AttentionMapper(w_dim, attention_values)

        if num_classes:
            self.class_embedding = nn.Embedding(num_classes, 512)

    def forward(self,
                viewdir,
                c,
                request_model_outputs=['sampler'],
                model_inputs={}):
        for output in request_model_outputs:
            assert output in [
                'sampler', 'sdf_eikonal_loss', 'sdf_distance_loss',
                'path_length', 'total_variation_loss', 'entropy_loss',
                'attention_values', 'bbox'
            ]
        for k in model_inputs.keys():
            assert k in [
                'freeze_noise', 'attention_values', 'attention_values_bias'
            ]

        if self.use_encoder:
            z, image = c
            c = self.emb(image)
            ws = self.mapping_network(z, c)
        else:
            if self.num_classes:
                if isinstance(c, (list, tuple)):
                    c, label = c
                    assert len(c.shape) == 2
                    label = self.class_embedding(label)
                else:
                    assert len(c.shape) == 3
            else:
                label = None
            # Unconditional
            if len(c.shape) == 3:
                if c.shape[1] == 1:
                    # Broadcast
                    ws = c.expand(-1, self.mapping_network.backbone.num_ws,
                                  -1).contiguous()
                else:
                    ws = c
            else:
                ws = self.mapping_network(c, label)

        if 'path_length' in request_model_outputs:
            assert torch.is_grad_enabled()
            ws = ws.contiguous().requires_grad_()

        if self.attention_values > 0:
            assert ws.shape[1] == 15
            w_tex = ws[:, 14]
            w_synthesis = ws[:, :14]
            if 'attention_values' in model_inputs:
                attention_values = model_inputs['attention_values']  # Override
            elif 'sampler' in request_model_outputs:
                attention_values = self.texture_mapper(w_tex)
                if 'attention_values_bias' in model_inputs:
                    attention_values = attention_values + model_inputs[
                        'attention_values_bias']
            else:
                attention_values = None
        else:
            w_synthesis = ws

        if self.use_viewdir and viewdir is not None:
            viewdir_mapper_closure = self.viewdir_mapper(viewdir)

        if 'freeze_noise' in model_inputs and model_inputs['freeze_noise']:
            block_kwargs = {'noise_mode': 'const'}
        else:
            block_kwargs = {}
        planes = self.synthesis_network(w_synthesis, **block_kwargs)
        planes = planes.view(c.shape[0], 3, 32, planes.shape[-2],
                             planes.shape[-1])

        model_outputs = {}
        if 'attention_values' in request_model_outputs:
            assert self.attention_values > 0
            model_outputs['attention_values'] = attention_values

        if 'path_length' in request_model_outputs:
            pl_noise = torch.randn_like(planes) / math.sqrt(
                planes.shape[-2] * planes.shape[-1])
            if self.attention_values > 0:
                pl_noise_attn = torch.randn_like(attention_values)
                pl_grad, = torch.autograd.grad(
                    (planes * pl_noise).sum() +
                    (attention_values * pl_noise_attn).sum(),
                    inputs=ws,
                    create_graph=True)
            else:
                pl_grad, = torch.autograd.grad((planes * pl_noise).sum(),
                                               inputs=ws,
                                               create_graph=True)
            ppl = pl_grad.square().sum(dim=-1).mean(dim=-1).sqrt()
            model_outputs['path_length'] = ppl

        xy = planes[:, 0]
        xz = planes[:, 1]
        yz = planes[:, 2]

        if 'sdf_eikonal_loss' in request_model_outputs \
                or 'total_variation_loss' in request_model_outputs or 'entropy_loss' in request_model_outputs:
            assert torch.is_grad_enabled()

            nstrata = 32
            bins_in = ops.sample_volume_stratified(planes.shape[0],
                                                   nstrata,
                                                   self.scene_range,
                                                   device=planes.device)

            requires_double_backward_model = False
            if 'sdf_eikonal_loss' in request_model_outputs:
                assert self.use_sdf and self.training
                bins_in.requires_grad_()
                requires_double_backward_model = True

            eik_coords = bins_in / self.scene_range
            eik_coords = eik_coords.view(planes.shape[0], 1, -1, 3)

            x_eik = self.decoder(
                xy,
                xz,
                yz,
                eik_coords,
                requires_double_backward=requires_double_backward_model)

            if 'sdf_eikonal_loss' in request_model_outputs:
                x_grad_eik, = torch.autograd.grad(
                    x_eik['density_or_distance'][..., -1].sum(),
                    bins_in,
                    create_graph=True)
                sdf_magnitude = x_grad_eik.norm(dim=-1)
                eikonal_loss = ((sdf_magnitude - 1)**2).flatten(1).mean(dim=1)
                model_outputs['sdf_eikonal_loss'] = eikonal_loss

            if 'sdf_distance_loss' in request_model_outputs:
                assert self.use_sdf
                with torch.no_grad():
                    target_distance = bins_in.norm(dim=-1) - 1  # Unit sphere
                loss_dist = F.mse_loss(
                    x_eik['density_or_distance'][..., -1].flatten(1),
                    target_distance.flatten(1),
                    reduction='none').mean(dim=1)
                model_outputs['sdf_distance_loss'] = loss_dist

            if 'total_variation_loss' in request_model_outputs or 'entropy_loss' in request_model_outputs:
                if 'total_variation_loss' in request_model_outputs:
                    eik_coords_detached = eik_coords.detach()
                    eik_coords_perturb = eik_coords_detached + torch.randn_like(
                        eik_coords_detached) * 0.004
                    x_eik_perturb = self.decoder(xy, xz, yz, eik_coords_perturb)

                if self.use_sdf:
                    beta = self.beta
                    alpha = 1 / self.alpha
                    neg_distance = -x_eik['density_or_distance'][..., -1]
                    if 'total_variation_loss' in request_model_outputs:
                        density_prealpha = laplace_cdf(neg_distance, beta)
                        density_prealpha_perturb = laplace_cdf(
                            -x_eik_perturb['density_or_distance'][..., -1],
                            beta)
                        model_outputs['total_variation_loss'] = F.l1_loss(
                            density_prealpha,
                            density_prealpha_perturb,
                            reduction='none').flatten(1).mean(dim=1)
                    if 'entropy_loss' in request_model_outputs:
                        model_outputs['entropy_loss'] = laplace_pdf(
                            neg_distance, beta).flatten(1).mean(dim=1)
                else:
                    # Standard NeRF density
                    density_pre = x_eik['density_or_distance'][..., -1] - 1
                    tv_term_ = torch.sigmoid(density_pre)
                    if 'total_variation_loss' in request_model_outputs:
                        tv_term_perturb_ = torch.sigmoid(
                            x_eik_perturb['density_or_distance'][..., -1] - 1)
                        model_outputs['total_variation_loss'] = F.l1_loss(
                            tv_term_, tv_term_perturb_,
                            reduction='none').flatten(1).mean(dim=1)
                    if 'entropy_loss' in request_model_outputs:
                        model_outputs['entropy_loss'] = (
                            tv_term_ * (1 - tv_term_)).flatten(1).mean(dim=1)

        def sampler(x_in, request_sampler_outputs=['sigma', 'rgb']):
            for output in request_sampler_outputs:
                assert output in [
                    'sdf_distance', 'sigma', 'rgb', 'normals', 'semantics',
                    'coords'
                ]

            sampler_outputs = {}

            bs = x_in.shape[0]
            depth = x_in.shape[-2]

            if 'normals' in request_sampler_outputs:
                assert self.use_sdf and torch.is_grad_enabled(
                ) and not self.training
                x_in = x_in.requires_grad_()

            x = x_in.view(bs, -1, 1, 3) / self.scene_range
            with torch.no_grad():
                mask = (x.abs() > 1).any(dim=-1).float()
                mask = mask.flatten(1, len(mask.shape) - 1)

            x = self.decoder(xy, xz, yz, x)

            density_or_distance = x['density_or_distance']
            features = x['features']

            if 'normals' in request_sampler_outputs:
                x_grad, = torch.autograd.grad(density_or_distance[...,
                                                                  -1].sum(),
                                              x_in,
                                              create_graph=False)
                sampler_outputs['normals'] = F.normalize(x_grad, dim=-1)
                density_or_distance = density_or_distance.detach()
                features = features.detach()
                x_in = x_in.detach()
                x = None

            if 'sdf_distance' in request_sampler_outputs:
                sampler_outputs['sdf_distance'] = density_or_distance

            if 'sigma' in request_sampler_outputs:
                if self.use_sdf:
                    beta = self.beta
                    alpha = 1 / self.alpha
                    neg_distance = -density_or_distance[..., -1]
                    density_prealpha = laplace_cdf(neg_distance,
                                                   beta) * (1 - mask)
                    sigma_final = alpha * density_prealpha
                    sampler_outputs['sigma'] = sigma_final
                else:
                    # Standard NeRF density
                    density_pre = density_or_distance[..., -1] - 1
                    sigma_final = F.softplus(density_pre) * (1 - mask)
                    sampler_outputs['sigma'] = sigma_final

            if 'coords' in request_sampler_outputs:
                sampler_outputs['coords'] = x_in
                if 'bbox' in request_model_outputs:
                    # Add bbox (for visualization purposes)
                    eps = 5e-2
                    x_flat = x_in.view(x_in.shape[0], -1, 3).abs()
                    bbox_mask = torch.ones_like(sigma_final)
                    bbox_mask *= 1 - (x_flat[..., [0, 1]] < self.scene_range -
                                      eps).all(dim=-1).float()
                    bbox_mask *= 1 - (x_flat[..., [0, 2]] < self.scene_range -
                                      eps).all(dim=-1).float()
                    bbox_mask *= 1 - (x_flat[..., [1, 2]] < self.scene_range -
                                      eps).all(dim=-1).float()
                    bbox_mask *= 1 - (x_flat[..., [1, 2]] < self.scene_range -
                                      eps).all(dim=-1).float()
                    bbox_mask *= (1 - mask)
                    sigma_final += 100 * bbox_mask

            if 'rgb' in request_sampler_outputs or 'semantics' in request_sampler_outputs:
                if self.use_viewdir:
                    features = viewdir_mapper_closure(features)

                if self.attention_values == 0:
                    rgb = wide_sigmoid_rescaled(features)
                else:
                    attention_probs = F.softmax(
                        features, dim=-1
                    )  # Attention probabilities (distribution over semantic classes)

                if 'semantics' in request_sampler_outputs:
                    assert self.attention_values > 0
                    sampler_outputs['semantics'] = attention_probs

                if 'rgb' in request_sampler_outputs:
                    if self.attention_values > 0:
                        rgb = torch.matmul(attention_probs, attention_values)
                    sampler_outputs['rgb'] = rgb

            return sampler_outputs

        if 'sampler' in request_model_outputs:
            model_outputs['sampler'] = sampler

        return model_outputs
