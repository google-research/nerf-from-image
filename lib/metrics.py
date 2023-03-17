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

import torch
import lpips
import skimage.metrics

from torch import nn


def range_check(im):
    # Checks that im is in [0, 1]
    with torch.no_grad():
        eps = 1e-1  # Some margin due to the wide sigmoid
        assert im.max() < 1 + eps, 'Range check failed'
        assert im.min() > -eps, 'Range check failed'


def psnr(pred, target, reduction='mean'):
    assert pred.shape == target.shape
    assert len(pred.shape) == 4
    assert pred.shape[1] == 3 or pred.shape[-1] == 3  # Ensure RGB image
    range_check(pred)
    range_check(target)
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    batch_psnr = -10 * torch.log10((pred - target).square().mean(dim=[1, 2, 3]))
    # We clamp each sample to max 60 dB, since a pixel-perfect reconstruction
    # would push the mean towards +infinity
    batch_psnr = batch_psnr.clamp(max=60)
    if reduction == 'mean':
        return batch_psnr.mean()
    else:
        return batch_psnr


def ssim(pred, target, reduction='mean'):
    assert pred.shape == target.shape
    assert len(pred.shape) == 4
    assert pred.shape[1] == 3
    range_check(pred)
    range_check(target)
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    device = pred.device
    if reduction == 'mean':
        pred = pred.cpu().flatten(0, 1).numpy()
        target = target.cpu().flatten(0, 1).numpy()
        return torch.FloatTensor([
            skimage.metrics.structural_similarity(pred,
                                                  target,
                                                  channel_axis=0,
                                                  data_range=1.)
        ]).to(device)
    else:
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        similarities = []
        for pred_elem, target_elem in zip(pred, target):
            similarities.append(
                skimage.metrics.structural_similarity(pred_elem,
                                                      target_elem,
                                                      channel_axis=0,
                                                      data_range=1.))
        return torch.FloatTensor(similarities).to(device)


def iou(alpha_pred, alpha_real, reduction='mean'):
    assert alpha_pred.shape == alpha_real.shape
    assert len(alpha_pred.shape) == 3 or (len(alpha_pred.shape) == 4 and
                                          alpha_pred.shape[1] == 1)
    range_check(alpha_pred)
    range_check(alpha_real)
    alpha_pred = alpha_pred > 0.5
    alpha_real = alpha_real > 0.5
    intersection = (alpha_pred & alpha_real).float().sum(dim=[-2, -1])
    union = (alpha_pred | alpha_real).float().sum(dim=[-2, -1])
    eps = 1e-6
    batch_iou = (intersection + eps) / (union + eps)
    if reduction == 'mean':
        return batch_iou.mean()
    else:
        return batch_iou.flatten()


class LPIPSLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg').eval()
        self.lpips.requires_grad_(False)

    def _compute_features(self, im):
        features = self.lpips.net(self.lpips.scaling_layer(im))
        normalized_features = []
        for i in range(self.lpips.L):
            feature_map = lpips.normalize_tensor(features[i])
            normalized_features.append(feature_map)
        return tuple(normalized_features)

    def forward(self, in0, in1=None, normalize=False, reduction='none'):
        if normalize:
            range_check(in0)
            # Map input in [0, 1] to [-1, 1] (if requested)
            in0 = 2 * in0 - 1
            if in1 is not None and not isinstance(in1, tuple):
                range_check(in1)
                in1 = 2 * in1 - 1

        features0 = self._compute_features(in0)
        if in1 is None:
            return features0
        elif not isinstance(in1, tuple):
            features1 = self._compute_features(in1)
        else:
            # Use cached features
            features1 = in1

        out_features = sum([
            lin((x - y).square()).mean(dim=[2, 3])
            for x, y, lin in zip(features0, features1, self.lpips.lins)
        ])
        if reduction == 'mean':
            return out_features.mean()
        else:
            return out_features
