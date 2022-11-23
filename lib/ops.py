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
import torch.nn.functional as F
import numpy as np


def sample_volume_stratified(batch_size, nstrata, scene_range, device=None):
    bins = torch.arange(nstrata - 1, device=device)
    bins = torch.stack(torch.meshgrid(bins, bins, bins, indexing='xy'),
                       dim=-1).float().unsqueeze(0).expand(
                           batch_size, -1, -1, -1, -1)
    bins = (bins + torch.rand_like(bins)) / (nstrata - 1) * 2 - 1
    return bins.flatten(1, 3) * scene_range


def filt2d(im, kernel):
    bs, nc = im.shape[:2]
    if len(kernel.shape) == 1:
        # Treat it as a separable filter
        kernel = kernel[None, :] * kernel[:, None]
    if len(kernel.shape) == 2:
        kernel = kernel[None, None, ...]
    output = F.conv2d(im.flatten(0, 1).unsqueeze(1),
                      kernel,
                      padding=kernel.shape[-1] // 2)
    return output.view(bs, nc, output.shape[2], output.shape[3])


def blur(image, i: int, blur_warmup_iters: int, white_background: bool):
    blur_sigma = max(1 - i / blur_warmup_iters, 0) * 10
    blur_size = np.floor(blur_sigma * 3)
    if blur_size > 0:
        f = torch.arange(
            -blur_size, blur_size + 1,
            device=image.device).div(blur_sigma).square().neg().exp2()

        if white_background:
            image = image - 1  # Adjustment for white background
        image = filt2d(image, f / f.sum())
        if white_background:
            image = image + 1
    return image


@torch.jit.script
def grid_sample2d(image, grid):
    """Implements grid_sample2d with double-differentiation support.

    Equivalent to F.grid_sample(..., mode='bilinear',
                                padding_mode='border', align_corners=True).
    """
    bs, nc, ih, iw = image.shape
    _, h, w, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (iw - 1)
    iy = ((iy + 1) / 2) * (ih - 1)

    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    ix_nw = torch.clamp(ix_nw.long(), 0, iw - 1)
    iy_nw = torch.clamp(iy_nw.long(), 0, ih - 1)

    ix_ne = torch.clamp(ix_ne.long(), 0, iw - 1)
    iy_ne = torch.clamp(iy_ne.long(), 0, ih - 1)

    ix_sw = torch.clamp(ix_sw.long(), 0, iw - 1)
    iy_sw = torch.clamp(iy_sw.long(), 0, ih - 1)

    ix_se = torch.clamp(ix_se.long(), 0, iw - 1)
    iy_se = torch.clamp(iy_se.long(), 0, ih - 1)

    image = image.view(bs, nc, ih * iw)

    nw_val = torch.gather(image, 2,
                          (iy_nw * iw + ix_nw).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))
    ne_val = torch.gather(image, 2,
                          (iy_ne * iw + ix_ne).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))
    sw_val = torch.gather(image, 2,
                          (iy_sw * iw + ix_sw).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))
    se_val = torch.gather(image, 2,
                          (iy_se * iw + ix_se).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))

    out_val = (nw_val.view(bs, nc, h, w) * nw.view(bs, 1, h, w) +
               ne_val.view(bs, nc, h, w) * ne.view(bs, 1, h, w) +
               sw_val.view(bs, nc, h, w) * sw.view(bs, 1, h, w) +
               se_val.view(bs, nc, h, w) * se.view(bs, 1, h, w))

    return out_val
