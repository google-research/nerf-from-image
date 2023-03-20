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

from typing import Optional


@torch.jit.script
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """Equivalent to tf.math.cumprod(..., exclusive=True)."""
    cumprod = torch.cumprod(tensor[..., :-1], dim=-1)
    cumprod = torch.cat((torch.ones_like(cumprod[..., :1]), cumprod), dim=-1)
    return cumprod


@torch.jit.script
def get_ray_bundle(height: int,
                   width: int,
                   focal_length: Optional[torch.Tensor],
                   tform_cam2world: torch.Tensor,
                   bbox: Optional[torch.Tensor],
                   center: Optional[torch.Tensor] = None):

    ii, jj = torch.meshgrid(
        torch.arange(width, device=tform_cam2world.device) / width,
        torch.arange(height, device=tform_cam2world.device) / height,
        indexing='xy')
    if focal_length is not None:
        # Perspective camera projection model
        if center is not None:
            ii = ii.unsqueeze(0) - 0.5 * (2 * center[:, 0, None, None] -
                                          1) - 0.5
            jj = jj.unsqueeze(0) - 0.5 * (2 * center[:, 1, None, None] -
                                          1) - 0.5
        else:
            ii = ii.unsqueeze(0) - 0.5
            jj = jj.unsqueeze(0) - 0.5

        if bbox is not None:
            ii = (bbox[:, 1:2, 0].unsqueeze(-1) *
                  (ii + 0.5) + bbox[:, 0:1, 0].unsqueeze(-1)) * 0.5
            jj = -(bbox[:, 1:2, 1].unsqueeze(-1) *
                   (-jj + 0.5) + bbox[:, 0:1, 1].unsqueeze(-1)) * 0.5

        ii = ii / focal_length.unsqueeze(-1).unsqueeze(-1)
        jj = jj / focal_length.unsqueeze(-1).unsqueeze(-1)

        directions = torch.stack((ii, -jj, -torch.ones_like(ii)), dim=-1)
        ray_directions = torch.sum(directions[..., None, :] *
                                   tform_cam2world[:, None, None, :3, :3],
                                   dim=-1)
        ray_origins = tform_cam2world[:, None, None, :3,
                                      -1].expand(ray_directions.shape)
    else:
        # Ortho camera projection model
        ii = (ii.unsqueeze(0) - 0.5) * 2
        jj = (jj.unsqueeze(0) - 0.5) * 2

        if bbox is not None:
            ii = (bbox[:, 1:2, 0].unsqueeze(-1) * (ii / 2 + 0.5) +
                  bbox[:, 0:1, 0].unsqueeze(-1))
            jj = -(bbox[:, 1:2, 1].unsqueeze(-1) *
                   (-jj / 2 + 0.5) + bbox[:, 0:1, 1].unsqueeze(-1))

        origins = torch.stack((ii, -jj, torch.zeros_like(ii)), dim=-1)
        directions = torch.stack(
            (torch.zeros_like(ii), torch.zeros_like(ii), -torch.ones_like(ii)),
            dim=-1)

        ray_origins = (torch.sum(
            origins[..., None, :] * tform_cam2world[:, None, None, :3, :3],
            dim=-1) + tform_cam2world[:, None, None, :3, -1])
        ray_directions = (torch.sum(
            directions[..., None, :] * tform_cam2world[:, None, None, :3, :3],
            dim=-1) / tform_cam2world[:, None, None, 3, 3].unsqueeze(-1))
        # It is expected that ray_directions is already normalized,
        # since the transformation is orthogonal!

    return ray_origins, ray_directions


@torch.jit.script
def compute_query_points_from_rays(ray_origins: torch.Tensor,
                                   ray_directions: torch.Tensor,
                                   near_thresh: torch.Tensor,
                                   far_thresh: torch.Tensor,
                                   num_samples: int,
                                   randomize: bool = True):

    near_plane = near_thresh.unsqueeze(-1)
    far_plane = far_thresh.unsqueeze(-1)
    depth_values = torch.lerp(
        near_plane, far_plane,
        torch.arange(num_samples, device=ray_origins.device) / num_samples)

    if len(depth_values.shape) != 4:
        depth_values = depth_values[:, None, None, :]
        near_plane = near_plane[:, None, None, :]
        far_plane = far_plane[:, None, None, :]

    if randomize:
        delta = (far_plane - near_plane) / num_samples
        depth_values = depth_values + torch.rand_like(depth_values) * delta

    query_points = (ray_origins[..., None, :] +
                    ray_directions[..., None, :] * depth_values[..., :, None])

    return query_points, depth_values


@torch.jit.script
def render_volume_density(sigma_a: torch.Tensor,
                          rgb: torch.Tensor,
                          ray_origins: torch.Tensor,
                          ray_directions: torch.Tensor,
                          depth_values: torch.Tensor,
                          normals: Optional[torch.Tensor] = None,
                          semantics: Optional[torch.Tensor] = None,
                          white_background: bool = True):

    zero_tensor = torch.zeros((1,),
                              dtype=ray_origins.dtype,
                              device=ray_origins.device)
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                       zero_tensor.expand(depth_values[..., :1].shape)),
                      dim=-1)

    dists = dists * ray_directions.norm(p=2, dim=-1, keepdim=True)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights.detach() * depth_values.detach()).sum(dim=-1)
    if normals is not None:
        normal_map = (weights[..., None].detach() * normals).sum(dim=-2)
    else:
        normal_map = None
    if semantics is not None:
        semantic_map = (weights[..., None] * semantics).sum(dim=-2)
    else:
        semantic_map = None
    mask = weights.sum(-1)

    if white_background:
        rgb_map = rgb_map + (1. - mask[..., None])
        if normal_map is not None:
            normal_map = normal_map + (1. - mask[..., None])

    return rgb_map, depth_map, mask, normal_map, semantic_map


@torch.jit.script
def render_volume_density_weights_only(
    sigma_a: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    depth_values: torch.Tensor,
) -> torch.Tensor:
    zero_tensor = torch.zeros((1,),
                              dtype=ray_origins.dtype,
                              device=ray_origins.device)
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                       zero_tensor.expand(depth_values[..., :1].shape)),
                      dim=-1)
    dists = dists * ray_directions.norm(p=2, dim=-1, keepdim=True)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    return weights


@torch.jit.script
def sample_pdf(bins,
               weights,
               num_samples: int,
               deterministic: bool = False) -> torch.Tensor:

    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1)

    if deterministic:
        u = torch.linspace(0.0,
                           1.0,
                           steps=num_samples,
                           dtype=weights.dtype,
                           device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples],
                       dtype=weights.dtype,
                       device=weights.device)

    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


@torch.jit.script
def compute_near_far_planes(ray_origins: torch.Tensor,
                            ray_directions: torch.Tensor, scene_range: float):

    out_shape = ray_origins.shape[:-1]
    ray_origins = ray_origins.detach().reshape(-1, 3)
    ray_directions = ray_directions.detach().reshape(-1, 3)

    bvol = torch.tensor([[-scene_range] * 3, [scene_range] * 3],
                        dtype=ray_origins.dtype,
                        device=ray_origins.device)

    invdir = 1 / ray_directions
    neg_sign = (invdir < 0).long()
    pos_sign = 1 - neg_sign

    xmin = (bvol[neg_sign[:, 0], 0] - ray_origins[:, 0]) * invdir[:, 0]
    xmax = (bvol[pos_sign[:, 0], 0] - ray_origins[:, 0]) * invdir[:, 0]
    ymin = (bvol[neg_sign[:, 1], 1] - ray_origins[:, 1]) * invdir[:, 1]
    ymax = (bvol[pos_sign[:, 1], 1] - ray_origins[:, 1]) * invdir[:, 1]
    zmin = (bvol[neg_sign[:, 2], 2] - ray_origins[:, 2]) * invdir[:, 2]
    zmax = (bvol[pos_sign[:, 2], 2] - ray_origins[:, 2]) * invdir[:, 2]

    mask = torch.ones(ray_origins.shape[:-1],
                      dtype=torch.bool,
                      device=ray_origins.device)
    mask[(xmin > ymax) | (ymin > xmax)] = False
    near_plane = torch.max(xmin, ymin)
    far_plane = torch.min(xmax, ymax)
    mask[(near_plane > zmax) | (zmin > far_plane)] = False
    near_plane = torch.max(near_plane, zmin)
    far_plane = torch.min(far_plane, zmax)

    near_plane[~mask] = near_plane[mask].min()
    far_plane[~mask] = far_plane[mask].max()

    # Clamp negative values or values too close to zero
    near_plane.clamp_(min=0.1)
    far_plane.clamp_(min=0.1)

    # Make sure that distance between near and far is at least epsilon
    eps = 1e-3
    mask_eps = (far_plane - near_plane) < eps
    far_plane[mask_eps] = near_plane[mask_eps] + eps

    near_plane = near_plane.reshape(out_shape)
    far_plane = far_plane.reshape(out_shape)

    return near_plane, far_plane
