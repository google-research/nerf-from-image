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
import numpy as np


def invert_space(mat):
    """Converts a view matrix from cam2world to world2cam and vice-versa."""
    out_mat = torch.zeros_like(mat)
    out_mat[:, :3, :3] = mat[:, :3, :3].transpose(-2, -1) / mat[:, 3:4, 3:4]
    out_mat[:, 3, 3] = 1
    out_mat[:, :3, 3] = -torch.sum(
        mat[:, :3, :3] / mat[:, 3:4, 3:4] * mat[:, :3, None, 3], dim=-2)
    return out_mat


def quaternion_rotate_vector(q, v):
    """Rotates a vector about the rotation described by a quaternion."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3

    qvec = q[:, 1:].unsqueeze(1).expand(-1, v.shape[1], -1)
    uv = torch.cross(qvec, v, dim=2)
    uuv = torch.cross(qvec, uv, dim=2)
    return v + 2 * (q[:, :1].unsqueeze(1) * uv + uuv)


def quaternion_to_matrix(q):
    """Converts a unit quaternion to a rotation matrix."""
    return quaternion_rotate_vector(
        q,
        torch.eye(3, device=q.device).unsqueeze(0).expand(q.shape[0], -1, -1))


def pose_to_matrix(z0, t2, s, q, camera_flipped: bool):
    """Converts our pose representation to a cam2world view matrix."""
    R = quaternion_to_matrix(q)
    if z0 is not None:
        # Full perspective
        f = 1 + z0.exp()
        t3 = torch.cat((t2 / s.unsqueeze(-1), (f / s).unsqueeze(-1)), dim=-1)
        mat = torch.zeros((q.shape[0], 4, 4), device=R.device)
        mat[:, 3, 3] = 1
        mat[:, :3, :3] = R
        mat[:, :3, 3] = (t3[:, None, :] * R).sum(dim=-1)
        if camera_flipped:
            mat[:, :3, 1:] *= -1
        return mat, f / 2
    else:
        mat = torch.zeros((q.shape[0], 4, 4), device=R.device)
        mat[:, 3, 3] = 1
        mat[:, :3, :3] = R
        t3 = torch.cat((t2, torch.ones_like(t2[:, :1]) * 10), dim=-1)
        mat[:, :3, 3] = (t3[:, None, :] * R).sum(dim=-1)
        if camera_flipped:
            mat[:, :3, 1:] *= -1
        return mat / s[:, None, None], None


def matrix_to_quaternion(matrix):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    q = np.empty((4,))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def matrix_to_pose(tform_cam2world, focal_length, camera_flipped: bool):
    """Converts a cam2world view matrix to our pose representation."""
    assert not tform_cam2world.requires_grad

    tform_cam2world = tform_cam2world.clone()
    if camera_flipped:
        tform_cam2world[:, :3, 1:] *= -1
    M_inv = invert_space(tform_cam2world)
    t3 = -M_inv[:, :3, 3]

    if focal_length is not None:
        z0 = torch.log(2 * focal_length - 1)
        s = 2 * focal_length / t3[:, 2]
    else:
        z0 = None
        s = 1 / tform_cam2world[:, 3, 3]

    t2 = t3[:, :2] * s.unsqueeze(-1)
    R = []
    for m in M_inv.cpu().numpy():
        R.append(torch.FloatTensor(matrix_to_quaternion(m)).unsqueeze(0))
    R = torch.cat(R, dim=0).to(t3.device)

    return z0, t2, s, R


def matrix_to_conditioning_vector(tform_cam2world, focal_length,
                                  camera_flipped: bool):
    """Converts a view matrix to a conditioning vector for the discriminator."""
    # Static focal length
    tform_cam2world = tform_cam2world.clone()
    if camera_flipped:
        tform_cam2world[:, :3, 1:] *= -1
    M_inv = invert_space(tform_cam2world)
    R = M_inv[:, :3, :3].flatten(1, 2)
    t3 = -M_inv[:, :3, 3]

    if focal_length is not None:
        z0 = torch.log(focal_length)  # We use unshifted log here
        s = 2 * focal_length / t3[:, 2]
    else:
        z0 = None
        s = 1 / tform_cam2world[:, 3, 3]
        z0 = torch.zeros_like(s)

    t2 = t3[:, :2] * s.unsqueeze(-1)
    cond_vector = torch.cat((z0.unsqueeze(-1), t2, s.unsqueeze(-1), R), dim=-1)
    return cond_vector


def rotation_matrix_distance(p, q):
    """Computes the geodesic distance (in degrees) between rotation matrices."""
    if p.shape[-1] == 4:
        p = p[:, :3, :3] / p[:, 3:4, 3:4]
        q = q[:, :3, :3] / q[:, 3:4, 3:4]
    pqt = p @ q.transpose(-2, -1)
    trace = pqt[:, 0, 0] + pqt[:, 1, 1] + pqt[:, 2, 2]
    cos_distance = ((trace - 1) / 2).clamp(-1, 1)
    return torch.acos(cos_distance) / np.pi * 180


def perturb_poses(tform_cam2world, avg_angle, *extra_args):
    """Randomly perturbs poses without affecting the pose distribution."""
    indices = []
    random_generator = torch.Generator().manual_seed(1234)
    for pose in tform_cam2world:
        dist = rotation_matrix_distance(
            pose.unsqueeze(0).expand(tform_cam2world.shape[0], -1, -1),
            tform_cam2world)
        target_distance = (torch.rand(
            (1,), generator=random_generator) * avg_angle * 2).item()
        indices.append((dist - target_distance).abs().argmin().item())

    transformed_args = [
        arg[indices].clone() if arg is not None else None for arg in extra_args
    ]
    return (tform_cam2world[indices].clone(), *transformed_args)
