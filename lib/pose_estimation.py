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

import cv2
import numpy as np
import torch


def select_best_valid_pose(tvec, err):
    best_error = np.inf
    best_idx = None
    for idx in range(len(tvec)):
        if tvec[idx][2] > 0 and err[idx][0] < best_error:
            best_error = err[idx][0]
            best_idx = idx
    return best_idx, best_error


def compute_pose_pnp(coords, masks, focal_proposals, refine=True):
    bs, height, width, _ = coords.shape
    ii, jj = np.meshgrid(np.arange(width) / width,
                         np.arange(height) / height,
                         indexing='xy')
    grid_xy = np.stack((ii, jj), axis=-1) - 0.5
    grid_xy = grid_xy.reshape(-1, 2)
    coords = coords.astype(np.float64)

    intrinsics = np.eye(3)
    all_world2cam_mat = []
    all_focal = []
    all_errors = []
    for idx in range(bs):
        foreground_indices, = np.where(masks[idx].flatten())
        pts_xyz = coords[idx].reshape(-1, 3)[foreground_indices]
        pts_screen = grid_xy[foreground_indices]
        best_error = np.inf
        best_pose = None
        for focal in focal_proposals:
            if len(foreground_indices) < 4:
                # If less than 4 points, do not attempt to solve
                break

            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal

            main_solver = cv2.SOLVEPNP_SQPNP
            fallback_solver = cv2.SOLVEPNP_EPNP

            solver = main_solver
            best_idx = None
            while True:
                try:
                    retval, rvec, tvec, err = cv2.solvePnPGeneric(pts_xyz,
                                                                  pts_screen,
                                                                  intrinsics,
                                                                  None,
                                                                  flags=solver)
                    best_idx, err = select_best_valid_pose(tvec, err)
                    if best_idx is not None:
                        break
                except:
                    pass

                if solver == main_solver:
                    solver = fallback_solver
                else:
                    # Unable to solve
                    solver = None
                    break

            if solver is None:
                # No solutions or negative z -> discard
                continue

            if refine:
                # Sometimes the pose is slightly off from the optimum due to
                # numerical errors, so we refine it using an iterative solver
                retval, rvec_, tvec_, err_ = cv2.solvePnPGeneric(
                    pts_xyz,
                    pts_screen,
                    intrinsics,
                    None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=True,
                    rvec=rvec[best_idx],
                    tvec=tvec[best_idx])

                # If z is negative, the previous (working) solution is ruined
                if retval == 1 and tvec_[0][2] > 0:
                    tvec = tvec_
                    rvec = rvec_
                    best_idx = 0
                    err = err_[0][0]

            if err < best_error:
                best_error = err
                best_pose = (rvec[best_idx], tvec[best_idx], focal)

        if best_pose is not None:
            rvec, tvec, focal = best_pose
        else:
            # Return dummy pose pointing outside the object (empty screen)
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            tvec[2] = -10
            focal = 1
            best_error = 10.
        world2cam_mat = np.eye(4)
        rot_mat = cv2.Rodrigues(rvec)[0]
        flip = np.eye(4)
        flip[1, 1] = flip[2, 2] = -1
        world2cam_mat[:3, :3] = rot_mat
        world2cam_mat[:3, 3] = tvec.flatten()
        all_world2cam_mat.append(flip @ world2cam_mat)
        all_focal.append(focal)
        all_errors.append(best_error)

    return (np.stack(all_world2cam_mat,
                     axis=0), np.stack(all_focal,
                                       axis=0), np.stack(all_errors, axis=0))


def get_focal_guesses(focal_length):
    if focal_length is not None:
        sorted_focals = focal_length.cpu().numpy().copy()
        sorted_focals.sort()
        focal_guesses = np.percentile(
            sorted_focals, [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
        focal_guesses = np.unique(focal_guesses)
    else:
        focal_guesses = None
    return focal_guesses
