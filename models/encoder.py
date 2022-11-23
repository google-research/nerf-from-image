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
from torch import nn
from models import segformer


class BootstrapEncoder(nn.Module):

    def __init__(self,
                 pose_regressor=True,
                 latent_regressor=True,
                 separate_backbones=False,
                 pretrained=True,
                 pretrained_model_path=None):
        super().__init__()

        if separate_backbones:
            assert pose_regressor and latent_regressor

        self.backbone = segformer.init_segformer(
            512,
            pretrained=pretrained,
            pretrained_model_path=pretrained_model_path)

        if separate_backbones:
            self.backbone_latent = segformer.init_segformer(
                512,
                pretrained=pretrained,
                pretrained_model_path=pretrained_model_path)

        self.pose_regressor = pose_regressor
        self.latent_regressor = latent_regressor
        self.separate_backbones = separate_backbones

        if pose_regressor:
            self.post = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 4, 3, padding=1),
            )
        if latent_regressor:
            self.w_regressor_pre = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.w_regressor_post = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, args.latent_dim),
                nn.LeakyReLU(0.2)  # Same as mapping network
            )

    def forward(self, x):
        features = self.backbone(x)

        if self.pose_regressor:
            # The output resolution of SegFormer is 1/4th of the input image,
            # so we upscale it.
            features_upscaled = F.interpolate(features,
                                              scale_factor=4,
                                              mode='bilinear',
                                              align_corners=False)
            features_upscaled = F.relu(features_upscaled, inplace=True)

            # Regress canonical map
            maps = self.post(features_upscaled)
            coords = maps[:, :3].permute(0, 2, 3, 1)
            segmentation = torch.sigmoid(maps[:, 3])
        else:
            coords = None
            segmentation = None

        if self.latent_regressor:
            if self.separate_backbones:
                # Use a different backbone
                features_latent = self.backbone_latent(x)
            else:
                features_latent = features
            features_latent = F.relu(features_latent)
            w = self.w_regressor_pre(features_latent)
            w = w.mean(dim=[2, 3])
            w = self.w_regressor_post(w).unsqueeze(1)
        else:
            w = None

        return coords, segmentation, w
