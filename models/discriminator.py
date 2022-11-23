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
from torch import nn
from models import stylegan
from lib import pose_utils


class Discriminator(nn.Module):

    def __init__(self,
                 resolution,
                 nc,
                 dataset_config,
                 conditional_pose=True,
                 use_encoder=False,
                 num_classes=None):
        super().__init__()

        self.conditional_pose = conditional_pose
        self.num_classes = num_classes
        self.use_encoder = use_encoder
        self.dataset_config = dataset_config

        if use_encoder:
            self.emb = ResidualEncoder(3, 256)
        if num_classes:
            self.label_embedding = nn.Embedding(num_classes, 512)

        mapping_kwargs = {}
        mapping_kwargs['lr_multiplier'] = 0.01
        mapping_kwargs['num_layers'] = 2
        mapping_kwargs['normalize_c'] = False

        c_dim = 0
        if self.conditional_pose:
            c_dim += 13
        if self.use_encoder:
            c_dim += 512
        if self.num_classes:
            c_dim += 512
        self.backbone = stylegan.DiscriminatorBackbone(
            c_dim, resolution, nc, mapping_kwargs=mapping_kwargs)

    def forward(self, x, iteration, pose=None, image=None, focal=None):

        if self.conditional_pose:
            cond_pose = pose_utils.matrix_to_conditioning_vector(
                pose, focal, self.dataset_config['camera_flipped'])
        if self.use_encoder:
            cond_image = self.emb(image)
        if self.num_classes:
            cond_label = self.label_embedding(image)

        if self.conditional_pose and self.num_classes:
            cond = torch.cat((cond_label, cond_pose), dim=-1)
        elif self.conditional_pose and self.use_encoder:
            cond = torch.cat((cond_image, cond_pose), dim=-1)
        elif self.conditional_pose:
            cond = cond_pose
        elif self.use_encoder:
            cond = cond_image
        elif self.num_classes:
            cond = cond_label
        else:
            cond = None

        return self.backbone(x, cond)
