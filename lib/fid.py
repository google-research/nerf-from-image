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
import numpy as np
from scipy import linalg

from pytorch_fid.inception import InceptionV3


def init_inception(inception_weights='tensorflow'):
    assert inception_weights in ['pytorch', 'tensorflow']
    block_dim = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[block_dim]
    print(f'Initializing Inception network, {inception_weights} weights...')
    inception_model = InceptionV3(
        [block_idx], use_fid_inception=(inception_weights == 'tensorflow'))
    inception_model.requires_grad_(False)
    inception_model.eval()
    return inception_model


def forward_inception_batch(inception_model, images):
    pred = inception_model(images)[0]
    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.data.cpu().numpy().reshape(images.shape[0], -1)


def calculate_stats(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
