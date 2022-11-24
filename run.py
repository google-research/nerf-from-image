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

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import sys
import math

from torch.utils import tensorboard
from torch import nn
from tqdm import tqdm

import arguments
from data import loaders
from lib import pose_utils
from lib import nerf_utils
from lib import utils
from lib import fid
from lib import ops
from lib import metrics
from lib import pose_estimation
from models import generator
from models import discriminator
from models import encoder

args = arguments.parse_args()
gpu_ids = list(range(args.gpus))

if args.gpus > 0 and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

if args.dataset == 'autodetect':
    assert args.resume_from
    args.dataset = loaders.autodetect_dataset(args.resume_from)

print(args)

experiment_name = arguments.suggest_experiment_name(args)
resume_from = None
log_dir = 'gan_logs'
report_dir = 'reports'
file_dir = 'gan_checkpoints'

checkpoint_dir = os.path.join(args.root_path, file_dir, experiment_name)
if not args.run_inversion:
    utils.mkdir(checkpoint_dir)
print('Saving checkpoints to', checkpoint_dir)

tensorboard_dir = os.path.join(args.root_path, log_dir, experiment_name)
report_dir = os.path.join(args.root_path, report_dir)
print('Saving tensorboard logs to', tensorboard_dir)
if not args.run_inversion:
    utils.mkdir(tensorboard_dir)
if args.run_inversion:
    print('Saving inversion reports to', report_dir)
    utils.mkdir(report_dir)

if args.run_inversion:
    writer = None  # Instantiate later
else:
    writer = tensorboard.SummaryWriter(tensorboard_dir)

if args.resume_from:
    # Resume from specified checkpoint
    if '.pth' in args.resume_from:
        # Load specified filename
        print('Attempting to load specified checkpoint (filename)...')
        last_checkpoint_dir = os.path.join(args.root_path, 'gan_checkpoints',
                                           args.resume_from)
        # Strip filename from checkpoint dir
        args.resume_from = os.path.dirname(args.resume_from)
    elif '+' in args.resume_from:
        args.resume_from, checkpoint_iter = args.resume_from.split('+')
        print(
            f'Attempting to load specified checkpoint (iteration {checkpoint_iter})...'
        )
        last_checkpoint_dir = os.path.join(args.root_path, 'gan_checkpoints',
                                           args.resume_from,
                                           f'checkpoint_{checkpoint_iter}.pth')
    else:
        # Load latest
        print('Attempting to load latest checkpoint...')
        last_checkpoint_dir = os.path.join(args.root_path, 'gan_checkpoints',
                                           args.resume_from,
                                           'checkpoint_latest.pth')

    if utils.file_exists(last_checkpoint_dir):
        print('Resuming from manual checkpoint', last_checkpoint_dir)
        with utils.open_file(last_checkpoint_dir, 'rb') as f:
            resume_from = torch.load(f, map_location='cpu')
    else:
        raise ValueError(
            f'Specified checkpoint {args.resume_from} does not exist!')
else:
    # Check if checkpoint exists
    last_checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    if utils.file_exists(last_checkpoint_dir):
        print('Resuming from', last_checkpoint_dir)
        with utils.open_file(last_checkpoint_dir, 'rb') as f:
            resume_from = torch.load(f, map_location='cpu')
        if resume_from['iteration'] < 12500:
            # Warm-up was not completed, so we might as well train from scratch
            resume_from = None
            print('Aborting resume (training from scratch)')

if resume_from is not None:
    print('Checkpoint iteration:', resume_from['iteration'])
    if 'fid_untrunc' in resume_from:
        print('Checkpoint unconditional FID:', resume_from['fid_untrunc'])
        print('Checkpoint unconditional FID (best):', resume_from['best_fid'])

if args.attention_values > 0:
    color_palette = utils.get_color_palette(args.attention_values).to(device)
else:
    color_palette = None

dataset_config, train_split, train_eval_split, test_split = loaders.load_dataset(
    args, device)

if args.perturb_poses > 0:
    print('Perturbing poses', args.perturb_poses)
    train_split.tform_cam2world, train_split.focal_length, train_split.bbox = pose_utils.perturb_poses(
        train_split.tform_cam2world, args.perturb_poses,
        train_split.focal_length, train_split.bbox)

    if train_eval_split.tform_cam2world.shape == train_split.tform_cam2world.shape:
        train_eval_split.tform_cam2world = train_split.tform_cam2world
        train_eval_split.focal_length = train_split.focal_length
        train_eval_split.bbox = train_split.bbox
    else:
        train_eval_split.tform_cam2world, train_eval_split.focal_length, train_eval_split.bbox = pose_utils.perturb_poses(
            train_eval_split.tform_cam2world, args.perturb_poses,
            train_eval_split.focal_length, train_eval_split.bbox)


def render(target_model,
           height,
           width,
           tform_cam2world,
           focal_length,
           center,
           bbox,
           model_input,
           depth_samples_per_ray,
           randomize=True,
           compute_normals=False,
           compute_semantics=False,
           compute_coords=False,
           extra_model_outputs=[],
           extra_model_inputs={},
           force_no_cam_grad=False):

    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, bbox, center)

    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(),
            dataset_config['scene_range'])

    query_points, depth_values = nerf_utils.compute_query_points_from_rays(
        ray_origins,
        ray_directions,
        near_thresh,
        far_thresh,
        depth_samples_per_ray,
        randomize=randomize,
    )

    if force_no_cam_grad:
        query_points = query_points.detach()
        depth_values = depth_values.detach()
        ray_directions = ray_directions.detach()

    if args.use_viewdir:
        viewdirs = ray_directions.unsqueeze(-2)
    else:
        viewdirs = None

    model_outputs = target_model(viewdirs, model_input,
                                 ['sampler'] + extra_model_outputs,
                                 extra_model_inputs)
    radiance_field_sampler = model_outputs['sampler']
    del model_outputs['sampler']

    request_sampler_outputs = ['sigma', 'rgb']
    if compute_normals:
        assert args.use_sdf
        request_sampler_outputs.append('normals')
    if compute_semantics:
        assert args.attention_values > 0
        request_sampler_outputs.append('semantics')
    if compute_coords:
        request_sampler_outputs.append('coords')
    sampler_outputs_coarse = radiance_field_sampler(query_points,
                                                    request_sampler_outputs)
    sigma = sampler_outputs_coarse['sigma'].view(*query_points.shape[:-1], -1)
    rgb = sampler_outputs_coarse['rgb'].view(*query_points.shape[:-1], -1)

    if compute_normals:
        normals = sampler_outputs_coarse['normals'].view(
            *query_points.shape[:-1], -1)
    else:
        normals = None

    if compute_semantics:
        semantics = sampler_outputs_coarse['semantics'].view(
            *query_points.shape[:-1], -1)
    else:
        semantics = None

    if compute_coords:
        coords = sampler_outputs_coarse['coords'].view(*query_points.shape[:-1],
                                                       -1)
    else:
        coords = None

    if args.fine_sampling:
        z_vals = depth_values
        with torch.no_grad():
            weights = nerf_utils.render_volume_density_weights_only(
                sigma.squeeze(-1), ray_origins, ray_directions,
                depth_values).flatten(0, 2)

            # Smooth weights as in EG3D
            weights = F.max_pool1d(weights.unsqueeze(1).float(),
                                   2,
                                   1,
                                   padding=1)
            weights = F.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = nerf_utils.sample_pdf(
                z_vals_mid.flatten(0, 2),
                weights[..., 1:-1],
                depth_samples_per_ray,
                deterministic=not randomize,
            )
            z_samples = z_samples.view(*z_vals.shape[:3], z_samples.shape[-1])

        z_values_sorted, z_indices_sorted = torch.sort(torch.cat(
            (z_vals, z_samples), dim=-1),
                                                       dim=-1)
        query_points_fine = ray_origins[
            ...,
            None, :] + ray_directions[..., None, :] * z_samples[..., :, None]

        sampler_outputs_fine = radiance_field_sampler(query_points_fine,
                                                      request_sampler_outputs)
        sigma_fine = sampler_outputs_fine['sigma'].view(
            *query_points_fine.shape[:-1], -1)
        rgb_fine = sampler_outputs_fine['rgb'].view(
            *query_points_fine.shape[:-1], -1)
        if compute_normals:
            normals_fine = sampler_outputs_fine['normals'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            normals_fine = None
        if compute_semantics:
            semantics_fine = sampler_outputs_fine['semantics'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            semantics_fine = None
        if compute_coords:
            coords_fine = sampler_outputs_fine['coords'].view(
                *query_points_fine.shape[:-1], -1)
        else:
            coords_fine = None

        sigma = torch.cat((sigma, sigma_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                  sigma.shape[-1]))
        rgb = torch.cat((rgb, rgb_fine), dim=-2).gather(
            -2,
            z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                  rgb.shape[-1]))
        if normals_fine is not None:
            normals = torch.cat((normals, normals_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      normals.shape[-1]))
        if semantics_fine is not None:
            semantics = torch.cat((semantics, semantics_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      semantics.shape[-1]))
        if coords_fine is not None:
            coords = torch.cat((coords, coords_fine), dim=-2).gather(
                -2,
                z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                      coords.shape[-1]))
        depth_values = z_values_sorted

    if coords is not None:
        semantics = coords

    rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted = nerf_utils.render_volume_density(
        sigma.squeeze(-1),
        rgb,
        ray_origins,
        ray_directions,
        depth_values,
        normals,
        semantics,
        white_background=dataset_config['white_background'])

    return rgb_predicted, depth_predicted, mask_predicted, normals_predicted, semantics_predicted, model_outputs


class GANLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor, target_positive: bool):
        if target_positive:
            return F.softplus(-x).mean()
        else:
            return F.softplus(x).mean()


def update_generator_ema(iteration):
    alpha = 0.5**(32 / 10000)
    with torch.no_grad():
        if iteration < 1000:
            alpha = math.pow(alpha, 100)
        elif iteration < 10000:
            alpha = math.pow(alpha, 10)
        g_state_dict = model.state_dict()
        for k, param in model_ema.state_dict().items():
            if torch.is_floating_point(param):
                param.mul_(alpha).add_(g_state_dict[k], alpha=1 - alpha)
            else:
                param.fill_(g_state_dict[k])


evaluation_res = args.resolution
inception_net = fid.init_inception('tensorflow').to(device).eval()

# Compute stats
print('Computing FID stats...')


def compute_real_fid_stats(images_fid_actual):
    n_images_fid = len(images_fid_actual)
    all_activations = []
    with torch.no_grad():
        test_bs = args.batch_size
        for idx in tqdm(range(0, n_images_fid, test_bs)):
            im = images_fid_actual[idx:idx + test_bs].to(device) / 2 + 0.5
            im = im.permute(0, 3, 1, 2)[:, :3]
            if evaluation_res == 64:
                im = F.avg_pool2d(im, 2)
            assert im.shape[-1] == evaluation_res
            all_activations.append(
                fid.forward_inception_batch(inception_net, im))
    all_activations = np.concatenate(all_activations, axis=0)
    return fid.calculate_stats(all_activations)


train_eval_split.fid_stats = compute_real_fid_stats(train_eval_split.images)
n_images_fid = len(train_eval_split.images)

if ((args.run_inversion and args.inv_use_testset) or
        args.use_encoder) and dataset_config['views_per_object_test']:
    print('Computing FID stats for test set...')
    test_split.fid_stats = compute_real_fid_stats(test_split.images)

random_seed = 1234
n_images_fid_max = 8000  # Matches Pix2NeRF evaluation protocol

random_generator = torch.Generator()
random_generator.manual_seed(random_seed)
if n_images_fid > n_images_fid_max:
    # Select random indices without replacement
    train_eval_split.eval_indices = torch.randperm(
        n_images_fid, generator=random_generator)[:n_images_fid_max].sort()[0]
else:
    if args.dataset.startswith('imagenet_'):
        # Select n_images_fid random poses (potentially repeated)
        remaining = n_images_fid
        train_eval_split.eval_indices = []
        while remaining > 0:
            nimg = len(train_eval_split.images)
            train_eval_split.eval_indices.append(
                torch.randperm(nimg, generator=random_generator)[:remaining])
            remaining -= len(train_eval_split.eval_indices[-1])
        train_eval_split.eval_indices = torch.cat(train_eval_split.eval_indices,
                                                  dim=0).sort()[0]
    else:
        assert len(train_split.images) == n_images_fid or len(
            train_split.images) == 2 * n_images_fid
        train_eval_split.eval_indices = torch.arange(n_images_fid)

print(f'Evaluating training FID on {len(train_eval_split.eval_indices)} images')

if args.use_encoder or args.run_inversion:

    def compute_view_perm(target_img_indices, views_per_object):
        if views_per_object == 1:
            # No novel views available (simple FID evaluation from random views, no PSNR)
            target_img_perm = torch.randperm(len(target_img_indices),
                                             generator=random_generator)
        else:
            # Novel views are available:
            # match each index to another index that represents the same object from a different view
            obj_index = {}
            for idx in target_img_indices:
                key = idx.item() // views_per_object
                if key not in obj_index:
                    obj_index[key] = []
                obj_index[key].append(idx.item())

            target_img_perm = []
            for idx in target_img_indices:
                key = idx.item() // views_per_object
                views = obj_index[key]
                if len(obj_index[key]) == 1:
                    target_img_perm.append(views[0])
                else:
                    while True:
                        tentative_view = torch.randint(
                            len(views), size=(1,),
                            generator=random_generator).item()
                        # Avoid selecting the same view
                        if tentative_view != idx.item():
                            target_img_perm.append(views[tentative_view])
                            break
        target_img_perm = torch.LongTensor(target_img_perm)
        return target_img_perm

    train_eval_split.eval_indices_perm = compute_view_perm(
        train_eval_split.eval_indices, dataset_config['views_per_object'])
    if dataset_config['views_per_object_test']:
        test_split.eval_indices = torch.arange(len(test_split.images))
        if len(test_split.eval_indices) > n_images_fid_max:
            test_split.eval_indices = torch.randperm(
                len(test_split.eval_indices),
                generator=random_generator)[:n_images_fid_max].sort()[0]
        print(f'Evaluating test set on {len(test_split.eval_indices)} images')
        test_split.eval_indices_perm = compute_view_perm(
            test_split.eval_indices, dataset_config['views_per_object_test'])

random_generator.manual_seed(random_seed)  # Re-seed
z_fid_untrunc = torch.randn(
    (len(train_eval_split.eval_indices), args.latent_dim),
    generator=random_generator)
z_fid_untrunc = z_fid_untrunc.to(device)

if dataset_config['views_per_object_test'] and (args.use_encoder or
                                                args.run_inversion):
    z_fid_untrunc_test = torch.randn(
        (len(test_split.eval_indices_perm), args.latent_dim),
        generator=random_generator)
    z_fid_untrunc_test = z_fid_untrunc_test.to(device)

torch.cuda.empty_cache()

batch_size = args.batch_size
use_viewdir = args.use_viewdir
supervise_alpha = args.supervise_alpha
use_encoder = args.use_encoder
use_r1 = args.r1 > 0
use_tv = args.tv > 0
use_entropy = args.entropy > 0

# Number of depth samples along each ray
depth_samples_per_ray = 64
if not args.fine_sampling:
    depth_samples_per_ray *= 2  # More fair comparison

num_iters = 0 if args.run_inversion else args.iterations
display_every = 5000


def create_model():
    return generator.Generator(
        args.latent_dim,
        dataset_config['scene_range'],
        attention_values=args.attention_values,
        use_viewdir=use_viewdir,
        use_encoder=args.use_encoder,
        disable_stylegan_noise=args.disable_stylegan_noise,
        use_sdf=args.use_sdf,
        num_classes=train_split.num_classes
        if args.use_class else None).to(device)


if args.dual_discriminator_l1 or args.dual_discriminator_mse:
    discriminator = None
else:
    discriminator = discriminator.Discriminator(
        args.resolution,
        nc=4 if supervise_alpha else 3,
        dataset_config=dataset_config,
        conditional_pose=args.conditional_pose,
        use_encoder=args.use_encoder,
        num_classes=train_split.num_classes if args.use_class else None,
    ).to(device)
discriminator_list = [discriminator]
if args.dual_discriminator:
    if args.use_encoder:
        # Instantiate another discriminator
        discriminator2 = discriminator.Discriminator(
            args.resolution,
            nc=4 if supervise_alpha else 3,
            dataset_config=dataset_config,
            conditional_pose=args.conditional_pose,
            num_classes=train_split.num_classes if args.use_class else None,
            use_encoder=False).to(device)
    else:
        discriminator2 = discriminator
    discriminator_list.append(discriminator2)


class ParallelModel(nn.Module):

    def __init__(self, resolution, model=None, model_ema=None, lpips_net=None):
        super().__init__()
        self.resolution = resolution
        self.model = model
        self.model_ema = model_ema
        self.lpips_net = lpips_net

    def forward(self,
                tform_cam2world,
                focal,
                center,
                bbox,
                c,
                use_ema=False,
                ray_multiplier=1,
                res_multiplier=1,
                pretrain_sdf=False,
                compute_normals=False,
                compute_semantics=False,
                compute_coords=False,
                encoder_output=False,
                closure=None,
                closure_params=None,
                extra_model_outputs=[],
                extra_model_inputs={},
                force_no_cam_grad=False):
        model_to_use = self.model_ema if use_ema else self.model
        if pretrain_sdf:
            return model_to_use(
                None,
                c,
                request_model_outputs=['sdf_distance_loss', 'sdf_eikonal_loss'])
        if encoder_output:
            return model_to_use.emb(c)

        output = render(model_to_use,
                        int(self.resolution * res_multiplier),
                        int(self.resolution * res_multiplier),
                        tform_cam2world,
                        focal,
                        center,
                        bbox,
                        c,
                        depth_samples_per_ray * ray_multiplier,
                        compute_normals=compute_normals,
                        compute_semantics=compute_semantics,
                        compute_coords=compute_coords,
                        extra_model_outputs=extra_model_outputs,
                        extra_model_inputs=extra_model_inputs,
                        force_no_cam_grad=force_no_cam_grad)
        if closure is not None:
            return closure(
                self, output[0], output[2], output[4], output[-1],
                **closure_params)  # RGB, alpha, semantics, extra_outptus
        else:
            return output


if args.use_encoder or args.run_inversion:
    loss_fn_lpips = metrics.LPIPSLoss().to(device)
else:
    loss_fn_lpips = None

if args.run_inversion:
    model = None
else:
    model = create_model()

model_ema = create_model()
model_ema.eval()
model_ema.requires_grad_(False)
if model is not None:
    model_ema.load_state_dict(model.state_dict())

parallel_model = nn.DataParallel(
    ParallelModel(args.resolution,
                  model=model,
                  model_ema=model_ema,
                  lpips_net=loss_fn_lpips), gpu_ids).to(device)
parallel_discriminator_list = [
    nn.DataParallel(d, gpu_ids) if d is not None else None
    for d in discriminator_list
]

total_params = 0
for param in model_ema.parameters():
    total_params += param.numel()
print('Params G:', total_params / 1000000, 'M')

for d_idx, d in enumerate(discriminator_list):
    if d is None:
        print(f'Params D_{d_idx}: none')
    else:
        total_params = 0
        for param in d.parameters():
            total_params += param.numel()
        print(f'Params D_{d_idx}:', total_params / 1000000, 'M')

criterion = GANLoss()

blur_warmup_iters = 12500
lr_g = args.lr_g
lr_d = args.lr_d
lr_warmup_iters = 2000
lr_warmup = True
if lr_warmup and resume_from is None:
    lr_d_target = lr_d
    lr_d /= 10
    lr_d_delta = (lr_d_target - lr_d) / (lr_warmup_iters / 2)

    lr_g_target = lr_g
    lr_g /= 10
    lr_g_delta = (lr_g_target - lr_g) / (lr_warmup_iters / 2)
else:
    lr_warmup = False  # Override if resuming from checkpoint
print('Effective G lr:', lr_g)
print('Effective D lr:', lr_d)

if model is not None:
    g_params = list(model.parameters())
    optimizer_g = torch.optim.Adam(g_params, lr=lr_g, betas=(0., 0.99))

    d_params = list(
        discriminator.parameters()) if discriminator is not None else []
    if args.use_encoder and args.dual_discriminator:
        d_params += list(discriminator2.parameters())
    optimizer_d = torch.optim.Adam(d_params, lr=lr_d, betas=(0., 0.99))
else:
    optimizer_g = None
    optimizer_d = None

# These are set to True dynamically during runtime to optimize memory usage
[d.requires_grad_(False) for d in discriminator_list if d is not None]
if model is not None:
    model.requires_grad_(False)

torch.manual_seed(random_seed)
np.random.seed(random_seed)
rng = np.random.RandomState(random_seed)
train_sampler = utils.EndlessSampler(len(train_split.images), rng)

# Seed CUDA RNGs separately
seed_generator = np.random.RandomState(random_seed)
for device_id in gpu_ids:
    with torch.cuda.device(device_id):
        gpu_seed = int.from_bytes(np.random.bytes(4), 'little', signed=False)
        torch.cuda.manual_seed(gpu_seed)

g_curve = []
d_real_curve = []
d_fake_curve = []
i = 0
best_fid = 1000
is_best = False
augment_p_effective = 0.
ppl_running_avg = None


def augment_impl(img, pose, focal, p, disable_scale=False, cached_tform=None):
    bs = img.shape[0] if img is not None else pose.shape[0]
    device = img.device if img is not None else pose.device

    if cached_tform is None:
        rot = (torch.rand((bs,), device=device) - 0.5) * 2 * np.pi
        rot = rot * (torch.rand((bs,), device=device) < p).float()

        if disable_scale:
            scale = torch.ones((bs,), device=device)
        else:
            scale = torch.exp2(torch.randn((bs,), device=device) * 0.2)
            scale = torch.lerp(torch.ones_like(scale), scale, (torch.rand(
                (bs,), device=device) < p).float())

        translation = torch.randn((bs, 2), device=device) * 0.1
        translation = torch.lerp(torch.zeros_like(translation), translation,
                                 (torch.rand(
                                     (bs, 1), device=device) < p).float())

        cached_tform = rot, scale, translation
    else:
        rot, scale, translation = cached_tform

    mat = torch.zeros((bs, 2, 3), device=device)
    mat[:, 0, 0] = torch.cos(rot)
    mat[:, 0, 1] = -torch.sin(rot)
    mat[:, 0, 2] = translation[:, 0]
    mat[:, 1, 0] = torch.sin(rot)
    mat[:, 1, 1] = torch.cos(rot)
    mat[:, 1, 2] = -translation[:, 1]
    if img is not None:
        mat_scaled = mat.clone()
        mat_scaled *= scale[:, None, None]
        mat_scaled[:, :, 2] = torch.sum(mat[:, :2, :2] *
                                        mat_scaled[:, :, 2].unsqueeze(-2),
                                        dim=-1)
        grid = F.affine_grid(mat_scaled, img.shape, align_corners=False)
        if dataset_config['white_background']:
            assert not args.supervise_alpha
            img = img - 1  # Adjustment for white background
        img_transformed = F.grid_sample(img,
                                        grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)
        if dataset_config['white_background']:
            img_transformed = img_transformed + 1  # Adjustment for white background
    else:
        img_transformed = None

    if pose is not None:
        M = torch.eye(4, device=device).unsqueeze(0).expand(mat.shape[0], 4,
                                                            4).contiguous()
        M[:, :2, :2] = mat[:, :2, :2]
        if focal is not None:
            focal = focal / scale
        pose = pose @ M.transpose(-2, -1)
        if focal is None:
            pose[:, :3, :3] *= scale[:, None, None]
            pose[:, 3:4, 3:4] *= scale[:, None, None]

        # Apply translation
        pose_orig = pose
        cam_inverted = pose_utils.invert_space(pose)
        if focal is not None:
            cam_inverted[:, :2, 3] -= translation * (-cam_inverted[:, 2:3, 3] /
                                                     (2 * focal[:, None]))
        else:
            cam_inverted[:, :2, 3] -= translation * pose_orig[:, 3:4, 3]
        pose = pose_utils.invert_space(cam_inverted)
        if focal is None:
            pose[:, :3, :3] *= pose_orig[:, 3:4, 3:4]
            pose[:, 3, 3] *= pose_orig[:, 3, 3]

    return img_transformed, pose, focal, cached_tform


def augment(img,
            pose,
            focal,
            p,
            disable_scale=False,
            cached_tform=None,
            return_tform=False):
    if p == 0 and cached_tform is None:
        return img, pose, focal

    assert img is None or pose is None or img.shape[0] == pose.shape[0]

    # Standard augmentation
    img_new, pose_new, focal_new, tform = augment_impl(img, pose, focal, p,
                                                       disable_scale,
                                                       cached_tform)

    if return_tform:
        return img_new, pose_new, focal_new, tform
    else:
        return img_new, pose_new, focal_new


if args.use_sdf and resume_from is None:
    print('SDF pre-training...')

    def pretrain_sdf():
        model.requires_grad_(True)
        optimizer_pretrain = torch.optim.Adam(
            model.parameters(), lr=args.lr_g)  # Original lr without warmup
        for i in range(1000):
            z_random = torch.randn((batch_size, args.latent_dim), device=device)

            if args.use_encoder:
                target_img_idx = train_sampler(batch_size)
                target_img = train_split[target_img_idx].images
                z_image = target_img.permute(0, 3, 1, 2)[:, :3]
                z = (z_random, z_image)
            elif args.use_class:
                cl = torch.randint(train_split.num_classes,
                                   size=(batch_size,)).to(device)
                z = (z_random, cl)
            else:
                z = z_random

            losses = parallel_model(None,
                                    None,
                                    None,
                                    None,
                                    c=z,
                                    pretrain_sdf=True)

            loss_dist = losses['sdf_distance_loss'].mean()
            loss = loss_dist.clone()
            eikonal_loss = losses['sdf_eikonal_loss'].mean()
            loss += args.eikonal * eikonal_loss
            loss.backward()
            if i % 100 == 0:
                print(
                    'dist',
                    loss_dist.item(),
                    'eik',
                    eikonal_loss.item(),
                )
            optimizer_pretrain.step()
            optimizer_pretrain.zero_grad(set_to_none=True)
        print('SDF pre-training done.')
        model.requires_grad_(False)
        model_ema.load_state_dict(model.state_dict())

    pretrain_sdf()
    torch.cuda.empty_cache()
    i = 0

if resume_from is not None:
    print('Loading specified checkpoint...')
    if model is not None and 'model' in resume_from:
        model.load_state_dict(resume_from['model'])
    model_ema.load_state_dict(resume_from['model_ema'])
    if not args.run_inversion:
        assert (discriminator is not None) == ('discriminator' in resume_from)
        if discriminator is not None:
            discriminator.load_state_dict(resume_from['discriminator'])
        if args.use_encoder and args.dual_discriminator:
            discriminator2.load_state_dict(resume_from['discriminator2'])
        optimizer_g.load_state_dict(resume_from['optimizer_g'])
        optimizer_d.load_state_dict(resume_from['optimizer_d'])
    if 'iteration' in resume_from:
        i = resume_from['iteration']
        print('Resuming from iteration', i)
    else:
        i = args.iterations

    if 'random_state' in resume_from:
        print('Restoring RNG state...')
        utils.restore_random_state(resume_from['random_state'], train_sampler,
                                   rng, gpu_ids)

    if 'lr_g' in resume_from:
        lr_g = resume_from['lr_g']
    if 'lr_d' in resume_from:
        lr_d = resume_from['lr_d']
    if 'best_fid' in resume_from:
        best_fid = resume_from['best_fid']
    if 'augment_p_effective' in resume_from:
        augment_p_effective = resume_from['augment_p']
    if args.path_length_regularization and 'ppl_running_avg' in resume_from:
        ppl_running_avg = resume_from['ppl_running_avg']


def sample_batch(batch_size, discriminator_idx, train_sampler=None):
    if train_sampler is not None:
        target_img_idx = train_sampler(batch_size)
    else:
        target_img_idx = rng.randint(train_split.images.shape[0],
                                     size=(batch_size,))
    target_img = train_split[target_img_idx, ..., :nc].images
    target_tform_cam2world = train_split[target_img_idx].tform_cam2world
    target_focal = train_split[target_img_idx].focal_length
    target_center = train_split[target_img_idx].center
    target_bbox = train_split[target_img_idx].bbox
    z_random = torch.randn((batch_size, args.latent_dim), device=device)
    z = z_image = None
    if args.use_encoder:
        z_image = target_img.permute(0, 3, 1, 2)[:, :3]
        z = (z_random, z_image)
    elif args.use_class:
        target_class = train_split[target_img_idx].classes
        z = (z_random, target_class)
        z_image = target_class
    else:
        z = z_random
        z_image = None

    no_augment = (args.dual_discriminator_l1 or
                  args.dual_discriminator_mse) and discriminator_idx == 0
    if args.augment_p > 0 and not no_augment:
        if dataset_config['is_highres']:
            target_img = train_split[target_img_idx, ..., :nc].images_highres
        target_img, target_tform_cam2world, target_focal = augment(
            target_img.permute(0, 3, 1, 2), target_tform_cam2world,
            target_focal, augment_p_effective)
        if dataset_config['is_highres']:
            target_img = F.avg_pool2d(target_img, 2)  # Anti-alias
        target_img = target_img.permute(0, 2, 3, 1)

    return target_img_idx, target_img, target_tform_cam2world, target_focal, target_center, target_bbox, z, z_image


while i < num_iters:
    nc = 4 if supervise_alpha else 3

    if not args.augment_ada:
        augment_p_effective = args.augment_p  # Fixed value

    unconditional_discriminator_idx = 1 if args.dual_discriminator else 0
    if i % 2 == 0:
        t1 = time.time()
        # G loop
        optimizer_g.zero_grad()
        model.requires_grad_(True)
        target_img_idx, target_img, target_tform_cam2world, target_focal, target_center, target_bbox, z, z_image = sample_batch(
            batch_size, 0)
        for discriminator_idx, target_discriminator in enumerate(
                parallel_discriminator_list):
            if discriminator_idx > 0:
                # Sample another pose
                target_img_idx = target_img = None
                z_image = None  # Second discriminator is always unconditional
                _, _, target_tform_cam2world, target_focal, target_center, target_bbox, _, _ = sample_batch(
                    batch_size, discriminator_idx)

            request_model_outputs = []
            if discriminator_idx == 0:
                if args.path_length_regularization:
                    request_model_outputs.append('path_length')
                if args.use_sdf:
                    request_model_outputs.append('sdf_eikonal_loss')
                if use_tv:
                    request_model_outputs.append('total_variation_loss')
                if use_entropy:
                    request_model_outputs.append('entropy_loss')
            rgb_predicted, depth_predicted, acc_predicted, normal_map, semantic_map, extra_model_outputs = parallel_model(
                target_tform_cam2world,
                target_focal,
                target_center,
                target_bbox,
                z,
                extra_model_outputs=request_model_outputs)
            if supervise_alpha and target_discriminator is not None:
                rgb_predicted = torch.cat(
                    (rgb_predicted, acc_predicted.unsqueeze(-1)), dim=-1)
            img_batch = rgb_predicted.permute(0, 3, 1, 2)

            if target_discriminator is None:
                # Use L1 or MSE
                loss_fn = F.mse_loss if args.dual_discriminator_mse else F.l1_loss
                # z_image is not blurred by default
                loss = loss_fn(
                    img_batch,
                    ops.blur(z_image, i, blur_warmup_iters,
                             dataset_config['white_background'])) * 10
            else:
                if discriminator_idx > 0:
                    z = z_image = None
                discriminated = target_discriminator(img_batch, i,
                                                     target_tform_cam2world,
                                                     z_image, target_focal)
                loss = criterion(discriminated, True)
            g_loss = loss.item()
            if args.dual_discriminator:
                loss = loss / 2  # Rescale

            if discriminator_idx == 0:
                if args.use_sdf:
                    eikonal_loss = extra_model_outputs['sdf_eikonal_loss'].mean(
                    )
                    loss += args.eikonal * eikonal_loss
                if use_tv:
                    tv_loss = extra_model_outputs['total_variation_loss'].mean()
                    tv_warmup_mul = min(i / blur_warmup_iters, 1.)
                    loss += (args.tv * tv_warmup_mul) * tv_loss
                    if writer is not None:
                        writer.add_scalar('loss/tv', tv_loss.item(), i)
                if use_entropy:
                    entropy_loss = extra_model_outputs['entropy_loss'].mean()
                    entropy_warmup_mul = min(i / blur_warmup_iters, 1.)
                    loss += (args.entropy * entropy_warmup_mul) * entropy_loss
                    if writer is not None:
                        writer.add_scalar('loss/entropy', entropy_loss.item(),
                                          i)
                if args.path_length_regularization:
                    ppl = extra_model_outputs['path_length']
                    pl_decay = 0.01
                    pl_weight = 2.
                    pl_weight = pl_weight * min(i / lr_warmup_iters, 1.)
                    if ppl_running_avg is None:
                        ppl_running_avg = ppl.mean().item()
                    pl_mean = ppl_running_avg * (
                        1 - pl_decay) + ppl.mean() * pl_decay
                    ppl_running_avg = pl_mean.item()
                    ppl_loss = (ppl - ppl_running_avg).square().mean()
                    loss += pl_weight * ppl_loss
                    if writer is not None:
                        writer.add_scalar('ppl/loss', ppl_loss.item(), i)
                        writer.add_scalar('ppl/running_avg', ppl_running_avg, i)
            loss.backward()
            if discriminator_idx == unconditional_discriminator_idx:
                g_curve.append(g_loss)
            if args.dual_discriminator and discriminator_idx == 0:
                g_suffix = '_cond'
            else:
                g_suffix = ''
            if writer is not None:
                writer.add_scalar(f'loss/g{g_suffix}', g_loss, i)
                if discriminator_idx == 0:
                    if args.use_sdf:
                        writer.add_scalar('loss/eikonal', eikonal_loss.item(),
                                          i)
                        writer.add_scalar('sdf/beta', model.beta.data.item(), i)
                        writer.add_scalar('sdf/beta_ema',
                                          model_ema.beta.data.item(), i)
                        writer.add_scalar('sdf/alpha', model.alpha.data.item(),
                                          i)
                        writer.add_scalar('sdf/alpha_ema',
                                          model_ema.alpha.data.item(), i)
        grad_norm_g = nn.utils.clip_grad_norm_(g_params,
                                               args.clip_gradient_norm).item()
        if writer is not None:
            writer.add_scalar('grad_norm/g', grad_norm_g, i)
        optimizer_g.step()
        if args.use_sdf:
            model.beta.data.clamp_(min=1e-3)
            model.alpha.data.clamp_(min=1e-3)
        model.requires_grad_(False)
        update_generator_ema(i)
    else:
        # D loop
        optimizer_d.zero_grad()
        [d.requires_grad_(True) for d in discriminator_list if d is not None]
        target_img_idx, target_img, target_tform_cam2world, target_focal, target_center, target_bbox, z, z_image = sample_batch(
            batch_size, 0, train_sampler)
        for discriminator_idx, target_discriminator in enumerate(
                parallel_discriminator_list):
            if target_discriminator is None:
                continue
            if discriminator_idx > 0:
                # Sample another pose
                target_img_idx, target_img, target_tform_cam2world, target_focal, target_center, target_bbox, _, _ = sample_batch(
                    batch_size, discriminator_idx)

            # "Real" phase
            target_img = ops.blur(target_img.permute(0, 3, 1,
                                                     2), i, blur_warmup_iters,
                                  dataset_config['white_background']).permute(
                                      0, 2, 3, 1)
            if use_r1 and i % 2 == 1:
                target_img = target_img.requires_grad_()
            target_img_disc = target_img.permute(0, 3, 1, 2)
            discriminated_real = target_discriminator(target_img_disc, i,
                                                      target_tform_cam2world,
                                                      z_image, target_focal)
            if use_r1 and i % 2 == 1:
                d_grad_real, = torch.autograd.grad(discriminated_real.sum(),
                                                   target_img,
                                                   create_graph=True)
                grad_penalty = d_grad_real.contiguous().view(
                    d_grad_real.shape[0], -1).square().sum(dim=1).mean()
            else:
                grad_penalty = 0

            loss_real = criterion(discriminated_real, True)
            (loss_real + (args.r1 / 2) * grad_penalty).backward()

            # Sample fresh poses for the "fake" phase
            if discriminator_idx == 0:
                target_img_idx, target_img, target_tform_cam2world, target_focal, target_center, target_bbox, z, z_image = sample_batch(
                    batch_size, discriminator_idx)
            else:
                target_img_idx, target_img, target_tform_cam2world, target_focal, target_center, target_bbox, _, _ = sample_batch(
                    batch_size, discriminator_idx)

            # "Fake" phase
            with torch.no_grad():
                rgb_predicted, depth_predicted, acc_predicted, normal_map, semantic_map, _ = parallel_model(
                    target_tform_cam2world, target_focal, target_center,
                    target_bbox, z)

            if supervise_alpha:
                rgb_predicted = torch.cat(
                    (rgb_predicted, acc_predicted.unsqueeze(-1)), dim=-1)

            target_img_disc_fake = rgb_predicted.permute(0, 3, 1, 2)

            if discriminator_idx > 0:
                z = z_image = None
            discriminated_fake = target_discriminator(target_img_disc_fake, i,
                                                      target_tform_cam2world,
                                                      z_image, target_focal)
            loss_fake = criterion(discriminated_fake, False)

            loss_fake.backward()

            if args.dual_discriminator and discriminator_idx == 0:
                is_conditional_discriminator = True
                d_suffix = '_cond'
            else:
                is_conditional_discriminator = False
                d_suffix = ''
            ada_interval = 4
            if i % (2 * ada_interval) == 2 * ada_interval - 1:
                sign_real = discriminated_real.detach().sign().mean()
                if writer is not None:
                    writer.add_scalar(f'augment/sign_real{d_suffix}',
                                      sign_real.item(), i)
                if args.augment_ada and discriminator_idx == unconditional_discriminator_idx:
                    if i < blur_warmup_iters:
                        # Force zero initially
                        augment_p_effective = 0
                    else:
                        ada_rampup = 500000
                        ada_delta = torch.sign(sign_real - args.ada_target) * (
                            batch_size * ada_interval) / ada_rampup
                        augment_p_effective = min(
                            max(augment_p_effective + ada_delta.item(), 0.),
                            args.augment_p)  # Clamp between 0 and augment_p
            if discriminator_idx == unconditional_discriminator_idx:
                d_real_curve.append(loss_real.item())
                d_fake_curve.append(loss_fake.item())
            if writer is not None:
                writer.add_scalar(f'loss/d_real{d_suffix}', loss_real.item(), i)
                writer.add_scalar(f'loss/d_fake{d_suffix}', loss_fake.item(), i)
                writer.add_scalar('augment/p', augment_p_effective, i)
                if use_r1 and i % 2 == 1:
                    writer.add_scalar(f'loss/r1{d_suffix}', grad_penalty.item(),
                                      i)
        grad_norm_d = nn.utils.clip_grad_norm_(d_params,
                                               args.clip_gradient_norm).item()
        if writer is not None:
            writer.add_scalar('grad_norm/d', grad_norm_d, i)
        optimizer_d.step()
        [d.requires_grad_(False) for d in discriminator_list if d is not None]

        if lr_warmup:
            if lr_d < lr_d_target:
                for param_group in optimizer_d.param_groups:
                    param_group['lr'] += lr_d_delta
                lr_d += lr_d_delta
                for param_group in optimizer_g.param_groups:
                    param_group['lr'] += lr_g_delta
                lr_g += lr_g_delta
            else:
                lr_warmup = False  # Turn off

        t2 = time.time()
        elapsed = batch_size / (t2 - t1)

    # We let training run for a few iterations in order to
    # diagnose early OOM errors.
    if i == 3 or (i + 1) % display_every == 0:
        print(f'[{i}] im/s', elapsed)

        def evaluate(z_fid, recon_mode=None, use_testset=False):
            target_split = test_split if use_testset else train_eval_split

            target_img_idx = target_split.eval_indices
            if recon_mode == 'random':
                target_img_idx_ = target_split.eval_indices_perm
            else:
                target_img_idx_ = target_img_idx
            target_tform_cam2world = target_split[
                target_img_idx_].tform_cam2world
            target_focal = target_split[target_img_idx_].focal_length
            target_center = target_split[target_img_idx_].center
            target_bbox = target_split[target_img_idx_].bbox
            if args.use_class:
                target_class = target_split[target_img_idx_].classes
            else:
                target_class = None

            all_activations = []
            test_bs = batch_size
            idx = 0

            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            total_mask_iou = 0
            while idx < z_fid.shape[0]:
                if test_bs != 1 and target_tform_cam2world[
                        idx:idx + test_bs].shape[0] < test_bs:
                    test_bs = 1

                if test_bs == 1:
                    test_model = parallel_model.module
                else:
                    test_model = parallel_model

                if args.use_encoder:
                    # Use non-cropped images (same as training)
                    images_in = test_split.images if use_testset else train_split.images
                    z_input = images_in[target_img_idx[idx:idx +
                                                       test_bs]].permute(
                                                           0, 3, 1,
                                                           2).to(device)[:, :3]
                    z_input = (z_fid[idx:idx + test_bs], z_input)
                elif args.use_class:
                    z_input = (z_fid[idx:idx + test_bs],
                               target_class[idx:idx + test_bs])
                else:
                    z_input = z_fid[idx:idx + test_bs]

                rgb_predicted_fid, depth_predicted_fid, acc_predicted_fid, normal_map_fid, semantic_map_fid, _ = test_model(
                    target_tform_cam2world[idx:idx + test_bs],
                    target_focal[idx:idx +
                                 test_bs] if target_focal is not None else None,
                    target_center[idx:idx + test_bs]
                    if target_center is not None else None,
                    target_bbox[idx:idx +
                                test_bs] if target_bbox is not None else None,
                    z_input,
                    use_ema=True,
                    ray_multiplier=1,
                    res_multiplier=1,
                    force_no_cam_grad=True,
                    compute_normals=args.use_sdf and idx == 0,
                    compute_semantics=args.attention_values > 0 and idx == 0)
                rgb_predicted_fid.clamp_(-1, 1)
                rgb_predicted_fid = rgb_predicted_fid.detach().permute(
                    0, 3, 1, 2) / 2 + 0.5

                views_per_object = dataset_config[
                    'views_per_object_test'] if use_testset else dataset_config[
                        'views_per_object']
                if recon_mode == 'front' or (recon_mode == 'random' and
                                             views_per_object is not None and
                                             views_per_object > 1):
                    eval_img = target_split[
                        target_img_idx_[idx:idx + test_bs]].images.permute(
                            0, 3, 1, 2)[:, :3] / 2 + 0.5  # Cropped
                    total_psnr += metrics.psnr(
                        rgb_predicted_fid, eval_img).item(
                        ) * rgb_predicted_fid.shape[0]  # Batch sum
                    total_ssim += metrics.ssim(
                        rgb_predicted_fid, eval_img).item(
                        ) * rgb_predicted_fid.shape[0]  # Batch sum
                    total_lpips += loss_fn_lpips(
                        rgb_predicted_fid, eval_img, normalize=True).mean(
                        ).item() * rgb_predicted_fid.shape[0]  # Batch sum
                    if dataset_config['has_mask']:
                        eval_img_alpha = target_split[target_img_idx_[idx:idx +
                                                                      test_bs],
                                                      ..., 3].images  # Cropped
                        total_mask_iou += metrics.iou(
                            acc_predicted_fid, eval_img_alpha).item(
                            ) * rgb_predicted_fid.shape[0]  # Batch sum
                else:
                    eval_img = None

                if idx == 0 and writer is not None:
                    suffix = '_untrunc'
                    recon_suffix = '_test' if use_testset else ''
                    if recon_mode is None:
                        prefix = 'gen'
                    else:
                        prefix = f'recon_{recon_mode}'
                        if eval_img is not None:
                            writer.add_images(
                                f'img_eval_{prefix}/ref{suffix}{recon_suffix}',
                                eval_img.cpu(), i)

                    writer.add_images(
                        f'img_eval_{prefix}/static{suffix}{recon_suffix}',
                        rgb_predicted_fid.cpu(), i)
                    if args.use_sdf and normal_map_fid is not None:
                        writer.add_images(
                            f'img_eval_{prefix}/static_normals{suffix}{recon_suffix}',
                            normal_map_fid.permute(0, 3, 1, 2).cpu() / 2 + 0.5,
                            i)
                    if args.attention_values > 0 and semantic_map_fid is not None:
                        semantic_map_fid = semantic_map_fid @ color_palette
                        writer.add_images(
                            f'img_eval_{prefix}/static_semantics{suffix}{recon_suffix}',
                            semantic_map_fid.permute(0, 3, 1, 2).cpu() / 2 +
                            0.5, i)

                assert rgb_predicted_fid.shape[-1] == evaluation_res
                if not use_testset:
                    all_activations.append(
                        fid.forward_inception_batch(inception_net,
                                                    rgb_predicted_fid))
                idx += test_bs

            if eval_img is not None:
                recon_suffix = '_test' if use_testset else ''
                total_psnr /= z_fid.shape[0]
                print(f'Recon PSNR{recon_suffix} {recon_mode}', total_psnr)
                total_ssim /= z_fid.shape[0]
                print(f'Recon SSIM{recon_suffix} {recon_mode}', total_ssim)
                total_lpips /= z_fid.shape[0]
                print(f'Recon LPIPS{recon_suffix} {recon_mode}', total_lpips)
                if dataset_config['has_mask']:
                    total_mask_iou /= z_fid.shape[0]
                    print(f'Recon IoU{recon_suffix} {recon_mode}',
                          total_mask_iou)
                if writer is not None:
                    writer.add_scalar(
                        f'reconstruction/psnr{recon_suffix}_{recon_mode}',
                        total_psnr, i)
                    writer.add_scalar(
                        f'reconstruction/ssim{recon_suffix}_{recon_mode}',
                        total_ssim, i)
                    writer.add_scalar(
                        f'reconstruction/lpips{recon_suffix}_{recon_mode}',
                        total_lpips, i)
                    if dataset_config['has_mask']:
                        writer.add_scalar(
                            f'reconstruction/iou{recon_suffix}_{recon_mode}',
                            total_mask_iou, i)

            if use_testset:
                return None
            all_activations = np.concatenate(all_activations, axis=0)
            assert len(all_activations) == z_fid.shape[0]
            fid_stats = fid.calculate_stats(all_activations)
            if use_testset:
                return fid.calculate_frechet_distance(*fid_stats,
                                                      *test_split.fid_stats)
            else:
                return fid.calculate_frechet_distance(
                    *fid_stats, *train_eval_split.fid_stats)

        fid_untrunc = None
        if not args.use_encoder:
            fid_untrunc = evaluate(z_fid_untrunc)
            print('FID generation: {:.02f}'.format(fid_untrunc))
            if writer is not None:
                writer.add_scalar('generation/fid_untrunc', fid_untrunc, i)
            current_fid = fid_untrunc  # For best fid

        if args.use_encoder:
            fid_recon_frontview = evaluate(z_fid_untrunc, recon_mode='front')
            print('FID reconstruction front view: {:.02f}'.format(
                fid_recon_frontview))
            if writer is not None:
                writer.add_scalar('reconstruction/fid_frontview',
                                  fid_recon_frontview, i)
            fid_recon_randomview = evaluate(z_fid_untrunc, recon_mode='random')
            print('FID reconstruction random view: {:.02f}'.format(
                fid_recon_randomview))
            if writer is not None:
                writer.add_scalar('reconstruction/fid_randomview',
                                  fid_recon_randomview, i)
            if args.use_encoder:
                current_fid = fid_recon_randomview
                fid_untrunc = fid_recon_randomview

            if dataset_config['views_per_object_test']:
                evaluate(z_fid_untrunc_test,
                         recon_mode='front',
                         use_testset=True)
                evaluate(z_fid_untrunc_test,
                         recon_mode='random',
                         use_testset=True)

        if current_fid < best_fid:
            best_fid = current_fid
            is_best = True

        if writer is not None:
            writer.add_scalar('img_per_sec', batch_size / (t2 - t1), i)
            writer.add_images(
                'img/ref',
                target_img[..., :3].permute(0, 3, 1, 2).cpu() / 2 + 0.5, i)
            writer.add_images(
                'img/rgb',
                rgb_predicted[..., :3].detach().permute(0, 3, 1, 2).cpu() / 2 +
                0.5, i)
            writer.add_images(
                'img/depth',
                (depth_predicted /
                 depth_predicted.max()).detach().cpu().unsqueeze(1), i)
            if args.use_sdf and normal_map is not None:
                writer.add_images(
                    'img/normals',
                    normal_map.detach().permute(0, 3, 1, 2).cpu() / 2 + 0.5, i)
            writer.add_images(
                'img/mask',
                acc_predicted.detach().cpu().unsqueeze(1).clamp(0, 1), i)

            z = torch.randn((batch_size, args.latent_dim), device=device)

            target_img_idx = np.random.randint(train_split.images.shape[0],
                                               size=(batch_size,))
            target_img_inf = train_split[target_img_idx].images
            target_tform_cam2world = train_split[target_img_idx].tform_cam2world
            target_focal = train_split[target_img_idx].focal_length
            target_center = train_split[target_img_idx].center
            target_bbox = train_split[target_img_idx].bbox
            if args.use_encoder:
                z_image = target_img_inf.to(device).permute(0, 3, 1, 2)[:, :3]
                z = (z, z_image)
            elif args.use_class:
                target_class = train_split[target_img_idx].classes
                z = (z, target_class)

            rgb_predicted_inf, depth_predicted_inf, _, normal_map_inf, semantic_map_inf, _ = parallel_model(
                target_tform_cam2world,
                target_focal,
                target_center,
                target_bbox,
                z,
                use_ema=True,
                ray_multiplier=1,
                force_no_cam_grad=True,
                compute_normals=args.use_sdf,
                compute_semantics=args.attention_values > 0)
            rgb_predicted_inf.clamp_(-1, 1)
            writer.add_images(
                'img/inference',
                rgb_predicted_inf.permute(0, 3, 1, 2).cpu() / 2 + 0.5, i)
            if args.use_sdf and normal_map_inf is not None:
                writer.add_images(
                    'img/inference_normals',
                    normal_map_inf.permute(0, 3, 1, 2).cpu() / 2 + 0.5, i)
            if args.use_encoder:
                writer.add_images(
                    'img/inference_ref',
                    target_img_inf[..., :3].permute(0, 3, 1, 2).cpu() / 2 + 0.5,
                    i)
            if args.attention_values > 0 and semantic_map_inf is not None:
                semantic_map_inf = semantic_map_inf @ color_palette
                writer.add_images(
                    'img/inference_semantics',
                    semantic_map_inf.permute(0, 3, 1, 2).cpu() / 2 + 0.5, i)

        # Save checkpoint
        def save_checkpoint(label):
            out_dict = {
                'model':
                    model.state_dict(),
                'model_ema':
                    model_ema.state_dict(),
                'optimizer_g':
                    optimizer_g.state_dict(),
                'optimizer_d':
                    optimizer_d.state_dict(),
                'iteration':
                    i + 1,
                'random_state':
                    utils.save_random_state(train_sampler, rng, gpu_ids),
                'lr_g':
                    lr_g,
                'lr_d':
                    lr_d,
                'best_fid':
                    best_fid,
                'fid_untrunc':
                    fid_untrunc,
                'augment_p':
                    augment_p_effective,
            }
            if discriminator is not None:
                out_dict['discriminator'] = discriminator.state_dict()
            if args.use_encoder and args.dual_discriminator:
                out_dict['discriminator2'] = discriminator2.state_dict()
            if args.path_length_regularization:
                out_dict['ppl_running_avg'] = ppl_running_avg
            with utils.open_file(
                    os.path.join(checkpoint_dir, f'checkpoint_{label}.pth'),
                    'wb') as f:
                torch.save(out_dict, f)

        save_checkpoint('latest')
        if is_best:
            save_checkpoint('best')
            is_best = False
        if (i + 1) % 50000 == 0:  # Save every 50k steps
            save_checkpoint(f'{i+1}')

    i += 1


def train_coord_regressor(writer):
    regress_pose = True
    regress_latent = True
    regress_separate = args.inv_use_separate

    batch_size = args.batch_size
    latent_str = ''
    if regress_separate:
        assert regress_pose and regress_latent
        latent_str += '_separate'

    cfg_xid = f'_{args.xid}' if len(args.xid) > 0 else ''
    coord_regressor_experiment_name = f'c{cfg_xid}'
    coord_regressor_experiment_name += f'{latent_str}'
    coord_regressor_experiment_name += f'_it{resume_from["iteration"]}'

    coord_regressor_checkpoint_dir = os.path.join(args.root_path,
                                                  'coords_checkpoints',
                                                  args.resume_from)

    print('Experiment name', coord_regressor_experiment_name)
    utils.mkdir(coord_regressor_checkpoint_dir)
    coord_regressor_checkpoint_path = os.path.join(
        coord_regressor_checkpoint_dir, coord_regressor_experiment_name)
    print('Saving to', coord_regressor_checkpoint_path + '_latest.pth')

    # Train from scratch
    coord_regressor = encoder.BootstrapEncoder(
        args.latent_dim,
        pose_regressor=regress_pose,
        latent_regressor=regress_latent,
        separate_backbones=regress_separate,
        pretrained_model_path=os.path.join(args.root_path,
                                           'coords_checkpoints'),
        pretrained=True,
    ).to(device)

    coord_regressor = nn.DataParallel(coord_regressor, gpu_ids)
    coord_regressor.requires_grad_(True)

    lr = 0.00006
    optimizer_coord = torch.optim.Adam(coord_regressor.parameters(), lr=lr)
    loss_curve_coord = []
    i = 0

    def save_checkpoint(suffix=None):
        if suffix is None:
            suffix = '_latest'
        with utils.open_file(f'{coord_regressor_checkpoint_path}{suffix}.pth',
                             'wb') as f:
            torch.save(
                {
                    'optimizer_coord':
                        optimizer_coord.state_dict(),
                    'model_coord':
                        coord_regressor.state_dict(),
                    'iteration':
                        i,
                    'lr':
                        lr,
                    'random_state':
                        utils.save_random_state(train_sampler, rng, gpu_ids),
                }, f)

    # Resume if cache is available
    if utils.file_exists(coord_regressor_checkpoint_path + '_latest.pth'):
        print(
            f'Restoring checkpoint from {coord_regressor_checkpoint_path}_latest.pth'
        )
        with utils.open_file(coord_regressor_checkpoint_path + '_latest.pth',
                             'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
            coord_regressor.load_state_dict(checkpoint['model_coord'])
            if 'optimizer_coord' in checkpoint:
                optimizer_coord.load_state_dict(checkpoint['optimizer_coord'])
            lr = checkpoint['lr']
            for param_group in optimizer_coord.param_groups:
                param_group['lr'] = lr
            i = checkpoint['iteration']
            if 'random_state' in checkpoint:
                utils.restore_random_state(checkpoint['random_state'],
                                           train_sampler, rng, gpu_ids)
        print(f'Resuming from iteration {i}...')

    def criterion_coords(pred, target, mask):
        # Assume channel last!
        assert pred.shape[-1] == 3
        return (pred - target).norm(dim=-1).mul(mask).mean()

    criterion_mask = nn.L1Loss()
    criterion_latent = nn.MSELoss()

    max_iters = 120000
    evaluate_every = 10000

    coord_regressor.train()
    t1 = time.time()
    while i < max_iters:
        optimizer_coord.zero_grad()

        target_img_idx = train_sampler(batch_size)
        target_tform_cam2world = train_split[target_img_idx].tform_cam2world
        target_focal = train_split[target_img_idx].focal_length
        target_center = train_split[target_img_idx].center
        target_bbox = train_split[target_img_idx].bbox

        z = torch.randn((batch_size, args.latent_dim), device=device)
        with torch.no_grad():
            # Render a synthetic image
            if args.use_class:
                target_class = model_ema.class_embedding(
                    train_split[target_img_idx].classes)
            else:
                target_class = None
            w = model_ema.mapping_network(z, target_class)
            target_image, _, target_mask, _, target_coords, _ = parallel_model(
                target_tform_cam2world,
                target_focal,
                target_center,
                target_bbox,
                w,
                use_ema=True,
                compute_coords=True)
            target_image.clamp_(-1, 1)

        pred_coords, pred_mask, pred_w = coord_regressor(
            target_image.permute(0, 3, 1, 2))
        loss = 0.
        loss_coords = None
        loss_mask = None
        loss_latent = None
        if regress_pose:
            loss_coords = criterion_coords(pred_coords, target_coords,
                                           target_mask)
            loss_mask = criterion_mask(pred_mask, target_mask)
            loss = loss_coords + loss_mask
            loss_curve_coord.append(loss_coords.item())
        if regress_latent:
            loss_latent = criterion_latent(pred_w, w[:, :1])
            loss = loss + loss_latent
        loss.backward()
        optimizer_coord.step()

        if writer is not None:
            if regress_pose:
                writer.add_scalar('coord_regressor/loss_coords',
                                  loss_coords.item(), i)
                if loss_mask is not None:
                    writer.add_scalar('coord_regressor/loss_mask',
                                      loss_mask.item(), i)
            if regress_latent:
                writer.add_scalar('coord_regressor/loss_latent',
                                  loss_latent.item(), i)

        debug_every = 1000
        if i % debug_every == 0:
            loss_coords = loss_coords.item() if loss_coords is not None else 0.
            loss_mask = loss_mask.item() if loss_mask is not None else 0.
            loss_latent = loss_latent.item() if loss_latent is not None else 0.
            print(
                f'[{i}] loss_coords {loss_coords:.05f} loss_mask {loss_mask:.05f} loss_latent {loss_latent:.05f}'
            )

        i += 1

        # Reduce LR at 50% of the training schedule
        if i == max_iters // 2:
            print('Reducing learning rate...')
            for param_group in optimizer_coord.param_groups:
                param_group['lr'] /= 10
            lr /= 10

        if i % evaluate_every == 0 and regress_pose:
            save_checkpoint()
            if i % 20000 == 0:
                save_checkpoint(f'_{i}')

    coord_regressor.eval()
    coord_regressor.requires_grad_(False)
    save_checkpoint()

    return coord_regressor


def estimate_poses_batch(target_coords, target_mask, focal_guesses):
    target_mask = target_mask > 0.9
    if focal_guesses is None:
        # Use a large focal length to approximate ortho projection
        is_ortho = True
        focal_guesses = [100.]
    else:
        is_ortho = False

    world2cam_mat, estimated_focal, errors = pose_estimation.compute_pose_pnp(
        target_coords.cpu().numpy(),
        target_mask.cpu().numpy(), focal_guesses)

    if is_ortho:
        # Convert back to ortho
        s = 2 * focal_guesses[0] / -world2cam_mat[:, 2, 3]
        t2 = world2cam_mat[:, :2, 3] * s[..., None]
        world2cam_mat_ortho = world2cam_mat.copy()
        world2cam_mat_ortho[:, :2, 3] = t2
        world2cam_mat_ortho[:, 2, 3] = -10.
        world2cam_mat = world2cam_mat_ortho

    estimated_cam2world_mat = pose_utils.invert_space(
        torch.from_numpy(world2cam_mat).float()).to(target_coords.device)
    estimated_focal = torch.from_numpy(estimated_focal).float().to(
        target_coords.device)
    if is_ortho:
        estimated_cam2world_mat /= torch.from_numpy(
            s[:, None, None]).float().to(estimated_cam2world_mat.device)
        estimated_focal = None

    return estimated_cam2world_mat, estimated_focal, errors


if args.run_inversion:
    # Global config
    use_testset = args.inv_use_testset
    use_pose_regressor = True
    use_latent_regressor = True
    loss_to_use = args.inv_loss
    lr_gain_z = args.inv_gain_z
    inv_no_split = args.inv_no_split
    no_optimize_pose = args.inv_no_optimize_pose

    batch_size = args.batch_size // 4 * len(gpu_ids)

    if args.dataset == 'p3d_car' and use_testset:
        split_str = 'imagenettest' if args.inv_use_imagenet_testset else 'test'
    else:
        split_str = 'test' if use_testset else 'train'
    if args.inv_use_separate:
        mode_str = '_separate'
    else:
        mode_str = '_joint'
    if no_optimize_pose:
        mode_str += '_nooptpose'
    else:
        mode_str += '_optpose'
    w_split_str = 'nosplit' if inv_no_split else 'split'
    cfg_xid = f'_{args.xid}' if len(args.xid) > 0 else ''
    cfg_string = f'i{cfg_xid}_{split_str}{mode_str}_{loss_to_use}_gain{lr_gain_z}_{w_split_str}'
    cfg_string += f'_it{resume_from["iteration"]}'

    print('Config string:', cfg_string)

    report_dir_effective = os.path.join(report_dir, args.resume_from,
                                        cfg_string)
    print('Saving report in', report_dir_effective)
    utils.mkdir(report_dir_effective)
    writer = tensorboard.SummaryWriter(report_dir_effective)

    if use_pose_regressor or use_latent_regressor:
        if args.coord_resume_from:
            print('Resuming from pose regressor', args.coord_resume_from)
            coord_regressor = encoder.BootstrapEncoder(
                args.latent_dim,
                pose_regressor=use_pose_regressor,
                latent_regressor=use_latent_regressor,
                separate_backbones=args.inv_use_separate,
                pretrained=False).to(device)

            coord_regressor = nn.DataParallel(coord_regressor, gpu_ids)
            checkpoint_path = os.path.join(args.root_path, 'coords_checkpoints',
                                           args.resume_from,
                                           f'{args.coord_resume_from}.pth')
            with utils.open_file(checkpoint_path, 'rb') as f:
                coord_regressor.load_state_dict(
                    torch.load(f, map_location='cpu')['model_coord'])
            coord_regressor.requires_grad_(False)
            coord_regressor.eval()
        else:
            coord_regressor = train_coord_regressor(writer)
            if args.inv_train_coord_only:
                print('Exit...')
                sys.exit(0)

    if use_pose_regressor:
        focal_guesses = pose_estimation.get_focal_guesses(
            train_split.focal_length)

    image_indices = test_split.eval_indices if use_testset else train_eval_split.eval_indices
    image_indices_perm = test_split.eval_indices_perm if use_testset else train_eval_split.eval_indices_perm

    if args.inv_encoder_only:
        checkpoint_steps = [0]
    elif lr_gain_z >= 10:
        checkpoint_steps = [0, 10]
    else:
        checkpoint_steps = [0, 30]

    report = {
        step: {
            'ws': [],
            'z0': [],
            'R': [],
            's': [],
            't2': [],
            'psnr': [],
            'psnr_random': [],
            'lpips': [],
            'lpips_random': [],
            'ssim': [],
            'ssim_random': [],
            'iou': [],
            'rot_error': [],
            'inception_activations_front': [],  # Front view
            'inception_activations_random': [],  # Random view
        } for step in checkpoint_steps
    }

    with torch.no_grad():
        z_avg = model_ema.mapping_network.get_average_w()

    idx = 0
    test_bs = batch_size

    report_checkpoint_path = os.path.join(report_dir_effective,
                                          'report_checkpoint.pth')
    if utils.file_exists(report_checkpoint_path):
        print('Found inversion report checkpoint in', report_checkpoint_path)
        with utils.open_file(report_checkpoint_path, 'rb') as f:
            report_checkpoint = torch.load(f)
            report = report_checkpoint['report']
            idx = report_checkpoint['idx']
            test_bs = report_checkpoint['test_bs']
    else:
        print('Inversion report checkpoint not found, starting from scratch...')

    while idx < len(image_indices):
        t1 = time.time()
        frames = []

        if test_bs != 1 and image_indices[idx:idx + test_bs].shape[0] < test_bs:
            test_bs = 1

        target_img_idx = image_indices[idx:idx + test_bs]
        target_img_idx_perm = image_indices_perm[idx:idx + test_bs]
        if use_testset:
            target_img = test_split[
                target_img_idx].images  # Target for optimization (always cropped)
            target_img_fid_ = target_img  # Target for evaluation (front view -- always cropped)
            target_tform_cam2world = test_split[target_img_idx].tform_cam2world
            target_focal = test_split[target_img_idx].focal_length
            target_center = None
            target_bbox = None
            views_per_object = dataset_config['views_per_object_test']
            if views_per_object > 1:
                target_img_fid_random_ = test_split[target_img_idx_perm].images

            if use_pose_regressor and 'p3d' in args.dataset:
                # Use views from training set (as test pose distribution is not available)
                target_center_fid = None
                target_bbox_fid = None

                target_tform_cam2world_perm = train_eval_split[
                    target_img_idx_perm].tform_cam2world
                target_focal_perm = train_eval_split[
                    target_img_idx_perm].focal_length
                target_center_perm = train_eval_split[
                    target_img_idx_perm].center
                target_bbox_perm = train_eval_split[target_img_idx_perm].bbox
            else:
                if use_pose_regressor:
                    target_center_fid = None
                    target_bbox_fid = None
                else:
                    target_center_fid = test_split[target_img_idx].center
                    target_bbox_fid = test_split[target_img_idx].bbox

                target_tform_cam2world_perm = test_split[
                    target_img_idx_perm].tform_cam2world
                target_focal_perm = test_split[target_img_idx_perm].focal_length
                target_center_perm = test_split[target_img_idx_perm].center
                target_bbox_perm = test_split[target_img_idx_perm].bbox
        else:
            target_img = train_split[target_img_idx].images
            views_per_object = dataset_config['views_per_object']

            # Target for evaluation
            if dataset_config['camera_projection_model'] == 'ortho':
                # Need different eval. for CUB
                target_img_fid_ = train_split[
                    target_img_idx].images  # Not cropped (use bbox)
            else:
                target_img_fid_ = train_eval_split[
                    target_img_idx].images  # Cropped
            target_tform_cam2world = train_split[target_img_idx].tform_cam2world
            target_focal = train_split[target_img_idx].focal_length
            target_center = None
            target_bbox = None
            target_center_fid = train_eval_split[target_img_idx].center
            target_bbox_fid = train_eval_split[target_img_idx].bbox

            target_tform_cam2world_perm = train_eval_split[
                target_img_idx_perm].tform_cam2world
            target_focal_perm = train_eval_split[
                target_img_idx_perm].focal_length
            target_center_perm = train_eval_split[target_img_idx_perm].center
            target_bbox_perm = train_eval_split[target_img_idx_perm].bbox

            if views_per_object > 1:
                target_img_fid_random_ = train_eval_split[
                    target_img_idx_perm].images  # Cropped

        gt_cam2world_mat = target_tform_cam2world.clone()
        z_ = z_avg.clone().expand(test_bs, -1, -1).contiguous()

        if use_pose_regressor or use_latent_regressor:
            with torch.no_grad():
                coord_regressor_img = target_img[..., :3].permute(0, 3, 1, 2)
                if test_bs == 1:
                    target_coords, target_mask, target_w = coord_regressor.module(
                        coord_regressor_img)
                else:
                    target_coords, target_mask, target_w = coord_regressor(
                        coord_regressor_img)
            if use_pose_regressor:
                assert target_coords is not None
                estimated_cam2world_mat, estimated_focal, _ = estimate_poses_batch(
                    target_coords, target_mask, focal_guesses)
                target_tform_cam2world = estimated_cam2world_mat
                target_focal = estimated_focal
            if use_latent_regressor:
                assert target_w is not None
                z_.data[:] = target_w

        if inv_no_split:
            z_ = z_.mean(dim=1, keepdim=True)

        z_ /= lr_gain_z
        z_ = z_.requires_grad_()

        z0_, t2_, s_, R_ = pose_utils.matrix_to_pose(
            target_tform_cam2world,
            target_focal,
            camera_flipped=dataset_config['camera_flipped'])
        if not no_optimize_pose:
            t2_.requires_grad_()
            s_.requires_grad_()
            R_.requires_grad_()
        if z0_ is not None:
            if not no_optimize_pose:
                z0_.requires_grad_()
            param_list = [z_, z0_, R_, s_, t2_]
            param_names = ['z', 'f', 'R', 's', 't']
        else:
            param_list = [z_, R_, s_, t2_]
            param_names = ['z', 'R', 's', 't']
        if no_optimize_pose:
            param_list = param_list[:1]
            param_names = param_names[:1]

        extra_model_inputs = {}
        optimizer = torch.optim.Adam(param_list, lr=2e-3, betas=(0.9, 0.95))
        grad_norms = []
        for _ in range(len(param_list)):
            grad_norms.append([])

        model_to_call = parallel_model if z_.shape[
            0] > 1 else parallel_model.module

        psnrs = []
        lpipss = []
        rot_errors = []
        niter = max(checkpoint_steps)

        def evaluate_inversion(it):
            item = report[it]
            item['ws'].append(z_.detach().cpu() * lr_gain_z)
            if z0_ is not None:
                item['z0'].append(z0_.detach().cpu())
            item['R'].append(R_.detach().cpu())
            item['s'].append(s_.detach().cpu())
            item['t2'].append(t2_.detach().cpu())

            # Compute metrics for report
            cam, focal = pose_utils.pose_to_matrix(
                z0_.detach() if z0_ is not None else None,
                t2_.detach(),
                s_.detach(),
                F.normalize(R_.detach(), dim=-1),
                camera_flipped=dataset_config['camera_flipped'])
            rgb_predicted, _, acc_predicted, normals_predicted, semantics_predicted, extra_model_outputs = model_to_call(
                cam,
                focal,
                target_center_fid,
                target_bbox_fid,
                z_.detach() * lr_gain_z,
                use_ema=True,
                compute_normals=args.use_sdf and idx == 0,
                compute_semantics=args.attention_values > 0,
                force_no_cam_grad=True,
                extra_model_outputs=['attention_values']
                if args.attention_values > 0 else [],
                extra_model_inputs={
                    k: v.detach() for k, v in extra_model_inputs.items()
                },
            )

            rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
                                                                2).clamp_(
                                                                    -1, 1)
            target_perm = target_img_fid_.permute(0, 3, 1, 2)
            item['psnr'].append(
                metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
                             target_perm[:, :3] / 2 + 0.5,
                             reduction='none').cpu())
            item['ssim'].append(
                metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
                             target_perm[:, :3] / 2 + 0.5,
                             reduction='none').cpu())
            if dataset_config['has_mask']:
                item['iou'].append(
                    metrics.iou(acc_predicted,
                                target_perm[:, 3],
                                reduction='none').cpu())
            item['lpips'].append(
                loss_fn_lpips(rgb_predicted_perm[:, :3],
                              target_perm[:, :3],
                              normalize=False).flatten().cpu())
            item['inception_activations_front'].append(
                torch.FloatTensor(
                    fid.forward_inception_batch(
                        inception_net, rgb_predicted_perm[:, :3] / 2 + 0.5)))
            if not (args.dataset == 'p3d_car' and use_testset):
                # Ground-truth poses are not available on P3D Car (test set)
                item['rot_error'].append(
                    pose_utils.rotation_matrix_distance(cam, gt_cam2world_mat))

            if writer is not None and idx == 0:
                if it == checkpoint_steps[0]:
                    writer.add_images(f'img/ref',
                                      target_perm[:, :3].cpu() / 2 + 0.5, i)
                writer.add_images('img/recon_front',
                                  rgb_predicted_perm.cpu() / 2 + 0.5, it)
                writer.add_images('img/mask_front',
                                  acc_predicted.cpu().unsqueeze(1).clamp(0, 1),
                                  it)
                if normals_predicted is not None:
                    writer.add_images(
                        'img/normals_front',
                        normals_predicted.cpu().permute(0, 3, 1, 2) / 2 + 0.5,
                        it)
                if semantics_predicted is not None:
                    writer.add_images(
                        'img/semantics_front',
                        (semantics_predicted @ color_palette).cpu().permute(
                            0, 3, 1, 2) / 2 + 0.5, it)

            # Test with random poses
            rgb_predicted, _, _, normals_predicted, semantics_predicted, _ = model_to_call(
                target_tform_cam2world_perm,
                target_focal_perm,
                target_center_perm,
                target_bbox_perm,
                z_.detach() * lr_gain_z,
                use_ema=True,
                compute_normals=args.use_sdf and idx == 0,
                compute_semantics=args.attention_values > 0 and idx == 0,
                force_no_cam_grad=True,
                extra_model_inputs={
                    k: v.detach() for k, v in extra_model_inputs.items()
                },
            )
            rgb_predicted_perm = rgb_predicted.detach().permute(0, 3, 1,
                                                                2).clamp(-1, 1)
            if views_per_object > 1:
                target_perm_random = target_img_fid_random_.permute(0, 3, 1, 2)
                item['psnr_random'].append(
                    metrics.psnr(rgb_predicted_perm[:, :3] / 2 + 0.5,
                                 target_perm_random[:, :3] / 2 + 0.5,
                                 reduction='none').cpu())
                item['ssim_random'].append(
                    metrics.ssim(rgb_predicted_perm[:, :3] / 2 + 0.5,
                                 target_perm_random[:, :3] / 2 + 0.5,
                                 reduction='none').cpu())
                item['lpips_random'].append(
                    loss_fn_lpips(rgb_predicted_perm[:, :3],
                                  target_perm_random[:, :3],
                                  normalize=False).flatten().cpu())
            item['inception_activations_random'].append(
                torch.FloatTensor(
                    fid.forward_inception_batch(
                        inception_net, rgb_predicted_perm[:, :3] / 2 + 0.5)))
            if writer is not None and idx == 0:
                writer.add_images('img/recon_random',
                                  rgb_predicted_perm.cpu() / 2 + 0.5, it)
                writer.add_images('img/mask_random',
                                  acc_predicted.cpu().unsqueeze(1).clamp(0, 1),
                                  it)
                if normals_predicted is not None:
                    writer.add_images(
                        'img/normals_random',
                        normals_predicted.cpu().permute(0, 3, 1, 2) / 2 + 0.5,
                        it)
                if semantics_predicted is not None:
                    writer.add_images(
                        'img/semantics_random',
                        (semantics_predicted @ color_palette).cpu().permute(
                            0, 3, 1, 2) / 2 + 0.5, it)

        if 0 in checkpoint_steps:
            evaluate_inversion(0)

        def optimize_iter(module, rgb_predicted, acc_predicted,
                          semantics_predicted, extra_model_outputs, target_img,
                          cam, focal):

            target = target_img[..., :3]

            rgb_predicted_for_loss = rgb_predicted
            target_for_loss = target
            loss = 0.
            if loss_to_use in ['vgg_nocrop', 'vgg', 'mixed']:
                rgb_predicted_for_loss_aug = rgb_predicted_for_loss.permute(
                    0, 3, 1, 2)
                target_for_loss_aug = target_for_loss.permute(0, 3, 1, 2)
                num_augmentations = 0 if loss_to_use == 'vgg_nocrop' else 15
                if num_augmentations > 0:
                    predicted_target_cat = torch.cat(
                        (rgb_predicted_for_loss_aug, target_for_loss_aug),
                        dim=1)
                    predicted_target_cat = predicted_target_cat.unsqueeze(
                        1).expand(-1, num_augmentations, -1, -1,
                                  -1).contiguous().flatten(0, 1)
                    predicted_target_cat, _, _ = augment(
                        predicted_target_cat, None, None, 1.0)
                    rgb_predicted_for_loss_aug = torch.cat(
                        (rgb_predicted_for_loss_aug,
                         predicted_target_cat[:, :3]),
                        dim=0)
                    target_for_loss_aug = torch.cat(
                        (target_for_loss_aug, predicted_target_cat[:, 3:]),
                        dim=0)
                loss = loss + module.lpips_net(
                    rgb_predicted_for_loss_aug, target_for_loss_aug
                ).mean() * rgb_predicted.shape[
                    0]  # Disjoint samples, sum instead of average over batch
            if loss_to_use in ['l1', 'mixed']:
                loss = loss + F.l1_loss(rgb_predicted_for_loss, target_for_loss
                                       ) * rgb_predicted.shape[0]
            if loss_to_use == 'mse':
                loss = F.mse_loss(rgb_predicted_for_loss,
                                  target_for_loss) * rgb_predicted.shape[0]

            if loss_to_use == 'mixed':
                loss = loss / 2  # Average L1 and VGG

            with torch.no_grad():
                psnr_monitor = metrics.psnr(rgb_predicted[..., :3] / 2 + 0.5,
                                            target[..., :3] / 2 + 0.5)
                lpips_monitor = module.lpips_net(
                    rgb_predicted[..., :3].permute(0, 3, 1, 2),
                    target[..., :3].permute(0, 3, 1, 2),
                    normalize=False)

            return loss, psnr_monitor, lpips_monitor, rgb_predicted

        for it in range(niter):
            cam, focal = pose_utils.pose_to_matrix(
                z0_,
                t2_,
                s_,
                F.normalize(R_, dim=-1),
                camera_flipped=dataset_config['camera_flipped'])

            loss, psnr_monitor, lpips_monitor, rgb_predicted = model_to_call(
                cam,
                focal,
                target_center,
                target_bbox,
                z_ * lr_gain_z,
                use_ema=True,
                ray_multiplier=1 if args.fine_sampling else 4,
                res_multiplier=1,
                compute_normals=False and args.use_sdf,
                force_no_cam_grad=no_optimize_pose,
                closure=optimize_iter,
                closure_params={
                    'target_img': target_img,
                    'cam': cam,
                    'focal': focal
                },
                extra_model_inputs=extra_model_inputs,
            )
            normal_map = None
            loss = loss.sum()
            psnr_monitor = psnr_monitor.mean()
            lpips_monitor = lpips_monitor.mean()
            if writer is not None and idx == 0:
                writer.add_scalar('monitor_b0/psnr', psnr_monitor.item(), it)
                writer.add_scalar('monitor_b0/lpips', lpips_monitor.item(), it)
                rot_error = pose_utils.rotation_matrix_distance(
                    cam, gt_cam2world_mat).mean().item()
                rot_errors.append(rot_error)
                writer.add_scalar('monitor_b0/rot_error', rot_error, it)

            if args.use_sdf and normal_map is not None:
                rgb_predicted = torch.cat(
                    (rgb_predicted.detach(), normal_map.detach()), dim=-2)

            loss.backward()
            for i, param in enumerate(param_list):
                if param.grad is not None:
                    grad_norms[i].append(param.grad.norm().item())
                else:
                    grad_norms[i].append(0.)
            optimizer.step()
            optimizer.zero_grad()
            R_.data[:] = F.normalize(R_.data, dim=-1)
            if z0_ is not None:
                z0_.data.clamp_(-4, 4)
            s_.data.abs_()

            if it + 1 in report:
                evaluate_inversion(it + 1)

        t2 = time.time()
        idx += test_bs
        print(
            f'[{idx}/{len(image_indices)}] Finished batch in {t2-t1} s ({(t2-t1)/test_bs} s/img)'
        )

        if idx % 512 == 0:
            # Save report checkpoint
            with utils.open_file(report_checkpoint_path, 'wb') as f:
                torch.save({
                    'report': report,
                    'idx': idx,
                    'test_bs': test_bs,
                }, f)

    # Consolidate stats
    for report_entry in report.values():
        for k, v in list(report_entry.items()):
            if len(v) == 0:
                del report_entry[k]
            else:
                report_entry[k] = torch.cat(v, dim=0)

    print()
    print('Useful information:')
    print('psnr_random: PSNR evaluated on novel views (in the test set, if available)')
    print('ssim_random: SSIM evaluated on novel views (in the test set, if available)')
    print('rot_error: rotation error in degrees (only reliable on synthetic datasets)')
    print('fid_random: FID of selected split, evaluated against stats of train split')
    print('fid_random_test: FID of selected split, evaluated against stats of test split')
    print()
    report_str_full = ''
    for iter_num, report_entry in report.items():
        report_str = f'[{iter_num} iterations]'
        for elem in [
                'psnr', 'psnr_random', 'lpips', 'lpips_random', 'ssim',
                'ssim_random', 'iou', 'rot_error'
        ]:
            if elem in report_entry:
                elem_val = report_entry[elem].mean().item()
                report_str += f' {elem} {elem_val:.05f}'
                report_entry[f'{elem}_avg'] = elem_val
                writer.add_scalar(f'report/{elem}', elem_val, iter_num)

        def add_inception_report(report_entry_key, tensorboard_key):
            global report_str
            if report_entry_key not in report_entry:
                return None
            fid_stats = fid.calculate_stats(
                report_entry[report_entry_key].numpy())
            fid_value = fid.calculate_frechet_distance(
                *fid_stats, *train_eval_split.fid_stats)
            report_entry[tensorboard_key] = fid_value
            report_str += f' {tensorboard_key} {fid_value:.02f}'
            del report_entry[report_entry_key]
            writer.add_scalar(f'report/{tensorboard_key}', fid_value, iter_num)
            if use_testset:
                fid_test = fid.calculate_frechet_distance(
                    *fid_stats, *test_split.fid_stats)
                report_entry[tensorboard_key + '_test'] = fid_test
                report_str += f' {tensorboard_key}_test {fid_test:.02f}'
                writer.add_scalar(f'report/{tensorboard_key}_test', fid_test,
                                  iter_num)
            return fid_value

        add_inception_report('inception_activations_front', 'fid_front')
        fid_value = add_inception_report('inception_activations_random',
                                         'fid_random')

        print(report_str)
        report_str_full += report_str + '\n'

    report_file_in = os.path.join(report_dir_effective, 'report')
    with utils.open_file(report_file_in + '.pth', 'wb') as f:
        torch.save(report, f)
    with utils.open_file(report_file_in + '.txt', 'w') as f:
        f.write(args.resume_from + '\n')
        f.write(cfg_string + '\n')
        f.write(report_str_full)
