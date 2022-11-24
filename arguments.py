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

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--gpus',
                        type=int,
                        default=4,
                        help='Number of GPUs to use')
    parser.add_argument(
        '--dataset',
        type=str,
        default='autodetect',
        help='Dataset among (shapenet_*, p3d_*, cub, imagenet_*, carla)')
    parser.add_argument(
        '--xid',
        type=str,
        default='',
        help='Additional information to embed in the experiment name')
    parser.add_argument('--resolution',
                        type=int,
                        default=128,
                        help='Rendering resolution')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Total batch size')
    parser.add_argument('--run_inversion', action='store_true')
    parser.add_argument('--resume_from',
                        type=str,
                        help='Load specified unconditional checkpoint')
    parser.add_argument('--root_path',
                        type=str,
                        default='.',
                        help='Root path for checkpoints')
    parser.add_argument('--data_path',
                        type=str,
                        default='datasets',
                        help='Root path for datasets')

    # Training settings
    parser.add_argument('--iterations',
                        type=int,
                        default=300000,
                        help='Number of training iterations for generator')
    parser.add_argument('--lr_g',
                        type=float,
                        default=0.0025,
                        help='Learning rate for generator')
    parser.add_argument('--lr_d',
                        type=float,
                        default=0.002,
                        help='Learning rate for discriminator')
    parser.add_argument('--dual_discriminator',
                        action='store_true',
                        help='Use dual discriminator in encoder-mode')
    parser.add_argument('--dual_discriminator_l1',
                        action='store_true',
                        help='Use L1+discriminator in encoder-mode')
    parser.add_argument('--dual_discriminator_mse',
                        action='store_true',
                        help='Use MSE+discriminator in encoder-mode')
    parser.add_argument('--r1',
                        type=float,
                        default=5.,
                        help='R1 regularization strength')
    parser.add_argument('--tv',
                        type=float,
                        default=0.5,
                        help='Total variation regularization strength')
    parser.add_argument('--entropy',
                        type=float,
                        default=0.05,
                        help='Entropy regularization strength')
    parser.add_argument('--eikonal',
                        type=float,
                        default=0.1,
                        help='Eikonal loss strength')
    parser.add_argument('--supervise_alpha',
                        action='store_true',
                        help='Include alpha channel in discriminator input')
    parser.add_argument('--conditional_pose',
                        type=bool,
                        default=True,
                        help='Condition discriminator on pose')
    parser.add_argument('--augment_p',
                        type=float,
                        default=0,
                        help='Maximum augmentation probability for ADA')
    parser.add_argument('--augment_ada', action='store_true', help='Enable ADA')
    parser.add_argument('--ada_target',
                        type=float,
                        default=0.6,
                        help='Sign target for ADA')
    parser.add_argument('--path_length_regularization',
                        action='store_true',
                        help='Enable path length regularization')
    parser.add_argument('--perturb_poses',
                        type=float,
                        default=0,
                        help='Random pose perturbation (in degrees)')
    parser.add_argument('--clip_gradient_norm',
                        type=float,
                        default=100.,
                        help='Clip gradient norm of G and D')

    # Model settings
    parser.add_argument('--fine_sampling',
                        type=bool,
                        default=True,
                        help='Enable two-pass coarse-fine sampling')
    parser.add_argument(
        '--attention_values',
        type=int,
        default=10,
        help='Number of values in color mapping (set >0 to enable)')
    parser.add_argument('--use_sdf',
                        type=bool,
                        default=True,
                        help='Enable SDF representation')
    parser.add_argument('--use_encoder',
                        action='store_true',
                        help='Train encoder-based mode')
    parser.add_argument('--use_viewdir',
                        action='store_true',
                        help='Enable view-dependent effects')
    parser.add_argument('--use_class',
                        action='store_true',
                        help='Use class-conditional backbone')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=512,
                        help='Size of latent code')
    parser.add_argument('--disable_stylegan_noise',
                        type=bool,
                        default=True,
                        help='Disable noise in StyleGAN backbone')

    # Model inversion params
    parser.add_argument('--inv_use_testset',
                        action='store_true',
                        help='Invert test set')
    parser.add_argument('--inv_use_imagenet_testset',
                        action='store_true',
                        help='Use ImageNet test set for P3D Cars')
    parser.add_argument(
        '--inv_use_separate',
        action='store_true',
        help='Use separate models for coord and latent regressor')
    parser.add_argument('--inv_loss',
                        type=str,
                        default='vgg',
                        help='Loss to use for inversion')
    parser.add_argument('--inv_gain_z',
                        type=int,
                        default=5,
                        help='Gain to use for inversion')
    parser.add_argument('--inv_no_split',
                        action='store_true',
                        help='Do not split latent code for inversion')
    parser.add_argument('--inv_no_optimize_pose',
                        action='store_true',
                        help='Do not optimize pose for inversion')
    parser.add_argument('--inv_train_coord_only',
                        action='store_true',
                        help='Exit after training encoder')
    parser.add_argument('--inv_encoder_only',
                        action='store_true',
                        help='Do not apply inversion (show result with N=0)')

    # Coord regressor params
    parser.add_argument('--coord_resume_from', type=str)

    args = parser.parse_args()

    if args.dual_discriminator_l1 and not args.dual_discriminator:
        print('INFO: --dual_discriminator_l1 implies --dual_discriminator')
        args.dual_discriminator = True

    if args.dual_discriminator_mse and not args.dual_discriminator:
        print('INFO: --dual_discriminator_mse implies --dual_discriminator')
        args.dual_discriminator = True

    return args


def suggest_experiment_name(args):
    if args.use_encoder:
        experiment_name = 'r'
    else:
        experiment_name = 'g'
    experiment_name += f'_{args.xid}' if len(args.xid) > 0 else ''
    experiment_name += f'_{args.dataset}'
    experiment_name += f'_res{args.resolution}_bs{args.batch_size}_d{args.latent_dim}_lrg_{args.lr_g}_lrd_{args.lr_d}'
    if args.r1 > 0:
        experiment_name += f'_r1_{args.r1}'
    if args.entropy > 0:
        experiment_name += f'_entropy_{args.entropy}'
    if args.tv > 0:
        experiment_name += f'_tv_{args.tv}'
    if args.dual_discriminator:
        experiment_name += '_dual'
        if args.dual_discriminator_mse:
            experiment_name += '_mse'
        elif args.dual_discriminator_l1:
            experiment_name += '_l1'
    if args.fine_sampling:
        experiment_name += '_fine'
    else:
        experiment_name += '_nofine'
    if args.use_sdf:
        experiment_name += f'_sdf_eik{args.eikonal}'
    else:
        experiment_name += '_nosdf'
    if args.attention_values > 0:
        experiment_name += f'_attn{args.attention_values}'
    if args.supervise_alpha:
        experiment_name += '_alpha'
    else:
        experiment_name += '_noalpha'
    if args.conditional_pose:
        experiment_name += '_pose'
    else:
        experiment_name += '_nopose'
    if args.perturb_poses > 0:
        experiment_name += f'_perturb{args.perturb_poses}'
    if args.augment_p > 0:
        experiment_name += f'_augment_p{args.augment_p}'
        if args.augment_ada:
            experiment_name += f'_ada{args.ada_target}'
    if args.use_viewdir:
        experiment_name += '_viewdir'
    if args.use_class:
        experiment_name += '_class'
    if args.path_length_regularization:
        experiment_name += '_ppl'
    if args.disable_stylegan_noise:
        experiment_name += '_nonoise'

    print('Experiment name', experiment_name)
    return experiment_name
