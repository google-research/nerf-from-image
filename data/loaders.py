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
import torch
import torch.nn.functional as F
import numpy as np
from data import datasets
from tqdm import tqdm


def get_dataset_config(dataset):
    if dataset.startswith('shapenet'):
        dataset_config = {
            'scene_range': 1.1 / 2,
            'white_background': True,
            'has_mask': False,
            'has_bbox': False,
            'is_highres': False,
            'views_per_object': 50,
            'views_per_object_test': 251,
            'camera_projection_model': 'perspective',
            'camera_flipped': False,
        }
    elif dataset.startswith('p3d_'):
        dataset_config = {
            'scene_range': 1.4,
            'white_background': False,
            'has_mask': True,
            'has_bbox': True,
            'is_highres': True,
            'views_per_object': 1,
            'views_per_object_test': 1,
            'camera_projection_model': 'perspective',
            'camera_flipped': True,
        }
    elif dataset.startswith('imagenet_'):
        dataset_config = {
            'scene_range': 1.4,
            'white_background': False,
            'has_mask': True,
            'has_bbox': True,
            'is_highres': True,
            'views_per_object': 1,
            'views_per_object_test': None,
            'camera_projection_model': 'perspective',
            'camera_flipped': True,
        }
    elif dataset == 'cub':
        dataset_config = {
            'scene_range': 2.0,
            'white_background': False,
            'has_mask': True,
            'has_bbox': True,
            'is_highres': True,
            'views_per_object': 1,
            'views_per_object_test': 1,
            'camera_projection_model': 'ortho',
            'camera_flipped': True,
        }
    elif dataset == 'carla':
        dataset_config = {
            'scene_range': 3.0,
            'white_background': True,
            'has_mask': False,
            'has_bbox': False,
            'is_highres': True,
            'views_per_object': 1,
            'views_per_object_test': None,
            'camera_projection_model': 'perspective',
            'camera_flipped': False,
        }
    else:
        raise ValueError('Invalid dataset')

    return dataset_config


def override_default_args(args):
    if args.dataset == 'cub':
        args.iterations = 200000
        args.disable_stylegan_noise = False
        args.supervise_alpha = True
        args.augment_p = 0.8
        args.augment_ada = True
        args.inv_use_testset = True

    if args.dataset.startswith('imagenet'):
        args.supervise_alpha = True
        args.augment_p = 0.8
        args.augment_ada = True

    if args.dataset == 'imagenet_elephant':
        args.iterations = 200000
        args.disable_stylegan_noise = False
        args.r1 = 10.

    if args.dataset.startswith('p3d'):
        args.supervise_alpha = True
        args.augment_p = 0.8
        args.augment_ada = True
        args.inv_use_testset = True

    if args.dataset == 'carla':
        args.use_viewdir = True
        args.augment_p = 0.8
        args.augment_ada = True

    if args.dataset.startswith('shapenet'):
        args.inv_use_testset = True
        # We disable pose fine-tuning due to the novel view evaluation
        args.inv_no_optimize_pose = True 


def get_dataset_loaders():
    return {
        'shapenet_cars': load_shapenet,
        'shapenet_chairs': load_shapenet,
        'p3d_car': load_custom,
        'cub': load_custom,
        'carla': load_carla,
        'imagenet_car': load_custom,
        'imagenet_airplane': load_custom,
        'imagenet_motorcycle': load_custom,
        'imagenet_zebra': load_custom,
        'imagenet_elephant': load_custom,
    }

def get_coco_mapping():
    return {
        'p3d_car': 2,
        'cub': 14,
        'imagenet_car': 2,
        'imagenet_airplane': 4,
        'imagenet_motorcycle': 3,
        'imagenet_zebra': 22,
        'imagenet_elephant': 20,
    }


class DatasetSplitView:

    def __init__(self, parent, idx):
        self.parent = parent
        self.idx = idx

    def __getattr__(self, attr):
        if isinstance(attr, (list, tuple)):
            outputs = []
            for elem in attr:
                parent_attr = getattr(self.parent, elem)
                if parent_attr is None:
                    outputs.append(None)
                else:
                    outputs.append(parent_attr[self.idx].to(parent.device))
            return outputs
        else:
            parent_attr = getattr(self.parent, attr)
            if parent_attr is None:
                return None
            else:
                return parent_attr[self.idx].to(self.parent.device)


class DatasetSplit:

    def __init__(self, device='cpu'):
        self.device = device
        self.images = None
        self.images_highres = None
        self.tform_cam2world = None
        self.focal_length = None
        self.bbox = None
        self.center = None
        self.classes = None
        self.num_classes = None

        self.fid_stats = None
        self.eval_indices = None
        self.eval_indices_perm = None

    def __getitem__(self, idx):
        return DatasetSplitView(self, idx)


def autodetect_dataset(experiment_name):
    dataset_choices = get_dataset_loaders().keys()
    found_dataset = None
    for choice in dataset_choices:
        if f'_{choice}_' in experiment_name:
            assert found_dataset is None
            found_dataset = choice
    if found_dataset:
        print(f'Autodetected {found_dataset} dataset')
        return found_dataset
    else:
        raise RuntimeError(
            'Unable to autodetect dataset, please specify it manually via --dataset'
        )


def load_dataset(args, device, manual_image=None):
    override_default_args(args)
    dataset_config = get_dataset_config(args.dataset)
    loader = get_dataset_loaders()[args.dataset]
    if manual_image is not None:
        extra_kwargs = {'manual_image': manual_image}
        args.augment_p = 0
    else:
        extra_kwargs = {}
    train_split, train_eval_split, test_split = loader(dataset_config, args,
                                                       device, **extra_kwargs)

    return dataset_config, train_split, train_eval_split, test_split


def insert_manual_image(dataset, split, manual_image):
    img, mask, _, _, _, _, _, bbox, _ = dataset.forward_img(None, manual_image)
    mask = mask[None, :, :]
    img = img * 2 - 1
    img *= mask
    img = np.concatenate((img, mask), axis=0)
    img = torch.FloatTensor(img).permute(1, 2, 0)
    split.images[0] = img
    if split.bbox[0] is not None and split.bbox[0].shape[-1] == 4:
        split.bbox[0] = torch.FloatTensor(bbox)

def load_custom(dataset_config, args, device, manual_image=None):
    if args.dataset.startswith('p3d_') or args.dataset.startswith('imagenet_'):
        dataset_inst = lambda *fn_args, **fn_kwargs: datasets.CustomDataset(
            args.dataset, *fn_args, **fn_kwargs, root_dir=args.data_path)
    else:
        dataset_inst = lambda *fn_args, **fn_kwargs: datasets.CUBDataset(
            *fn_args, **fn_kwargs, root_dir=args.data_path)

    img_size = args.resolution
    img_size_train = img_size * 2 if args.augment_p > 0 else img_size
    dataset = dataset_inst('train',
                           img_size=img_size_train,
                           crop=False,
                           add_mirrored=True)
    dataset_fid = dataset_inst('train',
                               img_size=img_size,
                               crop=True,
                               add_mirrored=False)
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=False,
                                         batch_size=32,
                                         num_workers=8,
                                         pin_memory=False)
    loader_fid = torch.utils.data.DataLoader(dataset_fid,
                                             shuffle=False,
                                             batch_size=32,
                                             num_workers=8,
                                             pin_memory=False)

    train_split = DatasetSplit(device)
    train_eval_split = DatasetSplit(device)

    if dataset_config['views_per_object_test'] and (args.use_encoder or
                                                    args.run_inversion):
        if args.dataset == 'p3d_car' and args.inv_use_imagenet_testset:
            test_split = 'imagenet_test'
        else:
            test_split = 'test'
        dataset_test = dataset_inst(test_split,
                                    img_size=img_size,
                                    crop=True,
                                    add_mirrored=False)
        loader_test = torch.utils.data.DataLoader(dataset_test,
                                                  shuffle=False,
                                                  batch_size=32,
                                                  num_workers=8,
                                                  pin_memory=False)
        test_split = DatasetSplit(device)
    else:
        test_split = None

    all_images = []
    all_images_highres = [] if args.augment_p > 0 else None
    all_images_fid = []
    all_poses = []
    all_focal = []
    all_bbox = []
    all_classes = []

    all_poses_fid = []
    all_focal_fid = []
    all_bbox_fid = []
    all_classes_fid = []

    for i, sample in enumerate(tqdm(loader)):
        if args.augment_p > 0:
            all_images_highres.append(sample['img'].clamp(-1, 1).permute(
                0, 2, 3, 1))
            all_images.append(
                F.avg_pool2d(sample['img'], 2).clamp(-1, 1).permute(0, 2, 3, 1))
        else:
            all_images.append(sample['img'].clamp(-1, 1).permute(0, 2, 3, 1))
        # We clone the tensors to avoid issues with shared memory
        all_poses.append(sample['pose'].clone())
        all_focal.append(sample['focal'].clone())
        all_bbox.append(sample['normalized_bbox'].clone())
        all_classes.append(sample['class'].clone())

    for i, sample in enumerate(tqdm(loader_fid)):
        all_images_fid.append(sample['img'].clamp(-1, 1).permute(0, 2, 3, 1))
        all_poses_fid.append(sample['pose'].clone())
        all_focal_fid.append(sample['focal'].clone())
        all_bbox_fid.append(sample['normalized_bbox'].clone())
        all_classes_fid.append(sample['class'].clone())

    if dataset_config['views_per_object_test'] and (args.use_encoder or
                                                    args.run_inversion):
        all_images_test = []
        all_poses_test = []
        all_focal_test = []
        all_bbox_test = []
        for i, sample in enumerate(tqdm(loader_test)):
            all_images_test.append(sample['img'].clamp(-1,
                                                       1).permute(0, 2, 3, 1))
            all_poses_test.append(sample['pose'].clone())
            all_focal_test.append(sample['focal'].clone())
            all_bbox_test.append(sample['normalized_bbox'].clone())
        test_split.images = torch.cat(all_images_test, dim=0)
        test_split.tform_cam2world = torch.cat(all_poses_test, dim=0)
        test_split.focal_length = torch.cat(all_focal_test, dim=0).squeeze(1)
        test_split.bbox = torch.cat(all_bbox_test, dim=0)
        print('Loaded test split with shape', test_split.images.shape)

        if manual_image is not None:
            # Replace first image with supplied image (demo inference)
            insert_manual_image(dataset_test, test_split, manual_image)

    train_split.images = torch.cat(all_images, dim=0)
    train_eval_split.images = torch.cat(all_images_fid, dim=0)
    all_images = None  # Free up memory
    all_images_fid = None
    if args.augment_p > 0:
        train_split.images_highres = torch.cat(all_images_highres, dim=0)
        all_images_highres = None  # Free up memory
    train_split.tform_cam2world = torch.cat(all_poses, dim=0)
    train_split.focal_length = torch.cat(all_focal, dim=0).squeeze(1)
    train_split.bbox = torch.cat(all_bbox, dim=0)
    train_split.classes = torch.cat(all_classes, dim=0)
    train_split.num_classes = train_split.classes.max().item() + 1
    if manual_image is not None:
        # Replace first image with supplied image (demo inference)
        insert_manual_image(dataset, train_split, manual_image)

    train_eval_split.tform_cam2world = torch.cat(all_poses_fid, dim=0)
    train_eval_split.focal_length = torch.cat(all_focal_fid, dim=0).squeeze(1)
    train_eval_split.bbox = torch.cat(all_bbox_fid, dim=0)
    train_eval_split.classes = torch.cat(all_classes_fid, dim=0)
    train_eval_split.num_classes = train_split.num_classes
    if manual_image is not None:
        # Replace first image with supplied image (demo inference)
        insert_manual_image(dataset_fid, train_eval_split, manual_image)

    if args.dataset == 'cub':
        # Ortho camera
        train_split.focal_length = None
        train_split.bbox = None
        train_eval_split.focal_length = None
        train_eval_split.bbox = None
        if test_split is not None:
            test_split.focal_length = None
            test_split.bbox = None
    else:
        # Training images are always uncropped
        train_split.bbox = None

    if not args.use_class or args.dataset != 'cub':
        train_split.classes = None
        train_split.num_classes = None
        train_eval_split.classes = None
        train_eval_split.num_classes = None

    print('Loaded train split with shape', train_split.images.shape)
    print('Loaded train_eval split with shape', train_eval_split.images.shape)
    return train_split, train_eval_split, test_split


def load_shapenet(dataset_config, args, device):
    np.random.seed(1234)
    shapenet_category = args.dataset.split('_')[1]
    shapenet_path = os.path.join(args.data_path, 'shapenet', shapenet_category)
    dataset = datasets.SRNDataset(shapenet_path, stage='train', limit=None)
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=False,
                                         batch_size=32,
                                         num_workers=16)
    train_split = DatasetSplit(device)
    train_eval_split = DatasetSplit(device)

    if args.use_encoder or args.run_inversion:
        dataset_test = datasets.SRNDataset(shapenet_path,
                                           stage='test',
                                           limit=None)
        loader_test = torch.utils.data.DataLoader(dataset_test,
                                                  shuffle=False,
                                                  batch_size=32,
                                                  num_workers=16)
        test_split = DatasetSplit(device)
    else:
        test_split = None

    def load_shapenet(loader):
        all_images = []
        all_poses = []
        focal_length = None
        center = None
        for i, sample in enumerate(tqdm(loader)):
            if i == 0:
                focal_length = sample['focal'][0]
                center = sample['c'][0]
            assert (sample['focal'] == focal_length).all()
            assert (sample['c'] == center).all()
            all_images.append(sample['images'].flatten(0, 1))
            all_poses.append(sample['poses'].flatten(0, 1))

        images = torch.cat(all_images, dim=0).permute(0, 2, 3, 1)
        tform_cam2world = torch.cat(all_poses, dim=0).to(device)
        focal_length = focal_length.to(device).expand(len(images))
        return images, tform_cam2world, focal_length

    train_split.images, train_split.tform_cam2world, train_split.focal_length = load_shapenet(
        loader)

    train_eval_split.images = train_split.images
    train_eval_split.tform_cam2world = train_split.tform_cam2world
    train_eval_split.focal_length = train_split.focal_length

    if args.use_encoder or args.run_inversion:
        test_split.images, test_split.tform_cam2world, test_split.focal_length = load_shapenet(
            loader_test)

    print(train_split.images.shape)
    return train_split, train_eval_split, test_split


def load_carla(dataset_config, args, device):
    img_size = args.resolution
    dataset = datasets.CARLADataset(os.path.join(args.data_path, 'carla'),
                                    image_size=img_size,
                                    upscale=args.augment_p > 0)
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=False,
                                         batch_size=32,
                                         num_workers=16)

    train_split = DatasetSplit(device)
    train_eval_split = DatasetSplit(device)
    test_split = None

    all_images = []
    all_images_highres = [] if args.augment_p > 0 else None
    all_poses = []

    focal_length = None
    center = None
    for i, sample in enumerate(tqdm(loader)):
        if i == 0:
            focal_length = sample['focal'][0]
            center = sample['c'][0]
        assert (sample['focal'] == focal_length).all()
        assert (sample['c'] == center).all()
        if args.augment_p > 0:
            all_images_highres.append(sample['image'])
            all_images.append(F.avg_pool2d(sample['image'], 2))
        else:
            all_images.append(sample['image'])
        all_poses.append(sample['pose'])

    train_split.images = torch.cat(all_images, dim=0).permute(0, 2, 3, 1)
    all_images = None  # Free up memory
    if args.augment_p > 0:
        train_split.images_highres = torch.cat(all_images_highres,
                                               dim=0).permute(0, 2, 3, 1)
        all_images_highres = None  # Free up memory
    train_split.tform_cam2world = torch.cat(all_poses, dim=0).to(device)
    train_split.focal_length = focal_length.to(device).expand(
        len(train_split.images))

    train_eval_split.images = train_split.images
    train_eval_split.tform_cam2world = train_split.tform_cam2world
    train_eval_split.focal_length = train_split.focal_length

    print(train_split.images.shape)
    return train_split, train_eval_split, test_split
