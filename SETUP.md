# Setup

# Pascal3D+ Cars
Download `PASCAL3D+_release1.1.zip` from the [official website](https://cvgl.stanford.edu/projects/pascal3d.html) and extract it to `datasets/p3d/`. Download the [P3D cache archive from the CMR repo](https://drive.google.com/file/d/1RbiCWu1ArD3ii-92o5xNkY6TzXBH0tgo/view?usp=sharing) ([source](https://github.com/akanazawa/cmr/blob/master/doc/train.md)) and extract it to `datasets/p3d/`. Finally, download [poses_perspective.zip](https://github.com/dariopavllo/textured-3d-gan/releases/download/v1.0/poses_perspective.zip) (source: [textured-3d-gan repo](https://github.com/dariopavllo/textured-3d-gan)) and extract it to `datasets/`. Your directory tree should look like this:
```
datasets/p3d/
datasets/p3d/PASCAL3D+_release1.1/
datasets/p3d/p3d_sfm_image/
datasets/p3d/p3d_car/detections.npy
datasets/p3d/p3d_car/poses_estimated_singletpl_perspective.bin
```

# CUB (Birds)
Download [CUB_200_2011.tgz](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) ([source](https://www.vision.caltech.edu/datasets/cub_200_2011/)) and extract it to `datasets/cub/`. Then, download [segmentations.tgz](https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1) ([source](https://www.vision.caltech.edu/datasets/cub_200_2011/)) and extract it to `datasets/cub/CUB_200_2011/`.

You also need to download the [poses](https://www.dropbox.com/sh/ea3yprgrcjuzse5/AAB476Nn0Lwbrt3iuedB9yzIa?dl=0) (source: [CMR repo](https://github.com/akanazawa/cmr)) and extract them so that your directory tree finally looks like this:
```
datasets/cub/
datasets/cub/data
datasets/cub/sfm
datasets/cub/CUB_200_2011/
datasets/cub/CUB_200_2011/segmentations/
```

# ImageNet
To set up images from ImageNet, you need to download the synsets in the table below from the full ImageNet22k dataset (depending on the category you want to evaluate). The full dataset can be found on  [Academic Torrents](https://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47).

| Category        | Synsets                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| motorcycle      | n03790512, n03791053, n04466871                                                                   |
| car             | n02814533, n02958343, n03498781, n03770085, n03770679, n03930630, n04037443, n04166281, n04285965 |
| airplane        | n02690373, n02691156, n03335030, n04012084                                                        |
| elephant        | n02504013, n02504458                                                                              |
| zebra           | n02391049, n02391234, n02391373, n02391508                                                        |

You should then copy the individual synset directories to `datasets/imagenet/images/`. You additionally need to download [poses_perspective.zip](https://github.com/dariopavllo/textured-3d-gan/releases/download/v1.0/poses_perspective.zip) (source: [textured-3d-gan repo](https://github.com/dariopavllo/textured-3d-gan)) and extract it to `datasets/`.

Your directory tree should eventually look like this:
```
datasets/imagenet/
datasets/imagenet/images/n*
datasets/imagenet/imagenet_*/detections.npy
datasets/imagenet/imagenet_*/poses_estimated_multitpl_perspective.bin
```

# ShapeNet (Chairs & Cars)
Download [srn_chairs.zip and srn_cars.zip](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing) (source: [PixelNeRF repo](https://github.com/sxyu/pixel-nerf)) and extract them to `datasets/shapenet/`.

# CARLA (synthetic cars)
Download [carla.zip](https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla.zip) and [carla_poses.zip](https://s3.eu-central-1.amazonaws.com/avg-projects/graf/data/carla_poses.zip) (source: [GRAF repo](https://github.com/autonomousvision/graf)) and extract them to `datasets/carla/`.

# SegFormer weights
If want to train the encoder from scratch, you first need to download the pretrained backbone weights `mit_b5.pth` (pre-trained on ImageNet) from the [SegFormer model archive](https://drive.google.com/corp/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia) ([source](https://github.com/NVlabs/SegFormer)) and extract it to `coords_checkpoints/`.
