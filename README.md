# Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion
*This is not an officially supported Google product.*

This repository contains the code for the paper
> Dario Pavllo, David Joseph Tan, Marie-Julie Rakotosaona, Federico Tombari. [Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion](https://arxiv.org/abs/2211.11674). In arXiv, 2022.

Our approach recovers an SDF-parameterized 3D shape, pose, and appearance from a single image of an object, without exploiting multiple views during training. More specifically, we leverage an unconditional 3D-aware generator, to which we apply a hybrid inversion scheme where a model produces a first guess of the solution which is then refined via optimization.

# Setup
Please follow the instructions in [SETUP.md](SETUP.md).

# Training
The unconditional generator can be trained as follows:
```
python run.py --dataset DATASET --path_length_regularization --gpus 4 --batch_size 32
```
where `DATASET` is any of `shapenet_cars`, `shapenet_chairs`, `carla`, `p3d_car`, `cub`, `imagenet_*` (see available ImageNet classes in [SETUP.md](SETUP.md)). The results in the paper were produced using a total batch size of 32. This requires either 4 GPUs with 40GB of memory each (e.g. A100) or 8 GPUs with at least 24 GB each.

Afterwards, the hybrid inversion procedure can be launched via:
```
python run.py --resume_from EXPERIMENT_NAME --run_inversion
```
where `EXPERIMENT_NAME` is the name of the experiment produced by the previous step.

*More details coming soon.*

# License
This code is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for more details.
