# Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion
*This is not an officially supported Google product.*

This repository contains the code for the paper
> Dario Pavllo, David Joseph Tan, Marie-Julie Rakotosaona, Federico Tombari. [Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion](https://arxiv.org/abs/2211.11674). In arXiv, 2022.

Our approach recovers an SDF-parameterized 3D shape, pose, and appearance from a single image of an object, without exploiting multiple views during training. More specifically, we leverage an unconditional 3D-aware generator, to which we apply a hybrid inversion scheme where a model produces a first guess of the solution which is then refined via optimization.

![](images/teaser.jpg)

# Setup
Please follow the instructions in [SETUP.md](SETUP.md).

# Training
The unconditional generator can be trained as follows:
```
python run.py --dataset DATASET --path_length_regularization --gpus 4 --batch_size 32
```
where `DATASET` is any of `shapenet_cars`, `shapenet_chairs`, `carla`, `p3d_car`, `cub`, `imagenet_*` (see available ImageNet classes in [SETUP.md](SETUP.md)). TensorBoard logs are exported to `gan_logs/`. The results in the paper were produced using a total batch size of 32, which requires either 4 GPUs with 40GB of memory each (e.g. A100) or 8 GPUs with at least 24 GB each.

Afterwards, the hybrid inversion procedure can be launched via:
```
python run.py --resume_from EXPERIMENT_NAME --run_inversion
```
where `EXPERIMENT_NAME` is the name of the experiment produced by the previous step (you can also find it in `gan_checkpoints/`). This will first train the encoder, save it to `coords_checkpoints/`, and finally launch the actual inversion procedure (whose outputs and TensorBoard logs are exported to `reports/`). You can also compute results in feed-forward mode by specifying `--inv_encoder_only`, which produces the numbers labeled as N=0 in the paper. For `p3d_car`, you can evaluate on our custom ImageNet test set by specifying `--inv_use_imagenet_testset` (otherwise, the official test set is used).

The full list of arguments can be found in [arguments.py](arguments.py)

*More details coming soon.*

# License
This code is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for more details.