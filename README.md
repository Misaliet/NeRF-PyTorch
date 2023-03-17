# NeRF-PyTorch

I assume you already know about [NeRF](https://www.matthewtancik.com/nerf) since you are looking for a PyTorch implementation of NeRF. I will skip the introduction of NeRF.

This project is not official, but tries to make it easier to do further research with NeRF based on PyTorch.

## Motivation:

I used this [krrish94/nerf-pytorch](https://github.com/krrish94/nerf-pytorch) in the beginning because it is claimed to be **~5-9x faster** than the [original release](https://github.com/bmild/nerf). 

However, this is not true. In the lego config file, it uses a smaller (half size) NeRF to speed up. If you switch to the same lego dataset settings in the original NeRF paper, the speed is basically same with another code [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). This is discussed in [this](https://github.com/krrish94/nerf-pytorch/issues/10) and [this](https://github.com/krrish94/nerf-pytorch/issues/6).

Not to mention there are many bugs inside (e.g., use "PaperNeRFModel" you will find this is definitely not tested; set "half_res" to "False" you will find another untested parameters; ...).

This code [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) is much better, but it still has some bugs. Like the depth video saving is broken, and the network saving is also broken if you do not use the fine_model.

I would like to pull requests to fix these issues then found out these repositories are not longer maintained when looking all the pull requests that are not handled yet.

Another inconvenience is that all these repositories load all data to GPU at once. This makes training fast, but not friendly if you want to add another network with NeRF for poor researchers who do not have a 24G or 32GB GPU.

So, I decided to write a pure PyTorch based NeRF.

## Standing on the shoulders of giants:

This code is based on [yenchenlin's nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).

The difference is that:

0. Code is separated within reasonable sub models.
1. Use cpu to load data instead of loading all data to GPU at once (currently only support Blender dataset).
3. Use "yaml" for configuration.
4. Fix some bugs.

### TODO:

Write a separate test script.

Make the code in the render part more readable.

## How to use
just type:

```
python train_nerf.py --config configs/lego.yml
```
