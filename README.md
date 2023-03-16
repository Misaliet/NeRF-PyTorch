# NeRF-PyTorch
## Update:
Use [yenchenlin's nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) code as foundation.

Currently separate main code to reasonable sub-module code.

Use "yaml" for configuration.

Fix depth video rendering issue.

### TODO:
Use PyTorch dataloader to save GPU memeory.

Write a separate test script.

Make the code in the render part more readable.

## Doing research on NeRF now, will write a pure PyTorch based NeRF later.
The pure PyTorch version of the NeRF code that can be found so far is not perfect or has some problems. For example, claiming to be ~5-9x faster than offical code but actually uses a smaller NeRF model to "cheat"; massive bugs in code that are not tested (like using "half_res: False" or using NeRF paper settings); or all the training data is copied to GPU at one time for efficiency which is not suitable for poor researchers.
I noticed that authors of these repositories have stopped processing pull requests to fix bugs or improve quality. So, I will write a new repository.
