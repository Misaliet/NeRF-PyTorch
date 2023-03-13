# NeRF-PyTorch

## Doing research on NeRF now, will write a pure PyTorch based NeRF later.
The pure PyTorch version of the NeRF code that can be found so far is not perfect or has some problems. For example, claiming to be ~5-9x faster than offical code but actually uses a smaller NeRF model to "cheat"; massive bugs in code that are not tested (like using "half_res: False" or using NeRF paper settings); or all the training data is copied to GPU at one time for efficiency which is not suitable for poor researchers.
