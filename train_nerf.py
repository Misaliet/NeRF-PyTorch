import os, sys
import numpy as np
import imageio
import json
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import argparse
import yaml

from data import load_llff_data, load_dv_data, load_blender_data, load_LINEMOD_data

from models import Embedder, NeRF, TinyNeRF
from misc import CfgNode, print_current_losses
from render import render, batchify, render_path
from render.render_helpers import *
from data import create_dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
# DEBUG = False


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    # skips = [4]
    skips = [args.skips]
    if args.model_name == 'TinyNeRF':
        model = TinyNeRF(input_ch=input_ch, input_ch_views=input_ch_views).to(device)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def train():

    # parser = config_parser()
    # args = parser.parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    args = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        args = CfgNode(cfg_dict)

    # Load data
    K = None
    
    dataset = create_dataset(args, 'train')
    hwf = dataset.get_misc()
    render_poses = dataset.get_render_poses().to(device)

    near = 2.
    far = 6.

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    K = torch.Tensor(K).to(device)

    # test set
    testset = create_dataset(args, 'test')
    poses = testset.get_poses()

    # # FIXME:
    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.yml')
    with open(f, 'w') as file:
        args.dump()

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # # TODO: check batch later
    # # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    # if use_batching:
    #     # For random ray batching
    #     print('get rays')
    #     rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    #     print('done, concats')
    #     rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    #     rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    #     rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    #     rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    #     rays_rgb = rays_rgb.astype(np.float32)
    #     print('shuffle rays')
    #     np.random.shuffle(rays_rgb)

    #     print('done')
    #     i_batch = 0

    # # TODO: check batch later
    # # Move training data to GPU
    # if use_batching:
    #     images = torch.Tensor(images).to(device)
    # poses = torch.Tensor(poses).to(device)
    # if use_batching:
    #     rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.i_iters + 1
    # change iter numbers to epoch numbers
    # iterator = iter(dataset)
    total_iters = args.i_iters + 1
    epoch = math.ceil(args.i_iters/len(dataset)) + 1
    
    print('Begin')
    
    start = start + 1
    pbar = tqdm(total=args.i_iters, position=0, leave=True)
    for i in range(start, epoch):
        time0 = time.time()

        # TODO: check batch later
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            for j, data in enumerate(dataset):
                cur_iter = (i-1) * len(dataset) + j + 1

                target = data['img']
                if args.white_bkgd:
                    target = target[...,:3]*target[...,-1:] + (1.-target[...,-1:])
                else:
                    target = target[...,:3]
                target = torch.Tensor(target).squeeze().to(device)
                pose = data['pose'][:, :3,:4].squeeze().to(device)

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if cur_iter < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if cur_iter == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                #####  Core optimization loop  #####
                rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                        verbose=cur_iter < 10, retraw=True,
                                                        **render_kwargs_train)

                optimizer.zero_grad()
                img_loss = img2mse(rgb, target_s)
                trans = extras['raw'][...,-1]
                loss = img_loss
                psnr = mse2psnr(img_loss)

                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s)
                    loss = loss + img_loss0
                    psnr0 = mse2psnr(img_loss0)

                loss.backward()
                optimizer.step()

                # NOTE: IMPORTANT!
                ###   update learning rate   ###
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                ################################

                dt = time.time()-time0
                # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
                #####           end            #####

                # Rest is logging
                if cur_iter%args.i_weights==0:
                    path = os.path.join(basedir, expname, '{:06d}.tar'.format(cur_iter))
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': None if not render_kwargs_train['network_fine'] else render_kwargs_train['network_fine'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    print('Saved checkpoints at', path)

                if cur_iter%args.i_video==0 and cur_iter > 0:
                    # Turn on testing mode
                    with torch.no_grad():
                        rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                    print('Done, saving', rgbs.shape, disps.shape)
                    moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, cur_iter))
                    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                    # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
                    imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.nanmax(disps)), fps=30, quality=8)

                    # if args.use_viewdirs:
                    #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                    #     with torch.no_grad():
                    #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                    #     render_kwargs_test['c2w_staticcam'] = None
                    #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

                if cur_iter%args.i_testset==0 and cur_iter > 0:
                    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(cur_iter))
                    os.makedirs(testsavedir, exist_ok=True)
                    print('test poses shape', poses.shape)
                    with torch.no_grad():
                        render_path(torch.Tensor(poses).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir)
                    print('Saved test set')

                if cur_iter%args.i_print==0:
                    # print(f"[TRAIN] Iter: {cur_iter} Loss: {loss.item()}  PSNR: {psnr.item()}")
                    log_name = os.path.join(basedir, expname, 'loss.txt')
                    message = f"[TRAIN] Iter: {cur_iter} Loss: {loss.item()}  PSNR: {psnr.item()}"
                    print_current_losses(log_name, message)
                    pbar.write(message)

                global_step += 1
                pbar.update(cur_iter - pbar.n)
                # pbar.display(message, cur_iter)
                # pbar.update()


if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
