import os
from data.base_dataset import BaseDataset
from data.base_dataset import make_dataset, pose_spherical
from PIL import Image
import json
import imageio
import cv2
import numpy as np
import torch


class BlenderDataset(BaseDataset):
    def __init__(self, opt, phase='train'):
        BaseDataset.__init__(self, opt)
        # self.data_dir = os.path.join(opt.datadir, phase) 
        meta = None
        with open(os.path.join(opt.datadir, 'transforms_{}.json'.format(phase)), 'r') as fp:
            meta = json.load(fp)
        self.img_paths, self.poses = make_dataset(meta, opt.datadir, phase, opt.testskip)

        self.H, self.W = imageio.imread(self.img_paths[0]).shape[:2]
        self.camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)#
        if opt.half_res:
            self.H = self.H//2
            self.W = self.W//2
            self.focal = self.focal/2.

        self.render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)


    def __getitem__(self, index):
        img = imageio.imread(self.img_paths[index])
        img = (np.array(img) / 255.).astype(np.float32)

        # self.H, self.W = img.shape[:2]
        if self.opt.half_res:
            # H = H//2
            # W = W//2
            # self.focal = self.focal/2.
            img_half_res = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            img = img_half_res

        pose = self.poses[index]
        pose = np.array(pose).astype(np.float32)

        # img = torch.from_numpy(img)
        # pose = torch.from_numpy(pose)

        return {'img': img, 'pose': pose, 'index': index}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)

    def get_render_poses(self):
        return self.render_poses

    def get_misc(self):
        return [self.H, self.W, self.focal]

    def get_poses(self):
        return np.asarray(self.poses)