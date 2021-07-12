import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Sampler, SequentialSampler, Dataset, DataLoader, WeightedRandomSampler
import re
import warnings

import pickle
from torch.utils.tensorboard import SummaryWriter
import argparse

import math
import os
from glob import glob

from collections import defaultdict
import numpy as np
from numpy.random import permutation

import utils
import time
from typing import Iterator
class ObjDataset(Dataset):
    # reads an obj file, and outputs a single point sampled at i-th face

    @torch.no_grad()
    def __init__(self, obj_path):
        # read an obj file
        obj_file = open(obj_path, 'r')
        obj = obj_file.read()
        obj_file.close()

        # wish we could do "v %f %f %f" and "f %d %d %d" lol
        vpattern = r"(?:v)\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)"
        fpattern = r"(?:f)\s+(\d+)(?:\/\d*){0,2}\s+(\d+)(?:\/\d*){0,2}\s+(\d+)(?:\/\d*){0,2}"

        v = re.findall(vpattern, obj)
        f = re.findall(fpattern, obj)

        v = torch.tensor([list(map(float, v_)) for v_ in v], dtype=torch.float)
        f = torch.tensor([list(map(lambda x: int(x)-1, f_)) for f_ in f], dtype=torch.long)     # index starts from 1 in obj, so substract 1

        
        # remove duplicate faces
        f_sorted, _ = torch.sort(f, 1)
        f_unique, ind2 = torch.unique(f_sorted, dim=0, return_inverse=True)

        check = torch.zeros(f_unique.shape[0]).bool()
        for i,t in enumerate(ind2):
            if not check[t]:
                f_unique[t] = f[i]
                check[t] = True

        f = f_unique
        

        vf = v[f]

        # obtain face normal with cross vector
        a1 = - vf[:,0] + vf[:,1] 
        a2 = - vf[:,1] + vf[:,2]
        
        fn = torch.cat([t.unsqueeze(1) for t in 
           [a1[:,1] * a2[:,2] - a1[:,2] * a2[:,1],
            a1[:,2] * a2[:,0] - a1[:,0] * a2[:,2],
            a1[:,0] * a2[:,1] - a1[:,1] * a2[:,0]]], dim=1)
        
        # add face normal to connected vertices
        vn = torch.zeros_like(v)
        vn = vn.index_add(0, f.reshape(-1), fn.repeat(1,3).reshape(-1,3))

        vn.index_add(0, f.view(-1), fn.repeat(1,3).reshape(-1,3))

        fnn = torch.norm(fn, dim=1, keepdim=True)
        vnn = torch.norm(vn, dim=1, keepdim=True)

        # normalization
        fn = fn / fnn
        fn = torch.where(torch.isnan(fn), torch.zeros_like(fn), fn)

        # normalization
        vn = vn / vnn
        vn = torch.where(torch.isnan(vn), torch.zeros_like(vn), vn)


        print(torch.norm(vn, dim=1).mean())

        self.v = v
        self.f = f
        self.vn = vn
        self.fn = fn
        self.vnn = vnn
        self.fnn = fnn

    
    def __len__(self):
        return len(self.f)
    
    @torch.no_grad()
    def __getitem__(self, idx):
        f = self.f[idx]
        v = self.v[f]
        vn = self.vn[f]

        a = 1
        b = 1
        while a+b > 1:
            r = torch.rand(2)
            a = r[0]
            b = r[1]

        # barycentric interpolation
        p = (1-a-b) * v[0] + a*v[1] + b*v[2]
        n = (1-a-b) * vn[0] + a*vn[1] + b*vn[2]
        n /= torch.norm(n)
        #n = self.fn[idx]

        return {'p': p,
                'n': n}

        #return {'p': self.v[idx], 'n': self.fn[idx]}

    def to_obj(self):
        obj_file = open("test.obj", 'w')

        for i,v in enumerate(self.v):
            obj_file.write(f"v {v[0]} {v[1]} {v[2]} {self.vn[i,0]} {self.vn[i,1]} {self.vn[i,2]}\n")
        
        for f in self.f:
            obj_file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

        #obj = obj_file.read()
        obj_file.close()

class PlyDataset(Dataset):
    def __init__(self):
        raise NotImplementedError


class InstanceDataset(Dataset):
    def __init__(self, instance_dir, img_sidelength):
        self.instance_dir = instance_dir
        self.img_sidelength = img_sidelength
    
    def __len__(self):
        return len(glob(os.path.join(self.instance_dir, os.path.join("rgb/", "*"))))
    
    @torch.no_grad()
    def __getitem__(self, idx):
        if type(idx) is list:
            return dict_collate_fn([self.__getitem__(i) for i in idx])
        elif type(idx) is slice:
            return dict_collate_fn([self.__getitem__(i) for i in range(idx.start, idx.stop, 1 if idx.step is None else idx.step)])
        else:
            intrinsics, _, _, _ = utils.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=self.img_sidelength)
            intrinsics = torch.from_numpy(intrinsics).float()

            pose_dir = os.path.join(self.instance_dir, "pose")
            pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

            pose = utils.load_pose(pose_paths[idx])
            pose = torch.from_numpy(pose).float()

            # TODO fuse intrinsics & pose to create 3D -> pixel transform. shape: (12)
            pose = torch.mm(intrinsics, torch.inverse(pose))[:3].T.view(-1)

            color_dir = os.path.join(instance_dir, "rgb")
            color_paths = sorted(utils.glob_imgs(color_dir))

            rgb = utils.load_rgb(color_paths[idx], sidelength=img_sidelength)
            rgb = rgb.reshape(3, -1).transpose(1, 0) * 2 - 1
            rgb = torch.from_numpy(rgb)

            return {'pose': pose, 'rgb': rgb}

class CategoryDataset(Dataset):
    def __init__(self, srn_dir, shapenet_dir, img_sidelength, batch_size, ray_batch_size):
        self.srn_dir = srn_dir
        self.shapenet_dir = shapenet_dir

        self.img_sidelength = img_sidelength
        self.batch_size = batch_size
        self.ray_batch_size = ray_batch_size
    
    def __len__(self):
        return len(glob(os.path.join(self.srn_dir, "*")))
    
    @torch.no_grad()
    def __getitem__(self, idx):
        instance_dir = sorted(glob(os.path.join(self.srn_dir, "*")))[idx]
        instance_id = os.path.relpath(instance_dir, start=self.srn_dir)

        scenes_ds = InstanceDataset(instance_dir, 
                          img_sidelength=self.img_sidelength)
        
        scenes_dl = DataLoader(ds, 
                        batch_size=self.batch_size,
                        shuffle=True)
        
        scenes_dict = next(iter(scenes_dl))

        # TODO implement a script to find category num
        category_num = 0
        obj_dir = os.path.join(self.shapenet_dir, os.path.join(category_num, instance_id))
        obj_dir = os.path.join(obj_dir, os.path.join("models", "model_normalized.obj"))

        mesh_ds = ObjDataset(obj_dir)
        mesh_dl = get_obj_dataloader(mesh_ds, self.ray_batch_size, num_workers=0)
        mesh_dict = next(iter(mesh_dl))

        # projection
        X = mesh_dict['p']
        P = scenes_dict['pose']

        P = P.view(*P.shape[:-1], 4, 3)                             # (m, 4, 3)
        X = X.unsqueeze(-3)                                         # (1, n, 3)
        X = torch.cat([X, torch.ones_like(X)[..., :1]], dim=-1)     # (1, n, 4)
        X = torch.matmul(X, P)                                      # (m, n, 3)
        x_hat = X[..., :-1] / X[..., -1]                            # (m, n, 2)


        return dict_collate_fn([scenes_dict, mesh_dict, {'uv': x_hat}])


### Data Augmentation Modules ###

def dict_collate_fn(batch):
    # concat a list of dicts into a single dict with concated lists
    # [{'a': x}, {'a': y}] -> {'a': [x, y]}

    d = defaultdict(list)

    for item in batch:
        if type(item) is dict:
            for key, value in item.items():
                d[key].append(value)
        else:
            d[None].append(item)
    
    keylist = list(d.keys())

    if len(keylist) == 1 and keylist[0] is None:
        return torch.cat([t.unsqueeze(0) for t in d[None]])
    
    else:
        for key, value in d.items():
            d[key] = torch.cat([t.unsqueeze(0) for t in value])
        
        return dict(d)


def get_obj_dataloader(dataset, num_samples, num_workers=0):
    fnn = torch.abs(dataset.fnn)
    sampler = WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), num_samples, replacement=True)

    return DataLoader(dataset, 
                      batch_size=num_samples, 
                      shuffle=False, 
                      sampler=sampler, 
                      batch_sampler=None, 
                      num_workers=num_workers,
                      collate_fn=dict_collate_fn)

def dict_to_device(d, device):
    for key, value in d.items():
        d[key] = value.to(device)
    return d