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


class RayDataset(Dataset):
    def __init__(self, width, height, focal_length=1, requires_grad=True, orthogonal=False, pose=None):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.orthogonal = orthogonal

        self.requires_grad = requires_grad

        self.depth = torch.zeros((width*height, 1))
        self.depth.requires_grad_(requires_grad)

        if pose is None:
            pose = PointTransform()
        self.apply_pose(pose)

    def set_focal_length(focal_length):
        self.focal_length = focal_length
        self.mm[2] = focal_length
        self.mx[2] = focal_length
        self.t = (self.mx - self.mm) / self.shape

    def apply_pose(self, pose):
        width = self.width
        height = self.height 
        focal_length = self.focal_length

        self.x0 = torch.zeros((3), dtype=torch.float)

        self.p = torch.from_numpy(np.mgrid[:width, :height]).permute(2,1,0).reshape(-1,2) + 0.5
        self.p = self.p / torch.tensor([width, height])

        self.n = torch.cat([self.p, torch.ones((self.p.shape[0], 1)) * focal_length], dim=1)
        self.p = torch.cat([self.p, torch.zeros((self.p.shape[0], 1))], dim=1)

        self.pose = pose
        self.x0 = self.pose(self.x0)
        self.p = self.pose(self.p)
        self.n = self.pose(self.n, False)

        self.ortho_n = torch.zeros((3), dtype=torch.float)
        self.ortho_n[2] = 1
        self.ortho_n = self.pose(self.ortho_n, False)

    def reset_depth(self):
        self.depth = torch.zeros(self.width*self.height)
        self.depth.requires_grad_(self.requires_grad)

    def __len__(self):
        return self.width * self.height
    
    def __getitem__(self, idx):
        
        d = self.depth[idx]
        
        if self.orthogonal:
            p = self.p[idx]
            n = self.ortho_n
        else:
            p = self.x0
            n = self.n[idx]

        return {'p': p, 'd': d, 'n': n}


class SceneRayDataset(RayDataset):
    def __init__(self, instance_dir, img_sidelength, idx):
        # get pose from instance
        intrinsics, _, _, _ = utils.parse_intrinsics(os.path.join(instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=img_sidelength)
        focal_length = img_sidelength / intrinsics[0,0]

        pose_dir = os.path.join(instance_dir, "pose")
        pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        pose = utils.load_pose(pose_paths[idx])
        pose = torch.from_numpy(pose).float()

        posetrans = PointTransform(rotation=pose[:3, :3], translation=pose[:3, 3].T)

        super(SceneRayDataset,self).__init__(img_sidelength, img_sidelength, focal_length=focal_length, pose=posetrans)

        param_dir = os.path.join(instance_dir, "params")

        color_dir = os.path.join(instance_dir, "rgb")
        color_paths = sorted(utils.glob_imgs(color_dir))

        self.rgb = utils.load_rgb(color_paths[idx], sidelength=img_sidelength)
        self.rgb = self.rgb.reshape(3, -1).transpose(1, 0)
        self.rgb = torch.from_numpy(self.rgb)
        
        self.visible = self.rgb.sum(dim=-1,keepdim=True) != 3
    

    def __getitem__(self, idx):
        #print("SceneRayDataset : ", time.time())

        ret = super(SceneRayDataset, self).__getitem__(idx)
        ret['rgb'] = self.rgb[idx]
        ret['visible'] = self.visible[idx]
        
        return ret

class FixedSampler(Sampler):
    def __init__(self, ind):
        self.ind = ind
    def __iter__(self) -> Iterator[int]:
        return iter(self.ind)
    def __len__(self):
        return len(self.ind)

class SceneDataset(Dataset):
    def __init__(self, instance_dir, img_sidelength, ray_batch_size):
        self.instance_dir = instance_dir
        self.img_sidelength = img_sidelength
        self.ray_batch_size = ray_batch_size
    
    def __len__(self):
        return len(glob(os.path.join(self.instance_dir, os.path.join("rgb/", "*"))))
    
    def __getitem__(self, idx):
        if type(idx) is list:
            return dict_collate_fn([self.__getitem__(i) for i in idx])
        elif type(idx) is slice:
            return dict_collate_fn([self.__getitem__(i) for i in range(idx.start, idx.stop, 1 if idx.step is None else idx.step)])
        else:  
            print("SceneDataset : ", time.time())

            ds = SceneRayDataset(self.instance_dir, img_sidelength=self.img_sidelength, idx=idx)
            
            ranges = np.array(list(range(len(ds.visible))))
            visible_idx = ranges[ds.visible.view(-1).numpy()]
            invisible_idx = ranges[~ds.visible.view(-1).numpy()]

            perm1 = permutation(len(visible_idx))[:self.ray_batch_size - self.ray_batch_size//2]
            perm2 = permutation(len(invisible_idx))[:self.ray_batch_size//2]

            indices = np.concatenate([visible_idx[perm1], invisible_idx[perm2]])
            indices = indices.tolist()

            dl = DataLoader(ds, 
                            batch_size=self.ray_batch_size,
                            sampler=FixedSampler(indices),
                            shuffle=False,
                            batch_sampler=None)
            
            return next(iter(dl))

class InstanceDataset(Dataset):
    def __init__(self, dataset_dir, img_sidelength, batch_size, ray_batch_size):
        self.dataset_dir = dataset_dir
        self.img_sidelength = img_sidelength
        self.batch_size = batch_size
        self.ray_batch_size = ray_batch_size
    
    def __len__(self):
        return len(glob(os.path.join(self.dataset_dir, "*")))
    
    def __getitem__(self, idx):
        instance_dir = sorted(glob(os.path.join(self.dataset_dir, "*")))[idx]
        ds = loaders.SceneDataset(instance_dir, 
                          img_sidelength=self.img_sidelength, 
                          ray_batch_size=self.ray_batch_size)
        
        dl = DataLoader(ds, 
                        batch_size=self.batch_size,
                        shuffle=True)
        
        return next(iter(dl))


### Data Augmentation Modules ###

def dict_collate_fn(batch):
    # concat a list of dicts into a single dict with lists as values
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

class PointTransform(nn.Module):
    def __init__(self, rotation = None, translation = None):
        super().__init__()

        if rotation is None:
            rotation = torch.eye(3)
        if translation is None:
            translation = torch.zeros((1,3))

        self.rotation = rotation
        self.translation = translation

    def forward(self, data, translate = True):
        in_dim = self.rotation.shape[1]
        out_dim = self.rotation.shape[0]
        original_shape = data.shape

        if data.device != self.rotation.device:
            rotation = self.rotation.to(data.device)
        else:
            rotation = self.rotation

        data = torch.matmul(data.view(-1, in_dim), rotation.T)

        if translate:
            if data.device != self.translation.device:
                translation = self.translation.to(data.device)
            else:
                translation = self.translation

            data += translation.view(1, out_dim)

        data = data.view(*(original_shape[:-1]), out_dim)

        return data