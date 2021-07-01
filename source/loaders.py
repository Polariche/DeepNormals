import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader
import re
import warnings

import pickle
from torch.utils.tensorboard import SummaryWriter
import argparse

import math
import os
from glob import glob

from collections import defaultdict

import utils

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

        self.depth = torch.zeros(width*height)
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

        self.p = torch.from_numpy(np.mgrid[:width, :height]).permute(2,1,0).reshape(-1,2) + 0.5
        self.p = self.p / torch.tensor([width, height])
        self.p = torch.cat([self.p, torch.ones((self.p.shape[0], 1)) * focal_length], dim=1)
        self.n = self.p / focal_length

        self.pose = pose
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
        p = self.p[idx]
        d = self.depth[idx]
        
        if self.orthogonal:
            n = self.ortho_n
        else:
            n = self.n[idx]

        return {'p': p, 'd': d, 'n': n}


# refer to SRN/dataio.py for original implementation

class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 instance_dir,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 img_sidelength=None,
                 num_images=-1):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(util.glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        intrinsics, _, _, _ = util.parse_intrinsics(os.path.join(self.instance_dir, "intrinsics.txt"),
                                                                  trgt_sidelength=self.img_sidelength)
        intrinsics = torch.Tensor(intrinsics).float()

        rgb = util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        rgb = rgb.reshape(3, -1).transpose(1, 0)

        pose = util.load_pose(self.pose_paths[idx])

        if self.has_params:
            params = util.load_params(self.param_paths[idx])
        else:
            params = np.array([0])

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "param": torch.from_numpy(params).float(),
            "intrinsics": intrinsics
        }
        return sample


class SceneClassDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 root_dir,
                 img_sidelength=None,
                 max_num_instances=-1,
                 max_observations_per_instance=-1,
                 specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
                 samples_per_instance=2):

        self.samples_per_instance = samples_per_instance
        self.instance_dirs = sorted(glob(os.path.join(root_dir, "*/")))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [SceneInstanceDataset(instance_idx=idx,
                                                   instance_dir=dir,
                                                   specific_observation_idcs=specific_observation_idcs,
                                                   img_sidelength=img_sidelength,
                                                   num_images=max_observations_per_instance)
                              for idx, dir in enumerate(self.instance_dirs)]

        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        for i in range(self.samples_per_instance - 1):
            observations.append(self.all_instances[obj_idx][np.random.randint(len(self.all_instances[obj_idx]))])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]

        return observations, ground_truth


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


def get_obj_dataloader(dataset, num_samples, num_workers=0, batch_size=1):
    fnn = torch.abs(dataset.fnn)
    sampler = WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), self.sample_n, replacement=True)

    return DataLoader(dataset, 
                      batch_size=batch_size, 
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