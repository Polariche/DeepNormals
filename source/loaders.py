import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader, WeightedRandomSampler
import re
import warnings

from torch.utils.tensorboard import SummaryWriter
import argparse


class ObjDataset(Dataset):

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

    """
    def __len__(self):
        return len(self.f)
    """
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


class UniformDataset(Dataset):
    def __init__(self, mm, mx):
        assert mm.shape == mx.shape
        assert mm.device == mx.device

        self.device = mm.device
        self.dim = mm.shape[0]
        self.mm = mm
        self.mx = mx

    def __getitem__(self, idx):
        return torch.rand(1,self.dim, device=self.device) * (self.mx - self.mm) + self.mm


class GridDataset(Dataset):
    def __init__(self, mm, mx, n):
        assert len(mm.shape) == 1
        assert mm.shape == mx.shape and mm.shape == n.shape
        assert mm.device == mx.device and mm.device == n.device

        self.device = mm.device
        self.dim = mm.shape[0]
        self.mm = mm
        self.mx = mx
        self.n = n

    def __getitem__(self, idx):
        a = torch.zeros_like(self.n)
        n_prod = torch.prod(self.n)

        for i, n_ in enumerate(self.n):
            n_prod /= n_
            a[i] = int(idx / n_prod)
            idx -= int(idx / n_prod)

        return (a + 0.5) * (self.mx - self.mm) + self.mm


### Data Augmentation Modules ###

class ObjUniformSample(nn.Module):
    """
    Given an ObjDataset, uniformly sample n points and their normals.
    """
    def __init__(self, sample_n):
        super().__init__()
        self.sample_n = sample_n

    def forward(self, dataset):
        fnn = torch.abs(dataset.fnn)
        samples = list(WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), self.sample_n, replacement=True))

        data = [dataset[samples[i]] for i in range(len(samples))]

        p = torch.cat([d['p'].unsqueeze(0) for d in data])
        n = torch.cat([d['n'].unsqueeze(0) for d in data])

        return {'p': p, 'n': n}


class PerturbNormal(nn.Module):
    """
    Perturb points by their normals, by random amount
    """

    def __init__(self, epsilon = 1.):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, dataset):
        p = dataset['p']
        n = dataset['n']

        assert p.shape[0] == n.shape[0]
        assert p.device == n.device

        dataset_n = p.shape[0]
        device = p.device

        s = torch.rand((dataset_n, 1)).to(device)

        p = torch.cat([p, p + s * n * self.epsilon])
        n = torch.cat([n, n])
        s = torch.cat([torch.zeros(dataset_n, 1).to(device), s])

        return {'p': p, 'n': n, 's': s}

