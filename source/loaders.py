import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader
import re
import warnings

class ObjDataset(Dataset):

    @torch.no_grad()
    def __init__(self, obj_path):
        # read an obj file
        obj_file = open(obj_path, 'r')
        obj = obj_file.read()
        obj_file.close()

        vpattern = r"(?:v)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        fpattern = r"(?:f)\s+(\d+)(?:\/\d?){0,2}\s+(\d+)(?:\/\d?){0,2}\s+(\d+)(?:\/\d?){0,2}"

        v = re.findall(vpattern, obj)
        f = re.findall(fpattern, obj)

        v = torch.tensor([list(map(float, v_)) for v_ in v], dtype=torch.float)
        f = torch.tensor([list(map(lambda x: int(x)-1, f_)) for f_ in f], dtype=torch.long)

        vf = v[f]

        a1 = vf[:,0] - vf[:,1] 
        a2 = vf[:,1] - vf[:,2]

        fn = torch.cat([t.unsqueeze(1) for t in 
           [a1[:,1] * a2[:,2] - a1[:,2] * a2[:,1],
            a1[:,2] * a2[:,0] - a1[:,0] * a2[:,2],
            a1[:,0] * a2[:,1] - a1[:,1] * a2[:,0]]], dim=1)
        fn = fn / torch.norm(fn, dim=1, keepdim=True)

        vn = torch.zeros_like(v)

        vn[f] = vn[f].add_(fn.unsqueeze(1).repeat(1,3,1))

        print(vn)

        vn = vn / torch.norm(vn, dim=1, keepdim=True)

        print(vn)
    
        self.v = v
        self.f = f
        self.vn = vn
        self.fn = fn

    def __len__(self):
        raise len(self.v)

    def __getitem__(self, idx):
        return {'xyz': self.v[idx]}

ObjDataset("../../../data/teapot.obj")