import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader
import re
import warnings

from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../../../data/pointcloud_test/checkpoints/', 
                        help='tensorboard checkpoints path')

class ObjDataset(Dataset):

    @torch.no_grad()
    def __init__(self, obj_path):
        # read an obj file
        obj_file = open(obj_path, 'r')
        obj = obj_file.read()
        obj_file.close()

        # wish we could do "v %f %f %f" and "f %d %d %d" lol
        vpattern = r"(?:v)\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)"
        fpattern = r"(?:f)\s+(\d+)(?:\/\d?){0,2}\s+(\d+)(?:\/\d?){0,2}\s+(\d+)(?:\/\d?){0,2}"

        v = re.findall(vpattern, obj)
        f = re.findall(fpattern, obj)

        v = torch.tensor([list(map(float, v_)) for v_ in v], dtype=torch.float)
        f = torch.tensor([list(map(lambda x: int(x)-1, f_)) for f_ in f], dtype=torch.long)     # index starts from 1 in obj, so substract 1

        vf = v[f]

        # obtain face normal with cross vector
        a1 = vf[:,0] - vf[:,1] 
        a2 = - vf[:,1] + vf[:,2]
        
        fn = torch.cat([t.unsqueeze(1) for t in 
           [a1[:,1] * a2[:,2] - a1[:,2] * a2[:,1],
            a1[:,2] * a2[:,0] - a1[:,0] * a2[:,2],
            a1[:,0] * a2[:,1] - a1[:,1] * a2[:,0]]], dim=1)

        # normalization
        fn = fn / torch.norm(fn, dim=1, keepdim=True)
        fn[torch.isnan(fn)] = 0.

        vn = torch.zeros_like(v)

        # add face normal to connected vertices
        for i in range(3):
            vn[f[:,i]] = vn[f[:,i]].add_(fn)

        # normalization
        vn = vn / torch.norm(vn, dim=1, keepdim=True)
        vn[torch.isnan(vn)] = 0.
    
        self.v = v
        self.f = f
        self.vn = vn
        self.fn = fn

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        return {'xyz': self.v[idx], 'n': self.vn[idx]}

"""
ds = ObjDataset("../../../data/train/02828884/model_001415.obj")
xyz = ds.v
c = ds.vn

args = parser.parse_args()

writer = SummaryWriter(args.tb_save_path)
writer.add_mesh("teapot", xyz.unsqueeze(0), colors=((c.float().unsqueeze(0)*0.5+0.5)*256).int(), faces=ds.f.unsqueeze(0))
writer.close()
"""