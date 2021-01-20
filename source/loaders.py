import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset, DataLoader
import re
import warnings

class ObjDataset(Dataset):

    def __init__(self, obj_path):
        # read an obj file
        obj_file = open(obj_path, 'r')
        obj = obj_file.read()
        obj_file.close()

        vpattern = r"(?:v)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        fpattern = r"(?:f)\s+(\d+)(?:\/\d?){0,2}\s+(\d+)(?:\/\d?){0,2}\s+(\d+)(?:\/\d?){0,2}"

        v = re.findall(vpattern, obj)
        f = re.findall(fpattern, obj)

        v = torch.tensor([list(v_) for v_ in v], dtype=torch.float)
        f = torch.tensor([list(f_) for f_ in f], dtype=torch.long)

        fn = v[f]
        print(fn.shape)

        self.v = v
        self.f = f

    def __len__(self):
        raise len(self.v)

    def __getitem__(self, idx):
        return {'xyz': self.v[idx]}

ObjDataset("../../../data/teapot.obj")