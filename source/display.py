import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import Siren
from utils import Sobel
from loaders import ObjDataset
import utils

from sklearn.neighbors import KDTree

import argparse
from torch.utils.data import  DataLoader, WeightedRandomSampler

parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DATA', help='path to file')

parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                        help='tensorboard checkpoints path')

parser.add_argument('--weight-save-path', dest='weight_save_path', metavar='PATH', default='../weights/', 
                        help='weight checkpoints path')

parser.add_argument('--pretrained-weight', dest='weight', metavar='PATH', default=None, 
                        help='pretrained weight')

parser.add_argument('--activation', dest='activation', metavar='activation', default='relu', 
                        help='activation of network; \'relu\' or \'sin\'')

parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                        help='batch size')
parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=100, 
                        help='epochs')

parser.add_argument('--epsilon', dest='epsilon', type=float, metavar='EPSILON', default=0.1, 
                        help='epsilon')
parser.add_argument('--omega', dest='omega', type=float, metavar='OMEGA', default=30, 
                        help='hyperparameter for periodic layer')
parser.add_argument('--lambda', dest='lamb', type=float, metavar='LAMBDA', default=0.005, 
                        help='hyperparameter for s : normal loss ratio')


parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')


def dist_from_to(p1, p2, requires_graph=True):
    p1_np = p1.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()

    p2_tree = KDTree(p2_np)

    d, ind = p2_tree.query(p1_np)

    if not requires_graph:
        # we don't need graph
        return torch.tensor(d, device=p1.device)   

    else:
        ind = torch.tensor(ind, device=p1.device)
        return torch.norm(p1 - p2[ind], dim=1, keepdim=True)

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True).to(device) 

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.weight))
        except:
            print("Couldn't load pretrained weight: " + args.weight)

    ds = ObjDataset(args.data)

    fnn = torch.abs(ds.fnn)
    samples = list(WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), 30000, replacement=True))

    data = [ds[samples[i]] for i in range(len(samples))]
    xyz = torch.cat([d['xyz'].unsqueeze(0) for d in data])
    xyz = xyz.to(device)

    n = 256
    with torch.no_grad():
        voxel_min = torch.min(ds.v, dim=0, keepdim=True, out=None).values.to(device)
        voxel_max = torch.max(ds.v, dim=0, keepdim=True, out=None).values.to(device)

        voxels = (torch.tensor([[i//n,n-i%n,0] for i in range(n*n)], dtype=torch.float) / n).to(device)
        
        ran = voxel_max - voxel_min

        voxels *= ran
        voxels += voxel_min

    for i in range(256):
        z = i/256 * ran[:,2]

        with torch.no_grad():
            voxels_ = voxels.clone().detach()
            voxels_[:,2] += z

        s, g = model(voxels_)
        g = torch.autograd.grad(s, [g], grad_outputs=torch.ones_like(s), create_graph=True)[0]
        g /= torch.norm(g, dim=1, keepdim=True)

        d = torch.clamp(dist_from_to(voxels_, xyz, requires_graph=False) / args.epsilon, -1, 1)

        writer.add_image("implicit", torch.clamp(s.reshape(n,n), 0, 1), i, dataformats='WH')

        writer.add_image("implicit_normals", (g.reshape(n,n,3)*128+128).int(), i, dataformats='WHC')

        writer.add_image("implicit_normals", (d.reshape(n,n)*0.5+0.5).int(), i, dataformats='WH')
    
    writer.close()


if __name__ == "__main__":
    main()
