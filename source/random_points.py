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
from torch.utils.data import  DataLoader, WeightedRandomSampler

from sklearn.neighbors import KDTree

from LGD import LGD, apply_step

import argparse

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
    samples = list(WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), 50000, replacement=True))

    data = [ds[samples[i]] for i in range(len(samples))]
    xyz = torch.cat([d['xyz'].unsqueeze(0) for d in data])
    
    tree_original = KDTree(xyz.cpu().numpy())

    # load 
    with torch.no_grad():
        x = (torch.rand(50000,3) - 0.5)
        x = x.to(device)
        x.requires_grad_(True)

    optimizer = optim.Adam([x], lr = 1e-3)

    for i in range(500):
        optimizer.zero_grad()

        s, x = model(x)
        (torch.pow(s, 2)).mean().backward()

        optimizer.step()

        if i%10 == 0:
            x_ = x.cpu().detach().numpy()
            tree_new = KDTree(x_)

            d1, _ = np.power(tree_original.query(x_, k=1),2)
            d2, _ = np.power(tree_new.query(xyz.cpu().numpy(), k=1),2)

            cd = (np.sum(d1) + np.sum(d2)) / 50000

            writer.add_mesh("point cloud regression", x.unsqueeze(0), colors=(F.pad(torch.tensor(d1), (0,2)).unsqueeze(0) / 0.0001 * 256).int(), global_step=i)
            writer.add_scalar("chamfer distance", cd, global_step=i)
    
    writer.close()

if __name__ == "__main__":
    main()
