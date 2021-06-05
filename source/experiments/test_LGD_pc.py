import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import Siren
from loaders import ObjDataset, ObjUniformSample, Dataset, UniformSample, GridDataset, PointTransform
from models.LGD import LGD, detach_var

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse

from sklearn.neighbors import KDTree


def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--lgd-weight', dest='lgd_weight', metavar='PATH', default=None, 
                            help='pretrained weight for LGD model')

    parser.add_argument('--sdf-weight', dest='sdf_weight', metavar='PATH', default=None, 
                            help='pretrained weight for SDF model')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--n', dest='n', type=int,metavar='N', default=30000, 
                            help='number of points to sample')

    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    n = args.n
    epoch = args.epoch
    lgd_step_per_epoch = args.lgd_step_per_epoch

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True).to(device) 

    if args.sdf_weight != None:
        try:
            model.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    model.eval() 
    for param in model.parameters():
        param.requires_grad = False

    
    ds = ObjDataset(args.data)
    sampler = ObjUniformSample(n)
    
    p = (sampler(ds)['p']).to(device)
    
    # load 
    with torch.no_grad():
        mm = torch.min(p, dim=0)[0]
        mx = torch.max(p, dim=0)[0]

        x = torch.rand(n,3).to(device) * (mx - mm) + mm
        x.requires_grad_(True)

        x_original = x.clone().detach()
    
    origin_eval = lambda x: torch.pow(x_original - x, 2).sum(dim=1).mean()
    sdf_eval = lambda x: torch.pow(model(x)[0], 2).sum(dim=1).mean()
    
    origin_eval_list = lambda x: origin_eval(x[0])
    sdf_eval_list = lambda x: sdf_eval(x[0])

    print("lgd")
    hidden = None

    lgd = LGD(3, 2, k=10).to(device)

    if args.lgd_weight != None:
        try:
            lgd.load_state_dict(torch.load(args.lgd_weight))
        except:
            print("Couldn't load pretrained weight: " + args.lgd_weight)

    # test LGD
    lgd.eval()
    for i in range(epoch):
        # evaluate losses
        loss = sdf_eval(x).mean()
        # update x
        [x], hidden = lgd.step(x, [origin_eval_list, sdf_eval_list], hidden, n)
        x = detach_var(x)
        hidden = detach_var(hidden)

        if i%10 == 0:
            writer.add_scalars("regression_loss", {"LGD": loss}, global_step=i)
            writer.add_mesh("point cloud regression_LGD", x.unsqueeze(0), global_step=i)
            writer.add_scalars("chamfer_distance", {"LGD": chamfer_distance(x, p)}, global_step=i)

            torch.save(lgd.state_dict(), args.weight_save_path+'lgd_%03d.pth' % epoch)
        
    writer.close()

if __name__ == "__main__":
    main()
