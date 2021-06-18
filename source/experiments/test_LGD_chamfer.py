import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from loaders import PSGDataset
from models.LGD import LGD, detach_var
from evaluate_functions import chamfer_distance, nearest_from_to, dist_from_to

from knn import knn

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse

from sklearn.neighbors import KDTree



def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--lgd-weight', dest='lgd_weight', metavar='PATH', default='../weights/', 
                            help='pretrained weight for LGD model')

    parser.add_argument('--idx', dest='idx', metavar='INDEX', type=int, default=0, 
                            help='index to test')

    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--perturb', dest='perturb', type=float,metavar='PERTURBATION', default=1e-2, 
                            help='perturbation')
    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    epoch = args.epoch
    perturb = args.perturb
    batchsize = args.batchsize


    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ds = PSGDataset(args.data)
    dl = DataLoader(ds, batch_size=batchsize, shuffle=False)
    dl_iter = iter(dl)

    knn_f = knn.apply

    chamfer_dist = lambda x, y: knn_f(x, y, 1).mean() + knn_f(y, x, 1).mean()
    chamfer_dist_list = lambda x: torch.cat([chamfer_dist(x[0][i], x_gt[i]).unsqueeze(0) for i in range(x_gt.shape[0])]).mean()
    
    print("lgd")
    hidden = None

    lgd = LGD(3, 1, k=10).to(device)

    if args.lgd_weight != None:
        try:
            lgd.load_state_dict(torch.load(args.lgd_weight))
        except:
            print("Couldn't load pretrained weight: " + args.lgd_weight)

    # test LGD
    lgd.eval()
    loss = 0
    for sample_batched in dl_iter:
        x_gt = sample_batched['pc_gt'].reshape(-1,16384,3).to(device)
        ind = [torch.randperm(16384)[:512] for i in range(x_gt.shape[0])]
        x = torch.cat([x_gt[i][ind[i]].unsqueeze(0) for i in range(x_gt.shape[0])]).detach_()

        x += torch.randn_like(x) * perturb
        x.requires_grad_()

        loss_old = sum([chamfer_dist(x[i], x_gt[i]) for i in range(x.shape[0])])

        for i in range(epoch):
            # update x
            [x], hidden = lgd.step(x, [chamfer_dist_list], hidden)
            x = detach_var(x)
            hidden = detach_var(hidden)
        
        loss_ = sum([chamfer_dist(x[i], x_gt[i]) for i in range(x.shape[0])])
        print(loss_old.item(), loss_.item())
        loss += loss_

    print("chamfer dist mean: ", loss.item() / len(ds)*32)    
    
    writer.close()


if __name__ == "__main__":
    main()
