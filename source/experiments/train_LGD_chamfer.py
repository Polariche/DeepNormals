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
from loaders import PSGDataset
from models.LGD import LGD, detach_var
from evaluate_functions import chamfer_distance, nearest_from_to, dist_from_to

import knn_cuda

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse

from sklearn.neighbors import KDTree


def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--weight-save-path', dest='weight_save_path', metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=1e-3, 
                            help='learning rate')
    parser.add_argument('--lgd-step', dest='lgd_step_per_epoch', type=int,metavar='LGD_STEP_PER_EPOCH', default=5, 
                            help='number of simulation steps of LGD per epoch')
    parser.add_argument('--n', dest='n', type=int,metavar='N', default=30000, 
                            help='number of points to sample')

    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    n = args.n
    lr = args.lr
    epoch = args.epoch
    lgd_step_per_epoch = args.lgd_step_per_epoch

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ds = PSGDataset(args.data)

    x = torch.cat([ds[i]['pc_pred'] for i in range(len(ds))])
    x_gt = torch.cat([ds[i]['pc_gt'] for i in range(len(ds))])

    chamfer_dist = lambda x, y: knn_cuda(x, y, 1).mean() + knn_cuda(y, x, 1).mean()
    chamfer_dist_list = lambda x: sum([chamfer_dist(x[i * 1024:i * 1024 + 1024], x_gt[i * 1024:i * 1024 + 1024]) for i in range(len(ds))])

    print("lgd")
    hidden = None

    lgd = LGD(3, 1, k=10).to(device)
    lgd_optimizer = optim.Adam(lgd.parameters(), lr= lr)

    # train LGD
    lgd.train()
    for i in range(epoch):
        print(i)
        # evaluate losses

        # update lgd parameters
        lgd_optimizer.zero_grad()
        lgd.loss_trajectory_backward(x, [chamfer_dist_list], None, 
                                     constraints=["None"], batch_size=1024 * len(ds), steps=lgd_step_per_epoch)
        lgd_optimizer.step()

        torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
        
    writer.close()

if __name__ == "__main__":
    main()
