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


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=1e-3, 
                            help='learning rate')
    parser.add_argument('--lgd-step', dest='lgd_step_per_epoch', type=int,metavar='LGD_STEP_PER_EPOCH', default=5, 
                            help='number of simulation steps of LGD per epoch')

    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    lr = args.lr
    epoch = args.epoch
    lgd_step_per_epoch = args.lgd_step_per_epoch
    idx = args.idx

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ds = PSGDataset(args.data)

    x = ds[idx]['pc_pred'].to(device).requires_grad_()
    x_gt = ds[idx]['pc_gt'].to(device)

    knn_f = knn.apply

    chamfer_dist = lambda x, y: knn_f(x, y, 1).mean()# + knn_f(y, x, 1).mean()
    chamfer_dist_list = lambda x: sum([chamfer_dist(x[0][i * 1024:i * 1024 + 1024], x_gt[i * 16384:i * 16384 + 16384]) for i in range(1)])

    print("lgd")
    hidden = None

    lgd = LGD(3, 1, k=10).to(device)
    lgd_optimizer = optim.Adam(lgd.parameters(), lr= lr)

    if args.lgd_weight != None:
        try:
            lgd.load_state_dict(torch.load(args.lgd_weight))
        except:
            print("Couldn't load pretrained weight: " + args.lgd_weight)

    # test LGD
    lgd.eval()
    for i in range(epoch):
        # evaluate losses
        loss = chamfer_dist(x, x_gt).mean()
        if i < 5:
            print(torch.autograd.grad(loss, x, grad_outputs=[torch.ones_like(loss)], create_graph=False))
        # update x
        [x], hidden = lgd.step(x, [chamfer_dist_list], hidden, 1024)
        x = detach_var(x)
        hidden = detach_var(hidden)

        if i%10 == 0:
            writer.add_scalars("regression_loss", {"LGD": loss}, global_step=i)
            writer.add_mesh("point cloud regression_LGD", x.unsqueeze(0), global_step=i)
        
    writer.close()

if __name__ == "__main__":
    main()
