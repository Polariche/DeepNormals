import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import PointSetGenerator
from loaders import PSGDataset
from models.LGD import LGD, detach_var
from evaluate_functions import chamfer_distance, nearest_from_to, dist_from_to

from knn import knn

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse

from sklearn.neighbors import KDTree

class VanilaOptimizer(optim.Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(VanilaOptimizer, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.add_(p.grad)

        return loss


def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')
    parser.add_argument('--weight-save-path', dest='weight_save_path', metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')

    parser.add_argument('--lgd-weight', dest='lgd_weight', metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=1e-3, 
                            help='learning rate')


    args = parser.parse_args()

    lr = args.lr
    epoch = args.epoch
    batchsize = args.batchsize

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ds = PSGDataset(args.data)
    dl = DataLoader(ds, batch_size=batchsize, shuffle=True)


    knn_f = knn.apply
    chamfer_dist = lambda x, y: knn_f(x, y, 1).mean() + knn_f(y, x, 1).mean()

    psg = PointSetGenerator().to(device)

    
    lgd = LGD(3, 1, k=10).to(device)

    if args.lgd_weight != None:
        try:
            lgd.load_state_dict(torch.load(args.lgd_weight))
        except:
            print("Couldn't load pretrained weight: " + args.lgd_weight)

    optimizer = optim.Adam(psg.parameters(), lr=lr) #VanilaOptimizer(psg.parameters())
    psg.train()
    for i in range(epoch):
        # select batches
        sample_batched = next(iter(dl))

        x = sample_batched['img'].reshape(-1,4,192,256).to(device)
        y_gt = sample_batched['pc_gt'].reshape(-1,16384,3).to(device)
        y = psg(x)
        

        optimizer.zero_grad()

        loss = torch.cat([chamfer_dist(y[i], y_gt[i]).unsqueeze(0) for i in range(x.shape[0])]).mean()
        lgd.gradient(y, loss)
        y.backward(- y.grad)

        optimizer.step()

        loss_eval = loss([y])
        print(i, loss_eval.item())

        writer.add_scalars("train_loss", {"Adam": loss_eval}, global_step=i)
        torch.save(psg.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
        
    writer.close()

if __name__ == "__main__":
    main()
