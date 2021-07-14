import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

#from models.models import SingleBVPNet, DeepSDFDecoder, Siren
from loaders import CategoryDataset, dict_collate_fn, dict_to_device
from models.LGD import Projector
from torch.utils.data import  DataLoader

import argparse

from sklearn.neighbors import KDTree

import time
from tqdm.autonotebook import tqdm
import knn_cuda


def find_nearest_correspondences_dist(x_hat, x, k=1):
    assert x_hat.shape == x.shape
    shp = x.shape

    # ((batches), n, m*2)
    x_hat = x_hat.view(-1, shp[-3], shp[-2], shp[-1])
    x = x.view(-1, shp[-3], shp[-2], shp[-1])

    dists = []
    for x_hat_, x_ in zip(x_hat, x):
        x_hat_ = x_hat_.transpose(0,1).reshape(shp[-2], -1)
        x_ = x_.transpose(0,1).reshape(shp[-2], -1)

        dist_, ind_ = knn_cuda.forward(x_hat_,x_,k)
        dists.append(dist_.mean().unsqueeze(0) / shp[-3])

    dist = torch.cat(dists).sum()
        
    return dist

def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--tb-save-path', dest='tb_save_path', type=str, metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--weight-save-path', dest='weight_save_path', type=str, metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')


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

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    category = CategoryDataset(srn_dir="/data/SRN/cars_train", shapenet_dir="/data/shapenetv2/ShapeNetCore.v2", img_sidelength=512, batch_size=args.batchsize, ray_batch_size=2)
    category_loader = DataLoader(category, batch_size=1, shuffle=True)


    projector = Projector(3).to(device)
    projector_optimizer = optim.Adam(projector.parameters(), lr=args.lr)

    

    # train LGD
    projector.train()
    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            samples = next(iter(category_loader))
            samples = dict_to_device(samples, device)

            X = (torch.randn_like(samples['p']) * 1e-2).requires_grad_(True)
            x = samples['uv']
            P = samples['pose']
            
            writer.add_mesh("input_view", 
                            (samples['p']).reshape(-1,3).unsqueeze(0), 
                            global_step=i+1)

            
            X_optimizer = optim.Adam([X], lr=args.lr)
            for j in range(10):
                X_optimizer.zero_grad()

                P_ = P.view(*P.shape[:-1], 4, 3)                             # (m, 4, 3)
                X_ = X.unsqueeze(-3)                                         # (1, n, 3)
                X_ = torch.cat([X_, torch.ones_like(X_)[..., :1]], dim=-1)     # (1, n, 4)
                X_ = torch.matmul(X_, P_)                                      # (m, n, 3)
                x_hat = X_[..., :-1] / X_[..., -2:-1]                         # (m, n, 2)

                L = find_nearest_correspondences_dist(x_hat, x, k=4)

                L.backward(retain_graph=True)
                tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (i, L, time.time() - start_time))

                X_optimizer.step()
            
            X_new = X.clone()

            writer.add_mesh("output_view", 
                            (X_new).reshape(-1,3).unsqueeze(0), 
                            global_step=i+1)

            
            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()