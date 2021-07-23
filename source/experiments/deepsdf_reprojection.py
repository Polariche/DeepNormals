

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
from models.models import DeepSDFNet
from torch.utils.data import  DataLoader

import argparse


import time
from tqdm.autonotebook import tqdm
from knn import knn
import utils

def gauss_newton(x, f, dx):
    y = f(x)
    dx = torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    dx_pinv = torch.pinverse(dx.unsqueeze(-2) + 1e-9)[..., 0]


    return x - y*dx_pinv


def lm(x, f, lamb = 1.1):
    y = f(x)
    dx = torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

    J = dx.unsqueeze(-1)
    Jt = J.transpose(-2, -1)
    JtJ = torch.matmul(Jt, J)


    k = JtJ.shape[-1]

    diag_JtJ = torch.cat([JtJ[..., i, i] for i in range(k)])
    diag_JtJ = diag_JtJ.view(-1, k, 1)
    diag_JtJ = torch.eye(k, device=x.device).unsqueeze(0).expand(diag_JtJ.shape[0], -1, -1) * diag_JtJ

    pinv = torch.matmul(torch.inverse(JtJ + lamb * diag_JtJ), Jt)

    delta = - pinv * y.unsqueeze(-1)
    delta = delta[..., 0, :]

    return x + delta


def find_nearest_correspondences_pos(x_hat, x, k=1):
    assert x_hat.shape[-1] == x.shape[-1]
    assert x_hat.shape[-3] == x.shape[-3]
    shp1 = x_hat.shape
    shp2 = x.shape

    # ((batches), m, n, 2)
    x_hat = x_hat.view(-1, shp1[-3], shp1[-2], shp1[-1])
    x = x.view(-1, shp2[-3], shp2[-2], shp2[-1])

    #dists = []
    poss = []
    for x_hat_, x_ in zip(x_hat, x):
        # ((batches), n, m*2)
        x_hat_ = x_hat_.transpose(0,1).reshape(shp1[-2], -1)    
        x_ = x_.transpose(0,1).reshape(shp2[-2], -1)

        knn_f = knn.apply
        _, ind  = knn_f(x_hat_,x_,k, True)
        ind = ind.long()
        poss.append(x_[ind].unsqueeze(0))
        
    pos = torch.cat(poss).view(*shp1[:-1], k, shp1[-1])
    return pos


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


    category = CategoryDataset(srn_dir="/data/SRN/cars_train", 
                                shapenet_dir="/data/shapenetv2/ShapeNetCore.v2", 
                                img_sidelength=512, 
                                batch_size=args.batchsize, 
                                ray_batch_size=3)

    category_loader = DataLoader(category, batch_size=1, shuffle=True)


    net = DeepSDFNet(8).to(device)
    net_optimizer = optim.Adam(net.parameters(), lr=args.lr)

    samples = next(iter(category_loader))
    samples = dict_to_device(samples, device)

    net.eval()
    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            X = (torch.randn_like(samples['p']) * 2e-1).requires_grad_(True)
            H = torch.zeros(*samples['p'].shape[:-2], 1, 256, device=device).requires_grad_(True)
            G = torch.zeros(*samples['p'].shape[:-1], 256, device=device).requires_grad_(True)
            P = samples['pose']

            Y = samples['p']

            X_original = X.clone().detach()
            Y_corr = find_nearest_correspondences_pos(X_original, Y)

            net_optimizer.zero_grad()
            
            H = H.expand(*[-1]*(len(H.shape)-2), X.shape[-2], -1)
            _input = torch.cat([X, H], dim=-1)

            for j in range(5):
                _input = lm(_input, net)

            X = _input[..., :X.shape[-1]]

            L = ((X - Y_corr)**2).sum(dim=-1).mean() + (net(Y)**2).sum(dim=-1).mean()
            print(L)
            L.backward(retain_graph=True)    

            net_optimizer.step()


            writer.add_mesh("input_view",
                            (samples['p']).reshape(-1,3).unsqueeze(0),
                            global_step=i+1,
                            colors=(torch.clamp((F.normalize(samples['n'], dim=-1).reshape(-1,3).unsqueeze(0)), -1, 1) * 128 + 128).int())

            writer.add_mesh("output_view",
                            (X).reshape(-1,3).unsqueeze(0),
                            global_step=i+1)
                            #colors=(F.normalize(X_new_grad, dim=-1).reshape(-1,3).unsqueeze(0) * 128 + 128).int())


            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()