

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
from loaders import InstanceDataset, dict_collate_fn, dict_to_device
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
    diag_JtJ = diag_JtJ.view(*JtJ.shape)

    pinv = torch.matmul(torch.inverse(JtJ + lamb * diag_JtJ), Jt)

    delta = - pinv * y.unsqueeze(-1)
    delta = delta[..., 0, :]

    return x + delta


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
    parser.add_argument('--image-sidelength', dest='img_length', type=int,metavar='length', default=5, 
                            help='')

    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    net = DeepSDFNet(8).to(device)
    net_optimizer = optim.Adam([*net.parameters()], lr=args.lr)
    net.eval()

    with torch.no_grad():
        # load an InstanceDataset
        instance = InstanceDataset("/data/SRN/cars_train/88c884dd867d221984ae8a5736280c", img_sidelength=args.img_length)
        instance_loader = DataLoader(instance, batch_size=2, shuffle=True)
    

    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            # load views
            samples = next(iter(instance_loader))

            rgb = samples['rgb']            # rgb; (b, l*l, 3)
            pose = samples['pose']          # pose; (b, 4, 4)
            mask = samples['mask']          # image; (b, l*l, 1)

            # initialize cam pos
            dir = torch.cat([torch.from_numpy(np.mgrid[:args.img_length,args.img_length-1:-1:-1].T.reshape(-1,2)/args.img_length-0.5).float(), 
                            torch.ones((args.img_length**2,1))], dim=-1).to(device)
            dir = dir.unsqueeze(0).expand(*rgb.shape)
            
            d = torch.zeros_like(dir)[:,:1].requires_grad_(True)
            h = torch.zeros(1,1,256).to(device)

            campos = lambda d: torch.mm(pose[...,:3,:3], d*dir) + pose[...,0,:3]
            sdf = lambda d: net(torch.cat([h, campos(d)], dim=-1))


            # find zero-set by levenberg-marquardt
            for i in range(5):
                d = lm(d, sdf)
            
            
            # view consistency loss

            # mask loss; BCE(mask, sigmoid(-a*d))
            bce = nn.BCELoss()
            L2 = bce(mask, F.sigmoid(-1e-1*sdf(d)))

            if i%10 ==0:
                writer.add_mesh("2D recon",
                                torch.cat([campos(0).reshape(-1,3).unsqueeze(0), campos(d).reshape(-1,3).unsqueeze(0)]),
                                global_step=i+1,
                                colors=rgb.reshape(-1,3).unsqueeze(0).expand(2,-1,-1))

            torch.save(net.state_dict(), args.weight_save_path+'model_%03d.pth' % i)

            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()