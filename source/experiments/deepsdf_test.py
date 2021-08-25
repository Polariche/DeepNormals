

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

    parser.add_argument('--sdf-weight', dest='sdf_weight', type=str, metavar='PATH', default='../weights/', 
                            help='sdf weight')


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


    net = DeepSDFNet(8).to(device)
    if args.sdf_weight != None:
        try:
            net.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    net.eval()

    dir = torch.tensor([[0,0,1]]).to(device)
    d = torch.zeros((1000,1)).to(device).requires_grad_(True)
    pos = torch.tensor([[0,0, -1e-2]]).to(device)

    sdf = lambda d: net(d*dir + pos)

    for i in range(10):
        d = lm(d, sdf)

    writer.add_mesh("input_view",
                    ((d*dir + pos)).reshape(-1,3).unsqueeze(0),
                    global_step=0)

    writer.close()

if __name__ == "__main__":
    main()