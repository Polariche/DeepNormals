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
from torch.utils.data import  DataLoader

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


def train(device, model, xyz, s_gt, n_gt,backward=True, lamb=0.005):
    s, n = model(xyz)

    for param in model.parameters():
        param.requires_grad = False
    
    nd = torch.norm(n, dim=1, keepdim=True)

    ones = s_gt == 0
    zeros = s_gt == 1

    loss_grad = torch.sum(torch.pow(nd - 1,2))
    loss_zeros = torch.sum((torch.pow(s,2) + (1 - torch.sum(n * n_gt, dim=1, keepdim=True)))[zeros])
    loss_ones = torch.sum(torch.exp(-nd)[ones])

    loss = loss_grad + loss_zeros + loss_ones 
    loss /= xyz.shape[0]
    
    if backward:
        for param in model.parameters():
            param.requires_grad = True
    loss.backward()

    return loss.detach(), s.detach(), n.detach()

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True).to(device) 

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.pretrained_weight))
        except:
            print("Couldn't load pretrained weight: " + args.pretrained_weight)


    # load 
    ds = ObjDataset("../../../data/train/02828884/model_001415.obj")

    n = ds.vn
    xyz = ds.v

    with torch.no_grad():
        xyz_aug = torch.cat([xyz, xyz + n * torch.rand(xyz.shape)], dim=0)
        s_aug = torch.cat([torch.zeros((xyz.shape[0], 1)), torch.ones((xyz.shape[0], 1))], dim=0)
        n_aug = n.repeat(2,1)

        n_aug = n_aug.to(device)
        xyz_aug = xyz_aug.to(device)
        s_aug = s_aug.to(device)

    writer.add_mesh("n_gt", xyz.unsqueeze(0), colors=(n.unsqueeze(0) * 128 + 128).int())

    for epoch in range(args.epoch):
        loss_t = 0

        optimizer.zero_grad()
        
        # train
        utils.model_train(model)
        loss_t, s, n = train(device, model, xyz_aug, s_aug, n_aug,backward=True, lamb= args.lamb)

        writer.add_scalars("loss", {'train': loss_t}, epoch)

        if epoch % 10 == 0:
            writer.add_mesh("s", xyz_aug.unsqueeze(0), colors=(s.unsqueeze(0).repeat(1,1,3) * 128 + 128).int(), global_step=epoch)
            writer.add_mesh("n", xyz_aug.unsqueeze(0), colors=(n.unsqueeze(0) * 128 + 128).int(), global_step=epoch)

        # update
        optimizer.step()

        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()
