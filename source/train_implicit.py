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
    s, xyz = model(xyz)
    
    n = torch.autograd.grad(s, [xyz], grad_outputs=torch.ones_like(s), create_graph=True)[0]
    nd = torch.norm(n, dim=1, keepdim=True)

    # modified loss in SIREN 4.2 for smooth transition between surface points and non-surface points
    # use probability : exp(- s_gt / (2*eps^2))

    p = lambda x : torch.exp(-x / (2*1e-4))
    p_gt = p(s_gt)

    # instead of using n_gt, we could use ECPN tangent loss
    loss_grad = torch.sum(1e2 * (1 - torch.sum(n * n_gt, dim=1, keepdim=True) / nd) * p_gt)
    loss_zeros = torch.sum(3e3 * torch.abs(s) * p_gt)
    loss_ones = torch.sum(1e2 * torch.exp(-1e2*nd) * (1-p_gt))

    loss = loss_grad + loss_zeros + loss_ones
    loss /= xyz.shape[0]
    
    if backward:
        if xyz.grad != None:
            xyz.grad.zero_()

        for param in model.parameters():
            if param.grad != None:
                param.grad.zero_()

        loss.backward()

    return loss.detach(), s.detach(), n.detach()

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True).to(device) 

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.pretrained_weight))
        except:
            print("Couldn't load pretrained weight: " + args.pretrained_weight)


    # load 
    ds = ObjDataset("../../../data/train/02828884/model_005004.obj")

    n = ds.vn
    xyz = ds.v

    with torch.no_grad():
        s_aug = torch.cat([torch.zeros((xyz.shape[0], 1)), torch.rand((xyz.shape[0], 1))], dim=0)
        xyz_aug = torch.cat([xyz, xyz + n * s_aug[xyz.shape[0]:] * 0.05], dim=0)
        n_aug = n.repeat(2,1)

        s_aug = s_aug.to(device)
        n_aug = n_aug.to(device)
        xyz_aug = xyz_aug.to(device)
        
        n_gt = n.to(device)
        xyz_gt = xyz.to(device)

    writer.add_mesh("1. n_gt", xyz.unsqueeze(0), colors=(n.unsqueeze(0) * 128 + 128).int())
    optimizer = optim.Adam(list(model.parameters()) + [xyz_aug], lr = 1e-4)

    for epoch in range(args.epoch):
        loss_t = 0

        optimizer.zero_grad()
        
        # train
        utils.model_train(model)
        loss_t, s, n = train(device, model, xyz_aug, s_aug, n_aug, backward=True, lamb= args.lamb)
        loss_x = 5e2 * torch.sum(torch.norm(xyz_aug - xyz_gt.repeat(2,1), dim=1))

        loss_x.backward()

        writer.add_scalars("loss", {'train': loss_t + loss_x.detach()}, epoch)

        # visualization
        n_normalized = n / torch.norm(n, dim=1, keepdim=True)
        n_error = (1 - torch.sum(n_normalized * n_aug, dim=1, keepdim=True) / torch.norm(n_aug, dim=1, keepdim=True))
        n_error[torch.isnan(n_error)] = -1

        n_error_originals = n_error[:xyz.shape[0]]

        writer.add_scalars("normal error", {'train': n_error_originals[n_error_originals > -1].detach().mean()}, epoch)

        if epoch % 10 == 0:
            writer.add_mesh("2. n", xyz_aug[xyz.shape[0]:].unsqueeze(0), colors=(n_normalized[:xyz.shape[0]:].unsqueeze(0) * 128 + 128).int(), global_step=epoch)
        
        if epoch % 10 == 0:
            writer.add_mesh("3. n_error", xyz_aug[xyz.shape[0]:].unsqueeze(0), colors=(F.pad(n_error[:xyz.shape[0]:], (0,2)).unsqueeze(0) * 256).int(), global_step=epoch)

        # update
        optimizer.step()

        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()
