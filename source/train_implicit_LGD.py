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
from torch.utils.data import  DataLoader, WeightedRandomSampler

from LGD import LGD

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


parser.add_argument('--abs', dest='abs', type=bool, metavar='BOOL', default=True, 
                        help='whether we should use ABS when evaluating normal loss')

parser.add_argument('--epsilon', dest='epsilon', type=float, metavar='EPSILON', default=0.1, 
                        help='epsilon')
parser.add_argument('--omega', dest='omega', type=float, metavar='OMEGA', default=30, 
                        help='hyperparameter for periodic layer')


parser.add_argument('--lambda', dest='lamb', type=float, metavar='LAMBDA', default=0.005, 
                        help='hyperparameter for s : normal loss ratio')


parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')


def train(device, model, xyz, s_gt, n_gt, backward=True, lamb=0.005, use_abs=True):
    s, xyz = model(xyz)

    n = torch.autograd.grad(s, [xyz], grad_outputs=torch.ones_like(s), create_graph=True)[0]

    # modified loss in SIREN 4.2 for smooth transition between surface points and non-surface points
    # use probability : exp(- s_gt / (2*eps^2))

    with torch.no_grad():
        p_gt = torch.exp(-s_gt / (2*1e-4))

    # instead of using n_gt, we could use ECPN tangent loss
    if use_abs:
        loss_grad = 1e2 * torch.mean((1 - torch.abs(torch.sum(n * n_gt, dim=1, keepdim=True)) / torch.norm(n, dim=1, keepdim=True)) * p_gt)
    else:
        loss_grad = 1e2 * torch.mean((1 - torch.sum(n * n_gt, dim=1, keepdim=True) / torch.norm(n, dim=1, keepdim=True)) * p_gt)
    loss_s = 3e2 * torch.mean(torch.pow(s - s_gt,2))

    #loss_zeros = 3e3 * torch.mean(torch.abs(s) * p_gt)
    #loss_ones = 1e2 * torch.mean(torch.exp(-1e2*s) * (1-p_gt))

    loss = loss_grad + loss_s #+ loss_zeros + loss_ones

    if backward:
        if xyz.grad != None:
            xyz.grad.zero_()

        for param in model.parameters():
            param.requires_grad_(True)
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
            model.load_state_dict(torch.load(args.weight))
        except:
            print("Couldn't load pretrained weight: " + args.weight)


    # load 
    ds = ObjDataset(args.data)
    fnn = torch.abs(ds.fnn)
    samples = list(WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), 50000, replacement=True))

    data = [ds[samples[i]] for i in range(len(samples))]

    xyz = torch.cat([d['xyz'].unsqueeze(0) for d in data])
    n = torch.cat([d['n'].unsqueeze(0) for d in data])

    with torch.no_grad():
        s_aug = torch.cat([torch.zeros((xyz.shape[0], 1)), torch.rand((xyz.shape[0], 1))], dim=0)
        xyz_aug = torch.cat([xyz, xyz + n * s_aug[xyz.shape[0]:] * args.epsilon], dim=0)
        n_aug = n.repeat(2,1)

        s_aug = s_aug.to(device)
        n_aug = n_aug.to(device)
        xyz_aug = xyz_aug.to(device)
        
        xyz_gt = xyz.to(device).repeat(2,1)

    writer.add_mesh("1. n_gt", xyz.unsqueeze(0), colors=(n.unsqueeze(0) * 128 + 128).int())

    lgd = LGD(list(model.parameters()), layers_generator=Siren).to(device)
    optimizer = optim.Adam(list(lgd.parameters()), lr = 1e-3)

    for epoch in range(args.epoch):
        lgd.zero_grad()
        
        # train
        utils.model_train(model)
        loss_t, s, n = train(device, model, xyz_aug, s_aug, n_aug, backward=True, lamb= args.lamb, use_abs=args.abs)

        loss_x = 1e2 * torch.sum(torch.pow(xyz_aug - xyz_gt, 2))
        loss_x.backward()

        writer.add_scalars("loss", {'train': loss_t + loss_x.detach()}, epoch)

        # visualization
        with torch.no_grad():
            
            n_normalized = n / torch.norm(n, dim=1, keepdim=True)
            
            if args.abs:
                n_error = torch.sum(torch.abs(n_normalized * n_aug), dim=1, keepdim=True) / torch.norm(n_aug, dim=1, keepdim=True)
            else: 
                n_error = torch.sum(n_normalized * n_aug, dim=1, keepdim=True) / torch.norm(n_aug, dim=1, keepdim=True)
                
            n_error = torch.acos(n_error) / np.arccos(0)

            n_error_originals = n_error[:xyz.shape[0]]

            writer.add_scalars("normal error", {'train': n_error_originals[~torch.isnan(n_error_originals)].detach().mean()}, epoch)
            
            if epoch % 10 == 0:
                print(epoch)
                writer.add_mesh("2. n", xyz_aug[:xyz.shape[0]].unsqueeze(0).detach().clone(), 
                                colors=(n_normalized[:xyz.shape[0]].unsqueeze(0).detach().clone() * 128 + 128).int(), 
                                global_step=epoch)
                
                writer.add_mesh("3. n_error", xyz_aug[:xyz.shape[0]].unsqueeze(0).detach().clone(), 
                                colors=(F.pad(n_error[:xyz.shape[0]], (0,2)).unsqueeze(0).detach().clone() * 256).int(), 
                                global_step=epoch)
                

        # update the model
        lgd.apply_step(model)

        # update lgd
        if epoch % 10 == 0:
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            s_aug = (torch.norm(xyz_aug.detach().clone().cpu() - xyz.repeat(2,1), dim=1, keepdim=True)/args.epsilon).to(device)


        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()