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
from loaders import ObjDataset, ObjUniformSample, Dataset, UniformSample, NormalPerturb, RandomAugment

import utils 

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse


def train(device, model, p, s_gt, n_gt, backward=True, lamb=0.005, use_abs=True):
    s, p = model(p)

    n = torch.autograd.grad(s, [p], grad_outputs=torch.ones_like(s), create_graph=True)[0]

    with torch.no_grad():
        p_gt = torch.exp(-s_gt / (2*1e-4))

    loss_grad = (1 - torch.sum(n * n_gt, dim=1, keepdim=True) / torch.norm(n, dim=1, keepdim=True)) * p_gt
    loss_grad[torch.norm(n, dim=1) == 0] = 0        # don't compute normal loss for normal-less points (e.g. outer points)
    loss_s = torch.pow(s - s_gt,2)

    loss = 1e2 * loss_grad.mean() + 3e2 * loss_s.mean()

    if backward:
        if p.grad != None:
            p.grad.zero_()

        for param in model.parameters():
            param.requires_grad_(True)
            if param.grad != None:
                param.grad.zero_()

        loss.backward()

    return loss.detach(), s.detach(), n.detach()

def main():
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


    parser.add_argument('--abs', dest='abs', type=bool, metavar='BOOL', default=False, 
                            help='whether we should use ABS when evaluating normal loss')

    parser.add_argument('--epsilon', dest='epsilon', type=float, metavar='EPSILON', default=0.1, 
                            help='epsilon')
    parser.add_argument('--lambda', dest='lamb', type=float, metavar='LAMBDA', default=0.005, 
                            help='hyperparameter for s : normal loss ratio')


    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

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
    samples_n = 20000

    augments = nn.Sequential(ObjUniformSample(samples_n),
                             NormalPerturb(args.epsilon))

    ds = augments(ds)

    print(ds)

    ds = RandomAugment(samples_n // 2, args.epsilon * 0.5)(ds)
    print(ds)

    p_aug = ds['p'].detach_().to(device)
    n_aug = ds['n'].detach_().to(device)
    s_aug = ds['s'].detach_().to(device)

    p = p_aug[:samples_n]
    n = n_aug[:samples_n]

    p_gt = p.repeat(2,1)


    writer.add_mesh("1. n_gt", p.unsqueeze(0), colors=(n.unsqueeze(0) * 128 + 128).int())

    optimizer = optim.Adam(list(model.parameters()), lr = 1e-4)

    for epoch in range(args.epoch):
        optimizer.zero_grad()
        
        # train
        utils.model_train(model)
        loss_t, s, n = train(device, model, p_aug, s_aug, n_aug, backward=True, lamb= args.lamb, use_abs=args.abs)

        loss_x = 1e2 * torch.sum(torch.pow(p_aug - p_gt, 2))
        loss_x.backward()

        writer.add_scalars("loss", {'train': loss_t + loss_x.detach()}, epoch)

        # visualization
        with torch.no_grad():
            
            n_normalized = n / torch.norm(n, dim=1, keepdim=True)
            
            n_error = torch.sum(n_normalized * n_aug, dim=1, keepdim=True) / torch.norm(n_aug, dim=1, keepdim=True)

            n_error_originals = n_error[:p.shape[0]]

            writer.add_scalars("cosine similarity", {'train': n_error_originals[~torch.isnan(n_error_originals)].detach().mean()}, epoch)
            
            if epoch % 10 == 0:
                print(epoch)
                writer.add_mesh("2. n", p_aug[:p.shape[0]].unsqueeze(0).detach().clone(), 
                                colors=(n_normalized[:p.shape[0]].unsqueeze(0).detach().clone() * 128 + 128).int(), 
                                global_step=epoch)
                
                writer.add_mesh("3. cosine similarity", p_aug[:p.shape[0]].unsqueeze(0).detach().clone(), 
                                colors=(F.pad(1 - n_error[:p.shape[0]], (0,2)).unsqueeze(0).detach().clone() * 256).int(), 
                                global_step=epoch)
                

        # update
        optimizer.step()

        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()
