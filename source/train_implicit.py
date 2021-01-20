import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import Siren
from utils import Sobel
import utils

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


def train_batch(device, model, xyz, s_gt, n_gt, batchsize, backward=True, lamb=0.005):
    loss_sum = 0
    bs = batchsize

    for j in range(xyz.shape[0] // bs):
        br = torch.arange(j*bs, (j+1)*bs, dtype=torch.long)

        xyz_ = xyz[br]
        s_, n_ = model(xyz_)

        for param in model.parameters():
            param.requires_grad = False
        
        sloss = torch.sum(torch.pow(s_ - s_gt[br],2))
        nloss = torch.sum(n_ * n_gt[br])

        loss = (1-lamb) * sloss + lamb * nloss
        loss /= xyz.shape[0]

        if backward:
            for param in model.parameters():
                param.requires_grad = True
            loss.backward()
        loss_sum += loss.detach() 

    return loss_sum

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device) 

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.pretrained_weight))
        except:
            print("Couldn't load pretrained weight: " + args.pretrained_weight)


    # load 

    bs = args.batchsize
    for epoch in range(args.epoch):
        loss_t = 0

        optimizer.zero_grad()
        
        # train
        utils.model_train(model)
        loss_t, f, g = train_batch(device, model, xyz[:,:3], xyz[:,3:], n, bs, backward=True, lamb= args.lamb)

        writer.add_scalars("loss", {'train': loss_t}, epoch)
        
        # update
        optimizer.step()

        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()
