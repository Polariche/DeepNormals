import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import DeepSDF, PositionalEncoding
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

parser.add_argument('--pe', dest='pe', metavar='PE', type=bool, default=True, 
                        help='positional encoding')
parser.add_argument('--pedim', dest='pedim', metavar='PE_DIMENSIONS', type=int, default=60, 
                        help='positional encoding dimension')

parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')


def train_batch(device, model, x, z_uv, h,w, batchsize, backward=True):
    loss_sum = 0
    bs = batchsize

    z_uv2 = torch.zeros(x.shape[0], 2)

    v,u = torch.meshgrid(torch.true_divide(torch.arange(h), h) - 0.5, 
                         torch.true_divide(torch.arange(w), w) - 0.5)
    v = v.reshape(-1,1).to(device)
    u = u.reshape(-1,1).to(device)

    for j in range(x.shape[0] // bs):
        br = torch.arange(j*bs, (j+1)*bs, dtype=torch.long)

        x_ = x[br].to(device)
        x_.requires_grad= True

        z_uv_ = z_uv[br].to(device)

        f_ = model(x_)
        n_ = utils.model_from_y(f_, x_)
        
        zx_ = (-n_[:,0] / n_[:,2]).view(-1,1)
        zy_ = (-n_[:,1] / n_[:,2]).view(-1,1)

        z_uv2_ = torch.cat([(zx_*x_[:,2].view(-1,1)/w) / (1 - zx_ * u - zy_ * v),
                            (zy_*x_[:,2].view(-1,1)/h) / (1 - zx_ * u - zy_ * v)],dim=1)
        z_uv2[br] = z_uv2_

        loss = torch.sum(torch.linalg.norm(z_uv_ - z_uv2_, dim=1)) / x.shape[0]

        if backward:
            loss.backward()
        loss_sum += loss.detach() 

    return loss_sum, z_uv2

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    if args.pe:
        model = nn.Sequential(PositionalEncoding(args.pedim),
                                DeepSDF(args.pedim, 1, activation=args.activation, omega_0 = args.omega)).to(device)
    else:
        model = DeepSDF(3, 1, activation=args.activation, omega_0 = args.omega).to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.pretrained_weight))
        except:
            print("Couldn't load pretrained weight: " + args.pretrained_weight)


    # read input depth
    depth = cv2.imread(args.data, -1).astype(np.float32) / 1000.
    depth = cv2.resize(depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    depth = torch.tensor(depth.T, device=device).unsqueeze(0).unsqueeze(0)

    w,h = depth.shape[2:]
    x,y = torch.meshgrid(torch.true_divide(torch.arange(w), w) - 0.5, 
                         torch.true_divide(torch.arange(h), h) - 0.5)

    xyz = torch.cat([x.to(device).unsqueeze(0).unsqueeze(0),
                    y.to(device).unsqueeze(0).unsqueeze(0), 
                    torch.ones((1,1,w,h)).to(device)], dim=1)
    xyz *= depth

    z_uv = Sobel(1).to(device)(depth)

    xyz = xyz.squeeze().view(3,-1).T.detach()
    z_uv = z_uv.squeeze().view(2,-1).T.detach()

    writer.add_image("z_uv", torch.cat([z_uv.view(h,w,2), torch.zeros((h,w,1)).to(device)], dim=2), 0, dataformats="HWC")

    bs = args.batchsize

    for epoch in range(args.epoch):
        loss_t = 0

        optimizer.zero_grad()
        
        # train
        utils.model_train(model)
        loss_t, z_uv2 = train_batch(device, model, xyz, z_uv, h,w, bs)

        writer.add_image("z_uv2", torch.cat([z_uv2.view(h,w,2), torch.zeros((h,w,1)).to(device)], dim=2), epoch, dataformats="HWC")
        writer.add_scalars("loss", {'train': loss_t}, epoch)
        
        # update
        optimizer.step()

        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()
