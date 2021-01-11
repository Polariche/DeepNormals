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
parser.add_argument('--omega', dest='omega', type=float, metavar='OMEGA', default=10, 
                        help='hyperparameter for periodic layer')

parser.add_argument('--pe', dest='pe', metavar='PE', type=bool, default=True, 
                        help='positional encoding')
parser.add_argument('--pedim', dest='pedim', metavar='PE_DIMENSIONS', type=int, default=60, 
                        help='positional encoding dimension')

parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')


def train_batch(device, model, x, y, batchsize, backward=True):
    
    loss_sum = 0
    y_cat = torch.zeros_like(x).to(device)[:,:1]

    bs = batchsize

    for j in range(x.shape[0] // bs):
        x_ = x[j*bs : (j+1)*bs].to(device)
        x_.requires_grad= True

        y_ = model(x_)
        y_cat[j*bs : (j+1)*bs] = y_

        # fit gradient
        y_ = utils.normal_from_y(y_, x_)
        loss = torch.sum(torch.mean(torch.norm(y_ - y, dim=1, keepdim=True), dim=(1,2,3)))

        if backward:
            loss.backward()
        loss_sum += loss.detach() 

    x.requires_grad=False

    return loss_sum, y_cat

def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # read input depth
    depth = cv2.imread(args.data, -1).astype(np.float32) / 1000.
    depth = cv2.resize(depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    depth = torch.tensor(depth.T, device=device).unsqueeze(0).unsqueeze(0)

    w,h = depth.shape[2:]
    x,y = torch.meshgrid(torch.true_divide(torch.arange(w), w) - 0.5, 
                         torch.true_divide(torch.arange(h), h) - 0.5)

    xyz = torch.cat([x.to(device).unsqueeze(0).unsqueeze(0),
                    y.to(device).unsqueeze(0).unsqueeze(0), 
                    depth], dim=1)

    normal = Sobel(3).to(device).normal(xyz).detach()

    writer.add_image("normal_GT", normal, 0, dataformats="NCWH")
    
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

    bs = args.batchsize
    d = torch.arange(-1, 1, 0.2).view(-1,1,1,1) * args.epsilon
    d = d + torch.rand(d.shape[0], 1, *xyz.shape[2:]) * 0.2 * args.epsilon
    d = d.to(device)

    d_valid = (torch.rand(10, 1, *xyz.shape[2:]) * 2 - 1.) * args.epsilon
    d_valid = d_valid.to(device)
    for epoch in range(args.epoch):
        loss_t = 0
        loss_d = 0

        optimizer.zero_grad()
        
        # validation
        utils.model_test(model)
        with torch.no_grad():
            xyz_valid = (xyz + d_valid * normal).detach()

            loss_d, y = train_batch(device, model, xyz_valid, normal, bs, backward=False)
            loss_d /= d_valid.shape[0]
            print(y.shape)
            writer.add_image("validation", y[0:1].repeat(1,3,1,1), epoch, dataformats="NCWH")


        # normal test
        xyz.requires_grad = True
        writer.add_image("normals", utils.normal_from_y(model(xyz), xyz), epoch, dataformats="NCWH")
        xyz.requires_grad = False


        # train
        utils.model_train(model)
        loss_t, _ = train_batch(device, model, xyz, normal, bs, backward=True)
        loss_t /= d.shape[0]

        writer.add_scalars("loss", {'train': loss_t, 'validation': loss_d}, epoch)
        
        # update
        optimizer.step()

        torch.save(model.state_dict(), args.weight_save_path+'model_%03d.pth' % epoch)
        
    
    writer.close()


if __name__ == "__main__":
    main()
