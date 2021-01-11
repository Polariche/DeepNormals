import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import DeepSDF, PositionalEncoding
from utils import Sobel

import argparse

parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DATA', help='path to file')

parser.add_argument('--save-path', dest='save_path', metavar='PATH', default='../checkpoints/', 
                        help='tensorboard checkpoints path')

parser.add_argument('--batchsize', dest='batchsize', metavar='BATCHSIZE', default=1,
                        help='batch size')
parser.add_argument('--epoch', dest='epoch', metavar='EPOCH', default=10, 
                        help='epochs')

parser.add_argument('--epsilon', dest='epsilon', metavar='EPSILON', default=0.1, 
                        help='epsilon')

parser.add_argument('--pe', dest='pe', metavar='PE', default=True, 
                        help='positional encoding')
parser.add_argument('--pedim', dest='pedim', metavar='PE_DIMENSIONS', default=10, 
                        help='positional encoding dimension')
parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')



def main():
    args = parser.parse_args()

    writer = SummaryWriter("../checkpoints/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # read input depth
    depth = cv2.imread(args.data, -1).astype(np.float32) / 1000.
    depth = torch.tensor(depth.T, device=device).unsqueeze(0).unsqueeze(0)

    w,h = depth.shape[2:]

    xyz = torch.cat([torch.meshgrid(torch.arange(w) / w - 0.5, 
                                    torch.arange(h) / h - 0.5, device=device), 
                    depth], dim=1)
    normal = Sobel(3).normal(xyz)

    
    # create models
    if args.pe:
        model = nn.Sequential(PositionalEncoding(args.pedim),
                                DeepSDF(args.pedim, 1), device=device)
    else:
        model = DeepSDF(3, 1, device=device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)


    for epoch in range(args.epoch):
        loss_ = 0

        optimizer.zero_grad()
        bs = args.batchsize
        d = torch.arange(-1, 1, 0.2).view(-1,1,1,1) * args.epsilon

        for j in range(d.shape[0] // bs):
            d_bs = d[j*bs:(j+1)*bs]
            data = (xyz + d_bs * normal).detach()
            y = model(data)

            writer.add_images('images', y.repeat(1,3,1,1), epoch)

            loss = torch.sum(torch.mean(torch.norm(y - d_bs/args.epsilon, dim=1, keepdim=True), dim=(1,2,3)))
            loss.backward()

            loss_ += loss.detach()
        
        optimizer.step()
        writer.add_scalar("loss", loss_, epoch)
    
    writer.close()


if __name__ == "__name__":
    main()