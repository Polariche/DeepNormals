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


parser.add_argument('--pretrained-weight', dest='weight', metavar='PATH', default=None, 
                        help='pretrained weight')

parser.add_argument('--batchsize', dest='batchsize', metavar='BATCHSIZE', default=1,
                        help='batch size')
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

    with torch.no_grad():
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


    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.pretrained_weight))
        except:
            print("Couldn't load pretrained weight: " + args.pretrained_weight)


if __name__ == "__name__":
    main()