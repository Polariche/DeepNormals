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
from loaders import ObjDataset, ObjUniformSample, Dataset, UniformSample, GridDataset, PointTransform
from models.LGD import LGD, detach_var

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse



def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--lgd-weight', dest='lgd_weight', metavar='PATH', default=None, 
                            help='pretrained weight for LGD model')
    parser.add_argument('--sdf-weight', dest='sdf_weight', metavar='PATH', default=None, 
                            help='pretrained weight for SDF model')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=200, 
                            help='epochs for adam and lgd')

    parser.add_argument('--width', dest='width', type=int,metavar='WIDTH', default=128, 
                            help='width for rendered image')
    parser.add_argument('--height', dest='height', type=int,metavar='HEIGHT', default=128, 
                            help='height for rendered image')

    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    width = args.width
    height = args.height
    epoch = args.epoch

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True).to(device) 

    if args.sdf_weight != None:
        try:
            model.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    model.eval() 
    for param in model.parameters():
        param.requires_grad = False

    
    # load 
    mm = torch.tensor([-0.1, -0.1, 0.1], device=device, dtype=torch.float)
    mx = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=torch.float)
    wh = torch.tensor([width, height, 1], device=device, dtype=torch.int)

    rot = torch.tensor([[1,0,0], [0,1,0], [0,0,1]], device=device, dtype=torch.float)
    trans = torch.tensor([[0, 0, -0.8]], device=device, dtype=torch.float)

    p_distribution = GridDataset(mm, mx, wh)

    d = torch.zeros((width*height, 1), device=device, dtype=torch.float).requires_grad_(True)
    

    sampler = nn.Sequential(UniformSample(width*height), 
                            PointTransform(rot))

    p = sampler(p_distribution)


    ds = ObjDataset(args.data)
    objsampler = ObjUniformSample(1000)
    x_preview = (objsampler(ds)['p']).to(device)


    d2_eval = lambda d: torch.pow(d, 2).mean()
    sdf_eval = lambda d: torch.pow(model(d * ray_n + p + trans)[0], 2).sum(dim=1).mean()
    d_eval = lambda d: (torch.tanh(d) - 1.).mean() * 0.5

    d2_eval_list = lambda d: d2_eval(d[0])
    sdf_eval_list = lambda d: sdf_eval(d[0])
    d_eval_list = lambda d: d_eval(d[0])


    writer.add_mesh("preview", torch.cat([(p + trans),  x_preview]).unsqueeze(0), global_step=0)

    print("adam")
    optimizer = optim.Adam([d], lr = lr)

    for i in range(epoch):
        optimizer.zero_grad()

        loss = sdf_eval(d)
        loss.backward(retain_graph=True)

        optimizer.step()

        if i%10 == 0:
            writer.add_scalars("regression_loss", {"Adam": loss}, global_step=i)
            writer.add_mesh("raymarch_Adam", torch.cat([(d * ray_n + trans + p),  x_preview]).unsqueeze(0), global_step=i)

    writer.close()

if __name__ == "__main__":
    main()
