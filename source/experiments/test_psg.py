import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import PointSetGenerator
from loaders import PSGDataset
from models.LGD import LGD, detach_var
from evaluate_functions import chamfer_distance, nearest_from_to, dist_from_to

from knn import knn

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse

from sklearn.neighbors import KDTree



def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')
    parser.add_argument('--psg-weight', dest='psg_weight', metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')


    args = parser.parse_args()
    batchsize = args.batchsize

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ds = PSGDataset(args.data)
    dl = DataLoader(ds, batch_size=batchsize, shuffle=False)
    dl_iter = iter(dl)

    knn_f = knn.apply
    chamfer_dist = lambda x, y: knn_f(x, y, 1).mean() + knn_f(y, x, 1).mean()

    psg = PointSetGenerator().to(device)
    if args.args_weight != None:
        try:
            psg.load_state_dict(torch.load(args.args_weight))
        except:
            print("Couldn't load pretrained weight: " + args.args_weight)

    loss = 0
    psg.test()
    for sample_batched in dl_iter:
        x = sample_batched['img'].to(device)
        y = psg(x)
        y_gt = sample_batched['pc_gt'].to(device)

        loss += sum([chamfer_dist(y[i], y_gt[i]) for i in range(y.shape[0])])

    print("chamfer dist mean: ", loss / len(ds))


        
    writer.close()

if __name__ == "__main__":
    main()
