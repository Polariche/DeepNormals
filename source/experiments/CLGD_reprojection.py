import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

#from models.models import SingleBVPNet, DeepSDFDecoder, Siren
from loaders import CategoryDataset, dict_collate_fn, dict_to_device
from models.CLGD import CLGD
from torch.utils.data import  DataLoader

import argparse


import time
from tqdm.autonotebook import tqdm
from knn import knn
import utils


def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--tb-save-path', dest='tb_save_path', type=str, metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--weight-save-path', dest='weight_save_path', type=str, metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=1e-3, 
                            help='learning rate')
    parser.add_argument('--lgd-step', dest='lgd_step_per_epoch', type=int,metavar='LGD_STEP_PER_EPOCH', default=5, 
                            help='number of simulation steps of LGD per epoch')


    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    category = CategoryDataset(srn_dir="/data/SRN/cars_train", 
                                shapenet_dir="/data/shapenetv2/ShapeNetCore.v2", 
                                img_sidelength=512, 
                                batch_size=args.batchsize, 
                                ray_batch_size=3)

    category_loader = DataLoader(category, batch_size=1, shuffle=True)


    clgd = CLGD().to(device)
    clgd_optimizer = optim.Adam(clgd.parameters(), lr=args.lr)

    # train LGD
    clgd.train()
    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            samples = next(iter(category_loader))
            samples = dict_to_device(samples, device)

            X = (torch.randn_like(samples['p']) * 2e-1).requires_grad_(True)
            H = torch.zeros(*samples['p'].shape[:-2], 1, 256, device=device).requires_grad_(True)
            G = torch.zeros(*samples['p'].shape[:-1], 256, device=device).requires_grad_(True)
            P = samples['pose']

            Y = samples['p']

            clgd.Y = Y

            clgd_optimizer.zero_grad()
            X_new, X_new_grad = clgd.backward(X, H, P, G)
            clgd_optimizer.step()

            writer.add_mesh("input_view",
                            (samples['p']).reshape(-1,3).unsqueeze(0),
                            global_step=i+1,
                            colors=(torch.clamp((F.normalize(samples['n'], dim=-1).reshape(-1,3).unsqueeze(0)), -1, 1) * 128 + 128).int())

            writer.add_mesh("output_view",
                            (X_new).reshape(-1,3).unsqueeze(0),
                            global_step=i+1,
                            colors=(F.normalize(X_new_grad, dim=-1).reshape(-1,3).unsqueeze(0) * 128 + 128).int())

            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()