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
from loaders import ObjDataset, Dataset, PointTransform, get_obj_dataloader
from models.LGD import LGD, detach_var
from evaluate_functions import chamfer_distance, nearest_from_to, dist_from_to

from torch.utils.data import  DataLoader, WeightedRandomSampler

import argparse

from sklearn.neighbors import KDTree

import time
from tqdm.autonotebook import tqdm

def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--weight-save-path', dest='weight_save_path', metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')

    parser.add_argument('--sdf-weight', dest='sdf_weight', metavar='PATH', default=None, 
                            help='pretrained weight for SDF model')


    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=1e-3, 
                            help='learning rate')
    parser.add_argument('--lgd-step', dest='lgd_step_per_epoch', type=int,metavar='LGD_STEP_PER_EPOCH', default=5, 
                            help='number of simulation steps of LGD per epoch')
    parser.add_argument('--n', dest='n', type=int,metavar='N', default=30000, 
                            help='number of points to sample')

    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    n = args.n
    lr = args.lr
    epoch = args.epoch
    lgd_step_per_epoch = args.lgd_step_per_epoch

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
    with torch.no_grad():
        x = (2*torch.rand(n,3)-1).to(device)
        x.requires_grad_(True)

        x_original = x.clone().detach()
    
    origin_eval = lambda x: torch.pow(x_original - x, 2).sum(dim=1).mean()
    sdf_eval = lambda x: torch.pow(model(x), 2).sum(dim=1).mean()
    
    origin_eval_list = lambda x: origin_eval(x[0])
    sdf_eval_list = lambda x: sdf_eval(x[0])

    print("lgd")
    hidden = None

    lgd = LGD(3, 1, k=10).to(device)
    lgd_optimizer = optim.Adam(lgd.parameters(), lr= lr)

    # train LGD
    lgd.train()
    with tqdm(total=args.epoch) as pbar:
        for i in range(epoch):
            start_time = time.time()

            # evaluate losses
            samples_n = n//32
            sample_inds = torch.randperm(n)[:samples_n]

            origin_eval_batch = lambda x: torch.pow(x_original[sample_inds] - x, 2).sum(dim=1).mean()
            origin_eval_batch_list = lambda x: origin_eval_batch(x[0])

            # update lgd parameters
            lgd_optimizer.zero_grad()
            train_loss, sigma_sum, lambda_sum, [x_converged] = lgd.loss_trajectory_backward(x[sample_inds], [sdf_eval_list], None, 
                                        constraints=["None"], batch_size=samples_n, steps=lgd_step_per_epoch)
            lgd_optimizer.step()
            
            tqdm.write("Epoch %d, Total loss %0.6f, Sigma %0.6f, Lambda %0.6f, iteration time %0.6f" % (i, train_loss[0], sigma_sum, lambda_sum, time.time() - start_time))

            writer.add_mesh("pointcloud_LGD_train", x_converged.unsqueeze(0), global_step=i+1)
            writer.add_scalars("train_loss", {"raymarch_LGD_train": train_loss[0]}, global_step=i)

            torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
            pbar.update(1)
    writer.close()

if __name__ == "__main__":
    main()