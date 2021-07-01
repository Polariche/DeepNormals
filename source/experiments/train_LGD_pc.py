import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from torch.utils.tensorboard import SummaryWriter

import sys
import os

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import SirenDecoder, DeepSDFDecoder, Siren
from loaders import SceneClassDataset, RayDataset, dict_collate_fn, PointTransform
from models.LGD import LGD, detach_var

from torch.utils.data import  DataLoader

import argparse

import time
from tqdm.autonotebook import tqdm

def main():
    parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DATA', help='path to file')

    parser.add_argument('--tb-save-path', dest='tb_save_path', type=str, metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--weight-save-path', dest='weight_save_path', type=str, metavar='PATH', default='../weights/', 
                            help='weight checkpoints path')

    parser.add_argument('--sdf-weight', dest='sdf_weight', type=str, metavar='PATH', default=None, 
                            help='pretrained weight for SDF model')

    parser.add_argument('--sdf-model', dest='sdf_model', type=str, metavar='SDFTYPE', default="DeepSDF", 
                            help='SDF model; DeepSDF or Siren')

    parser.add_argument('--sdf-specs', dest='sdf_sepcs', type=str, metavar='PATH', default='../checkpoints/', 
                            help='tensorboard checkpoints path')

    parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                            help='batch size')
    parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=500, 
                            help='epochs for adam and lgd')
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=1e-3, 
                            help='learning rate')
    parser.add_argument('--lgd-step', dest='lgd_step_per_epoch', type=int,metavar='LGD_STEP_PER_EPOCH', default=5, 
                            help='number of simulation steps of LGD per epoch')

    parser.add_argument('--width', dest='width', type=int,metavar='WIDTH', default=128, 
                            help='width for rendered image')
    parser.add_argument('--height', dest='height', type=int,metavar='HEIGHT', default=128, 
                            help='height for rendered image')

    parser.add_argument('--hidden-feats', dest='hidden_features', type=int,metavar='HIDDEN_FEATURES', default=64, 
                            help='hidden feature dimension')
    parser.add_argument('--hidden-type', dest='hidden_type', metavar='HIDDEN_TYPE', default='autodecoder', 
                            help='how hidden features will be handled; \'autodecoder\' or \'lstm\'')


    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
    # Open a SDF model
    if args.sdf_model == "DeepSDF":
        with open(args.sdf_specs) as specs_file:
            specs = json.load(specs_file)
            model = DeepSDFDecoder(specs["CodeLength"], **specs["NetworkSpecs"])
    elif args.sdf_model == "Siren":
        model = SirenDecoder(mode='mlp')
    elif args.sdf_model == "OldSiren":
        model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True)
    else:
        raise NotImplementedError  

    if args.sdf_weight != None:
        try:
            model.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    # fix SDF model weight
    model.to(device)
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False


    # Create LGD
    hidden_features = args.hidden_features

    if args.hidden_type == 'autodecoder':
        lgd = LGD(3+hidden_features, 1, k=10, hidden_features=0).to(device)
    elif args.hidden_type == 'lstm':
        lgd = LGD(3, 1, k=10, hidden_features=hidden_features).to(device)
    else:
        raise NotImplementedError

    lgd_optimizer = optim.Adam(lgd.parameters(), lr=args.lr, weight_decay=1e-3)

    p = (2*torch.rand(args.batchsize, 3)-1).to(device).requires_grad_()

    # train LGD
    lgd.train()
    with tqdm(total=args.epoch) as pbar:

        for i in range(args.epoch):
            start_time = time.time()

            samples_n = args.batchsize//32
            sample_inds = torch.randperm(args.batchsize)[:samples_n]
            p_sampled = p[sample_inds]

            hidden = torch.zeros((*p_sampled.shape[:-1], hidden_features), device=device).requires_grad_()

            l2 = lambda targets: torch.pow(model(targets[0]), 2).sum(dim=1).mean()

            # update lgd parameters
            lgd_optimizer.zero_grad()

            #if args.hidden_type == 'autodecoder':
            #    train_loss, sigma_sum, lambda_sum, [p] = lgd.loss_trajectory_backward([p, hidden], [l2], 
            #                                                                            hidden=None, 
            #                                                                            constraints=["Zero"],
            #                                                                            #additional=ray_pt,
            #                                                                            steps=args.lgd_step_per_epoch)
            #elif args.hidden_type == 'lstm':
            train_loss, sigma_sum, lambda_sum, [p_converged] = lgd.loss_trajectory_backward(p_sampled, [l2], 
                                                                                    hidden=hidden, 
                                                                                    constraints=["Zero"],
                                                                                    #additional=ray_pt,
                                                                                    steps=args.lgd_step_per_epoch)
            #else:
            #    raise NotImplementedError
            
            lgd_optimizer.step()

            tqdm.write("Epoch %d, Total loss %0.6f, Sigma %0.6f, Lambda %0.6f, iteration time %0.6f" % (i, train_loss[0], sigma_sum, lambda_sum, time.time() - start_time))
            
            writer.add_mesh("pointcloud_LGD_train", p_converged.unsqueeze(0), global_step=i+1)
            writer.add_scalars("train_loss", {"raymarch_LGD_train": train_loss[0]}, global_step=i)

            torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)

            pbar.update(1)
 
    writer.close()

if __name__ == "__main__":
    main()
