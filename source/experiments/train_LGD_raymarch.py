import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from models.models import SingleBVPNet, DeepSDFDecoder, Siren
from loaders import SceneClassDataset, RayDataset, dict_collate_fn, PointTransform
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
        model = SingleBVPNet(type="sine", in_features=3)
    elif args.sdf_model == "OldSiren":
        model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True)
    else:
        raise NotImplementedError  
    model.to(device)
        
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
        # load a RayDataset
        rays = RayDataset(args.width, args.height)
        rays.apply_pose(PointTransform(rotation=torch.eye(3) * 0.5, translation=torch.tensor([0., 0., -0.5])))

        rayloader = DataLoader(rays, collate_fn=dict_collate_fn, batch_size=args.batchsize, shuffle=True)
    
    print("lgd")

    lgd = LGD(1, 3, k=10).to(device)
    lgd_optimizer = optim.Adam(lgd.parameters(), lr=args.lr)

    # train LGD
    lgd.train()
    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            iter_ray = iter(rayloader)
            sampled_rays = next(iter_ray)

            d = sampled_rays['d'].to(device)
            p = sampled_rays['p'].to(device)
            n = sampled_rays['n'].to(device)

            l1 = lambda targets: torch.pow(targets[0], 2).sum(dim=1).mean()
            l2 = lambda targets: torch.pow(model(p + targets[0]*n), 2).sum(dim=1).mean()
            l3 = lambda targets: (torch.tanh(targets[0]) - 1).sum(dim=1).mean()
            ray_pt = lambda targets: p + targets[0]*n

            # update lgd parameters
            lgd_optimizer.zero_grad()
            train_loss, sigma_sum, lambda_sum, [d_converged] = lgd.loss_trajectory_backward(d, [l1, l2, l3], 
                                                                                            None, 
                                                                                            constraints=["None", "Zero", "Positive"], 
                                                                                            steps=args.lgd_step_per_epoch)
            lgd_optimizer.step()
            
            tqdm.write("Epoch %d, Total loss %0.6f, Sigma %0.6f, Lambda %0.6f, iteration time %0.6f" % (i, train_loss[0], sigma_sum, lambda_sum, time.time() - start_time))

            writer.add_mesh("pointcloud_LGD_train", (p+d_converged*n).unsqueeze(0), global_step=i+1)
            writer.add_scalars("train_loss", {"raymarch_LGD_train": train_loss[0]}, global_step=i)

            torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()