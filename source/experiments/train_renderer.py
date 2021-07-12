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
from loaders import RayDataset, SceneDataset, dict_collate_fn, PointTransform, dict_to_device
from models.LGD import Renderer

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


    parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                            help='output file')

    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Open a SDF model & coloring model
    if args.sdf_model == "DeepSDF":
        with open(args.sdf_specs) as specs_file:
            specs = json.load(specs_file)
            sdf = DeepSDFDecoder(specs["CodeLength"], **specs["NetworkSpecs"])
            color = nn.Sequential(DeepSDFDecoder(specs["CodeLength"], **specs["NetworkSpecs"]), nn.Tanh())

    elif args.sdf_model == "Siren":
        sdf = SingleBVPNet(type="sine", in_features=3)
        color = nn.Sequential(SingleBVPNet(type="sine", in_features=9, out_features=3), nn.Tanh())

    elif args.sdf_model == "OldSiren":
        sdf = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True)
        color = nn.Sequential(Siren(in_features=9, out_features=3, hidden_features=256, hidden_layers=5, outermost_linear=True), nn.Tanh())

    else:
        raise NotImplementedError

    sdf.to(device)
    color.to(device)    

    if args.sdf_weight != None:
        try:
            sdf.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    # load 
    with torch.no_grad():
        # load a SceneDataset
        instance = SceneDataset("/data/SRN/cars_train/88c884dd867d221984ae8a5736280c", img_sidelength=512, ray_batch_size=args.batchsize)
        instance_loader = DataLoader(instance, batch_size=2, shuffle=True)
    

    renderer = Renderer(3, sdf=sdf, color=color).to(device)
    renderer_optimizer = optim.Adam(renderer.parameters(), lr=args.lr)

    # train LGD
    renderer.train()
    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            ins = next(iter(instance_loader))
            ins = dict_to_device(ins, device)
            
            color = (ins['rgb'] * 128 + 128).int()

            writer.add_mesh("input_view", 
                            (ins['p']+ins['n']).reshape(-1,3).unsqueeze(0), 
                            global_step=i+1, 
                            colors=color.reshape(-1,3).unsqueeze(0))

            renderer_optimizer.zero_grad()
            total_loss, lr1, lr2, lag1, lag2 = renderer.loss_trajectory_backward(ins)
            renderer_optimizer.step()

            tqdm.write("Epoch %d, Total loss %0.6f, Sigma1 %0.6f, Sigma2 %0.6f, Lambda1 %0.6f, Lambda2 %0.6f, iteration time %0.6f" % (i, total_loss, lr1, lr2, lag1, lag2, time.time() - start_time))

            color = (ins['rgb'] * 128 + 128).int()
            writer.add_mesh("output_view", 
                            (ins['p']+ins['n'] * ins['d']).reshape(-1,3).unsqueeze(0), 
                            global_step=i+1, 
                            colors=color.reshape(-1,3).unsqueeze(0))

            #writer.add_scalars("train_loss", {"raymarch_LGD_train": train_loss[1]}, global_step=i)

            #torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()