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
            sdf = DeepSDFDecoder(specs["CodeLength"], **specs["NetworkSpecs"])
    elif args.sdf_model == "Siren":
        sdf = SingleBVPNet(type="sine", in_features=3)
    elif args.sdf_model == "OldSiren":
        sdf = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True)
    else:
        raise NotImplementedError  
    sdf.to(device)
        
    if args.sdf_weight != None:
        try:
            sdf.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    # load 
    with torch.no_grad():
        # load a RayDataset
        instance = SceneDataset("/data/SRN/cars_train/88c884dd867d221984ae8a5736280c", img_sidelength=512, ray_batch_size=args.batchsize)
        instance_loader = DataLoader(instance, batch_size=2, shuffle=True)
    

    renderer = Renderer(3, sdf=model).to(device)
    renderer_optimizer = optim.Adam(renderer.parameters(), lr=args.lr)

    # train LGD
    renderer.train()
    with tqdm(total=args.epoch) as pbar:
        for i in range(args.epoch):
            start_time = time.time()

            ins = next(iter(instance_loader))
            
            #tqdm.write("Epoch %d, Total loss %0.6f, Sigma %0.6f, Lambda %0.6f, iteration time %0.6f" % (i, train_loss[1], sigma_sum, lambda_sum, time.time() - start_time))

            writer.add_mesh("pointcloud_LGD_train", (p+d_converged*n).unsqueeze(0), global_step=i+1)
            writer.add_scalars("train_loss", {"raymarch_LGD_train": train_loss[1]}, global_step=i)

            torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
            pbar.update(1)

    writer.close()

if __name__ == "__main__":
    main()