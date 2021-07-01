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

from models.models import SingleBVPNet, DeepSDFDecoder
from loaders import SceneClassDataset, RayDataset, dict_collate_fn, PointTransform
from models.LGD import LGD, detach_var

from torch.utils.data import  DataLoader

import argparse

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
    parser.add_argument('--lr', dest='lr', type=float,metavar='LEARNING_RATE', default=5e-3, 
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
    if args.sdf_model is "DeepSDF":
        with open(args.sdf_specs) as specs_file:
            specs = json.load(specs_file)
            model = DeepSDFDecoder(specs["CodeLength"], **specs["NetworkSpecs"])

    elif args.sdf_model is "Siren":
        model = SingleBVPNet(type="sine", in_features=3)

    if args.sdf_weight != None:
        try:
            model.load_state_dict(torch.load(args.sdf_weight))
        except:
            print("Couldn't load pretrained weight: " + args.sdf_weight)

    # fix SDF model weight
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False


    # Create LGD
    hidden_features = args.hidden_features

    if args.hidden_type is 'autodecoder':
        lgd = LGD(1+hidden_features, 3, k=10, hidden_features=0, additional_features=3).to(device)
    elif args.hidden_type is 'lstm':
        lgd = LGD(1, 3, k=10, hidden_features=hidden_features, additional_features=3).to(device)
    else:
        raise NotImplementedError

    lgd_optimizer = optim.Adam(lgd.parameters(), lr= args.lr)


    # load a RayDataset
    rays = RayDataset(args.width, args.height)
    rayloader = DataLoader(rays, collate_fn=dict_collate_fn, batch_size=args.batchsize, shuffle=True)
    rayloader.apply_pose(PointTransform(rotation=torch.eye(3) * 0.5, position=torch.tensor(0., 0., -0.5)))

    # train LGD
    lgd.train()
    with tqdm(total=args.epochs) as pbar:

        for i in range(args.epoch):
            start_time = time.time()

            iter_ray = iter(rayloader)
            sampled_rays = next(iter_ray)

            d = sampled_rays['d'].cuda()
            p = sampled_rays['p'].cuda()
            n = sampled_rays['n'].cuda()
            hidden = torch.zeros((*d.shape[:-1], hidden_features))

            l1 = lambda targets: torch.pow(targets[0], 2).mean()
            l2 = lambda targets: torch.pow(model(p + targets[0]*n), 2).mean()
            l3 = lambda targets: (torch.tanh(targets[0]) - 1).mean()
            ray_pt = lambda targets: p + targets[0]*n


            # update lgd parameters
            lgd_optimizer.zero_grad()

            if args.hidden_type is 'autodecoder':
                train_loss, _, _ = lgd.loss_trajectory_backward([d, hidden], [l1, l2, l3], 
                                        hidden=None, 
                                        constraints=["None", "Zero", "Positive"],
                                        additional=ray_pt,
                                        steps=args.lgd_step_per_epoch)
            elif args.hidden_type is 'lstm':
                train_loss, _, _ = lgd.loss_trajectory_backward(d, [l1, l2, l3], 
                                        hidden=hidden, 
                                        constraints=["None", "Zero", "Positive"],
                                        additional=ray_pt,
                                        steps=args.lgd_step_per_epoch)
            else:
                raise NotImplementedError
            
            lgd_optimizer.step()

            tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (i, train_loss[1], time.time() - start_time))
            torch.save(lgd.state_dict(), args.weight_save_path+'model_%03d.pth' % i)
            
            pbar.update(1)
 
    writer.close()

if __name__ == "__main__":
    main()
