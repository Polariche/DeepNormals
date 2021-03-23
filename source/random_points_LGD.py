import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import Siren
from utils import Sobel
from loaders import ObjDataset
import utils
from torch.utils.data import  DataLoader, WeightedRandomSampler

from LGD import LGD, detach_var

import argparse

from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser(description='Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DATA', help='path to file')

parser.add_argument('--tb-save-path', dest='tb_save_path', metavar='PATH', default='../checkpoints/', 
                        help='tensorboard checkpoints path')

parser.add_argument('--weight-save-path', dest='weight_save_path', metavar='PATH', default='../weights/', 
                        help='weight checkpoints path')

parser.add_argument('--pretrained-weight', dest='weight', metavar='PATH', default=None, 
                        help='pretrained weight')

parser.add_argument('--activation', dest='activation', metavar='activation', default='relu', 
                        help='activation of network; \'relu\' or \'sin\'')

parser.add_argument('--batchsize', dest='batchsize', type=int, metavar='BATCHSIZE', default=1,
                        help='batch size')
parser.add_argument('--epoch', dest='epoch', type=int,metavar='EPOCH', default=100, 
                        help='epochs')

parser.add_argument('--epsilon', dest='epsilon', type=float, metavar='EPSILON', default=0.1, 
                        help='epsilon')
parser.add_argument('--omega', dest='omega', type=float, metavar='OMEGA', default=30, 
                        help='hyperparameter for periodic layer')
parser.add_argument('--lambda', dest='lamb', type=float, metavar='LAMBDA', default=0.005, 
                        help='hyperparameter for s : normal loss ratio')


parser.add_argument('--outfile', dest='outfile', metavar='OUTFILE', 
                        help='output file')

def chamfer_distance(p1, p2, use_torch=False):
    if use_torch:
        # O(n^2) memory GPU computation on Torch; faster, but more expensive
        cd = torch.cdist(p1, p2)

        return torch.min(cd, dim=0)[0].mean() + torch.min(cd, dim=1)[0].mean()

    else:
        # O(nlog n) memory CPU computation with Sklearn; slower, but cheaper
        p1 = p1.detach().cpu().numpy()
        p2 = p2.detach().cpu().numpy()

        p1_tree = KDTree(p1)
        p2_tree = KDTree(p2)

        d1, _ = p1_tree.query(p2)
        d2, _ = p2_tree.query(p1)

        return np.mean(np.power(d1,2)) + np.mean(np.power(d2,2))


def dist_from_to(p1, p2, requires_graph=True):
    p1_np = p1.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()

    p2_tree = KDTree(p2_np)

    d, ind = p2_tree.query(p1_np)

    if not requires_graph:
        # we don't need graph
        return torch.tensor(d, device=p1.device)   

    else:
        ind = torch.tensor(ind, device=p1.device)
        return torch.norm(p1 - p2[ind], dim=1, keepdim=True)

def nearest_from_to(p1, p2):
    p1_np = p1.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()

    p2_tree = KDTree(p2_np)

    _, ind = p2_tree.query(p1_np)

    return torch.tensor(ind, device=p1.device, dtype=torch.long).squeeze()


def main():
    args = parser.parse_args()

    writer = SummaryWriter(args.tb_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create models
    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True).to(device) 

    if args.weight != None:
        try:
            model.load_state_dict(torch.load(args.weight))
        except:
            print("Couldn't load pretrained weight: " + args.weight)

    model.eval() 
    for param in model.parameters():
        param.requires_grad = False

    n = 30000
    ds = ObjDataset(args.data)
    fnn = torch.abs(ds.fnn)
    samples = list(WeightedRandomSampler(fnn.view(-1) / torch.sum(fnn), n, replacement=True))

    data = [ds[samples[i]] for i in range(len(samples))]
    xyz = torch.cat([d['xyz'].unsqueeze(0) for d in data])
    xyz = xyz.to(device)
    
    # load 
    with torch.no_grad():
        mm = torch.min(xyz, dim=0)[0]
        mx = torch.max(xyz, dim=0)[0]

        x = torch.rand(n,3).to(device) * (mx - mm) + mm
        x.requires_grad_(True)

        x_original = x.clone().detach()
    
    sdf_eval = lambda x: torch.pow(model(x)[0], 2).sum(dim=1).mean()
    sdf_eval_list = lambda x: sdf_eval(x[0])

    eps = args.epsilon
    
    x_target = xyz[nearest_from_to(x, xyz)]

    print("adam")
    optimizer = optim.Adam([x], lr = 1e-3)

    for i in range(500):
        optimizer.zero_grad()

        loss = sdf_eval(x)
        loss.backward(retain_graph=True)

        optimizer.step()

        if i%10 == 0:
            writer.add_scalars("regression_loss", {"Adam": loss}, global_step=i)
            writer.add_mesh("point cloud regression_Adam", x.unsqueeze(0), global_step=i)

            writer.add_scalars("chamfer_distance", {"Adam": chamfer_distance(x, xyz)}, global_step=i)

    with torch.no_grad():
        x = x_original.clone().detach()
        x.requires_grad_(True)

    print("lgd")
    hidden = None

    lgd = LGD(3, 1, k=10).to(device)
    lgd_optimizer = optim.Adam(lgd.parameters(), lr=5e-4)

    # train LGD
    lgd.train()
    for i in range(100):
        print(i)
        # evaluate losses
        samples_n = n//32
        sample_inds = torch.randperm(n)[:samples_n]

        #gt_eval = lambda x: torch.clamp(torch.pow(x - x_target[sample_inds],2).sum(dim=1), -eps**2, eps**2).mean()
        gt_eval = lambda x: torch.pow(x - x_target[sample_inds],2).sum(dim=1).mean()
        gt_eval_list = lambda x: gt_eval(x[0])

        # update lgd parameters
        lgd_optimizer.zero_grad()
        lgd.trajectory_backward(x[sample_inds], sdf_eval_list, gt_eval_list, batch_size=samples_n, steps=15)
        lgd_optimizer.step()

    # test LGD
    lgd.eval()
    for i in range(200):
        # evaluate losses
        loss = sdf_eval(x).mean()
        # update x
        [x], hidden = lgd.step(x, sdf_eval_list, hidden, n)
        x = detach_var(x)
        hidden = detach_var(hidden)

        if i%10 == 0:
            writer.add_scalars("regression_loss", {"LGD": loss}, global_step=i)
            writer.add_mesh("point cloud regression_LGD", x.unsqueeze(0), global_step=i)

            #writer.add_scalars("chamfer_distance", {"LGD": chamfer_distance(x, xyz)}, global_step=i)
            
    
    writer.close()

if __name__ == "__main__":
    main()
