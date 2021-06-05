import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neighbors import KDTree

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


def L2(x):
    return torch.pow(x, 2).mean()

def plus_sign(x):
    # 0 if plus, -1 if minus
    return (x / torch.abs(x) - 1).mean() * 0.5

def minus_sign(x):
    # 1 if plus, 0 if minus
    return (x / torch.abs(x) + 1).mean() * 0.5