import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.autograd import Variable
import numpy as np

class LGD(nn.Module):
    def __init__(self, dim_targets, num_losses, mid_features, hidden_features):
        super(LGD, self).__init__()
 
        self.dim_targets = dim_targets              # D: total dimension of targets
        self.num_losses = num_losses                # L: number of losses
        self.mid_features = mid_features
        self.hidden_features = hidden_features      # H: hidden state
 
        # layers : L*D + H -> D + H
        self.layers = nn.Sequential(nn.Linear(dim_targets * num_losses + hidden_features, mid_features, bias=False),
                      nn.PReLU(),
 
                      *([nn.Linear(mid_features, mid_features, bias=False), 
                         nn.PReLU()]*3),
 
                      nn.Linear(mid_features, dim_targets + hidden_features, bias=False))
 
        """
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.uniform_(param, -1 / mid_features, 1 / mid_features)
        """

    def forward(self, targets, losses, hidden=None, batch_size=1):
        # targets : list of targets to optimize; flattened to (n, -1) later
        # losses : list of losses
 
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]
 
        h = self.hidden_features
        n = batch_size
        t = len(targets)
        device = targets[0].device
 
        targets_grad = torch.zeros((n, 0)).to(device)
 
        if hidden is None:
            hidden = torch.zeros((n, h)).to(device)
 
        # input : dL1/dx1, dL1/dx2, ..., dL2/dx1, dL2/dx2, ..., hidden
        # input size : L*D + H
        for loss in losses:
            targets_grad_l = torch.autograd.grad(loss, targets, grad_outputs=[torch.ones_like(loss) for _ in range(t)], create_graph=True)
            targets_grad_l = [grad.view(n, -1) for grad in targets_grad_l]
            targets_grad = torch.cat([targets_grad, *targets_grad_l], dim=1)
 
        x = torch.cat([targets_grad, hidden], dim=1)

        # output : new grad, new hidden
        # output size : D + H
        y = self.layers(x)
 
        if h > 0:
            dx = y[:,:-h]
            hidden = y[:,-h:]
        else:
            dx = y
            hidden = None
 
        return dx, hidden
 
 
    def step(self, targets, losses, hidden=None, batch_size=1, return_dx=False):
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]
 
        dx, hidden = self(targets, losses, hidden, batch_size)
 
        new_targets = []
        k = 0
 
        for target in targets:
            d = target.shape[1]
 
            target = target + dx[:, k:k+d].view(*target.shape)
            new_targets.append(target)
 
            k += d
        
        if return_dx:
            return new_targets, hidden, dx
        else:
            return new_targets, hidden
 
    def loss_trajectory(self, targets, loss_func, hidden=None, batch_size=1, steps=10):
        # used for training LGD model itself
        if type(targets) is not list:
            targets = [targets]
 
        loss = loss_func(targets)
        for i in range(steps):
            targets, hidden, dx = self.step(targets, loss, hidden, batch_size, return_dx=True)
            loss = loss_func(targets)

            loss_trajectory = loss / steps
            loss_trajectory += 7*torch.norm(dx, dim=1).mean() / steps     # regularizer for dx

            loss_trajectory.backward(retain_graph=True)

 
def detach_var(v):
    if v is None:
        return v
        
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var