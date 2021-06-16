import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.autograd import Variable
import numpy as np
import knn_cuda


import numpy as np

# referenced https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py

def knn_self(x, k=10, return_dist=False):
    _, ind = knn_cuda.forward(x,x,k)

    return ind

    """
    with torch.no_grad():
        xx = torch.sum(x**2, dim=1, keepdim=True)
        d = xx.repeat(1, xx.shape[0]).add_(xx.T)
        d = d.add_(torch.matmul(x, x.T).mul_(-2))

    ind = torch.topk(-d, k=k, dim=1).indices

    if return_dist:
        return ind, d[ind]
    else:
        return ind
    """

def graph_features(x, k=10):
    n = x.shape[-2]
    ind = knn_self(x, k).long()

    ones = [1]*(len(x.shape)-1)

    x_ = x.unsqueeze(-2).repeat(*ones,k,1)
    feat = torch.cat([x[ind] - x_, x_], dim=-1)

    return feat.view(*x.shape[:-2], n*k, -1)
    


def lin(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels, bias=False),
                        #nn.BatchNorm1d(out_channels),
                        #nn.LeakyReLU(negative_slope=0.2))
                        nn.PReLU())
    
    
class EdgeConv(nn.Module):
    def __init__(self, layers, k=10):   
        super(EdgeConv,self).__init__()
        self.layers = layers
        self.k = k

    def forward(self, x):
        k = self.k
        n = x.shape[-2]                                      # n x c

        x = graph_features(x, k=k)                          # (n*k) x c
        x = self.layers(x)                                   # (n*k) x c'
        x = x.view(n, k, -1)                                # n x k x c'
        x = x.max(dim=-2, keepdim=False)[0].contiguous()     # n x c'

        return x

class DGFC(nn.Module):
    def __init__(self, in_features, out_features, mid_features, k=10):
        super(DGFC, self).__init__()
        self.layers = nn.Sequential(EdgeConv(lin(in_features*2, mid_features), k=k),
                      *([EdgeConv(lin(mid_features*2, mid_features), k=k)]*5),
                      EdgeConv(lin(mid_features*2, out_features), k=k))
        
    def forward(self, x, hidden):
        return self.layers(x), hidden


class DGCNN(nn.Module):
    # implementation of segmentation network, without classification

    def __init__(self, in_channels, out_channels, k=10):
        super(DGCNN, self).__init__()
        self.k=k

        self.conv1 = EdgeConv(nn.Sequential(lin(in_channels*2, 64), lin(64, 64)), k=k)
        self.conv2 = EdgeConv(nn.Sequential(lin(64*2, 64), lin(64, 64)), k=k)
        self.conv3 = EdgeConv(lin(64*2, 64), k=k)
        self.conv4 = lin(64*3, 1024)

        self.linear1 = lin(1024 + 64*3, 256)
        self.linear2 = lin(256, 256)
        self.linear3 = lin(256, 128)
        self.linear4 = lin(128, out_channels)

        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
        self.dp3 = nn.Dropout(p=0.2)

        self.linears = nn.Sequential(self.linear1, self.dp1,
                                     self.linear2, self.dp2,
                                     self.linear3, self.dp3,
                                     self.linear4)
    
    def forward(self, x, hidden):
        n = x.shape[0]              # n x ci

        x1 = self.conv1(x)          # n x 64
        x2 = self.conv2(x1)         # n x 64
        x3 = self.conv3(x2)         # n x 64
        
        x4 = torch.cat([x1, x2, x3], dim=-1)                 # n x 192
        
        x5 = self.conv4(x4)                                 # n x 1024
        x5 = x5.max(dim=0, keepdim=True)[0].repeat(n,1)     # n x 1024
        x = torch.cat([x1, x2, x3, x5], dim=-1)              # n x (1024 + 64*3)

        x = self.linears(x)         # n x co
        
        return x, hidden

class LGD(nn.Module):
    def __init__(self, dim_targets, num_losses, k=10, concat_input=True):
        super(LGD, self).__init__()
 
        hidden_features = 0
 
        self.dim_targets = dim_targets              # D: total dimension of targets
        self.num_losses = num_losses                # L: number of losses
 
        self.hidden_features = hidden_features      # H: hidden state
        self.k = k                                  # K: k-nearest neighbors
        self.concat_input = concat_input
        
        # layers : (D+1)*L + H -> L + H
 
        inc = dim_targets * num_losses + hidden_features
        if concat_input:
            inc += dim_targets

        ouc = 2*num_losses + hidden_features
        
 
        self.layers = DGCNN(inc, ouc, k=k)
 
        self.init_params()
 
 
    def init_params(self):
        for m in self.modules():
            try:
                weight = getattr(m, 'weight')
 
                if len(weight.shape) > 1:
                    with torch.no_grad():
                        k = weight.shape[1]
                        a = np.sqrt(0.75 / k)
                        weight.data.uniform_(-a, a)
 
            except AttributeError:
                continue
 
            try:
                bias = getattr(m, 'bias')
                if bias is not None:
                    with torch.no_grad():
                        bias.data = torch.zeros_like(bias.data)
 
            except AttributeError:
                continue
 
 
    def forward(self, targets, losses, hidden=None, batch_size=1):
        # targets : list of targets to optimize; flattened to (n, -1) later
        # losses : list of losses
 
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]

        h = self.hidden_features
        n = np.prod(list(targets[0].shape[:-1])) #flatten the batch groups such that our input is in the shape (n, c)
        t = len(targets)

        assert sum([target.shape[1] for target in targets]) == self.dim_targets
        assert len(losses) == self.num_losses
        assert np.prod([np.prod(list(target.shape[:-1])) == n for target in targets]) == 1      # assume batch size is same for every parameter
        
 
        targets_grad = torch.zeros((n, 0)).to(targets[0].device)
 
        if hidden is None:
            hidden = torch.zeros((2, n, h)).to(targets[0].device)
 
        # input : dL1/dx1, dL1/dx2, ..., dL2/dx1, dL2/dx2, ..., hidden
        # input size : L*D + H
        for loss_f in losses:
            loss = loss_f(targets)
            targets_grad_l = torch.autograd.grad(loss, targets, grad_outputs=[torch.ones_like(loss) for _ in range(t)], create_graph=False)
            targets_grad_l = [grad.view(n, -1) for grad in targets_grad_l]
            targets_grad = torch.cat([targets_grad, *targets_grad_l], dim=1)
 
        targets_grad[torch.isnan(targets_grad)] = 0
        targets_grad[torch.isinf(targets_grad)] = 0

        if self.concat_input:
            targets = [target.view(n, -1) for target in targets]
            x = torch.cat([*targets, targets_grad], dim=1)
        
        else:
            x = targets_grad
 
        # output : new lr, new hidden
        # output size : L + H
        lr, hidden = self.layers(x, hidden)
 
        return lr, hidden, x
 
    def step(self, targets, losses, hidden=None, batch_size=1, return_lr=False):
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]

        lr, hidden, dx = self(targets, losses, hidden, batch_size)

        idx_start = self.dim_targets if self.concat_input else 0

        dx = dx[:,idx_start:].view(batch_size, self.num_losses, self.dim_targets)

        # lr[:, :self.num_losses] = learning rate (sigma)
        # lr[:, self.num_losses:] = evaluation rate (lambda)

        d_target = (lr[:,:self.num_losses].unsqueeze(-1) * dx).sum(dim=1)

        new_targets = []
        k = 0
 
        for target in targets:
            d = target.shape[1]
 
            new_targets.append(target + d_target[:,k:k+d].view(target.shape))
 
            k += d
        
        if return_lr:
            return new_targets, hidden, lr
        else:
            return new_targets, hidden
 

    def loss_trajectory_backward(self, targets, losses, hidden=None, constraints = None, batch_size=1, steps=10):
        # used for training LGD model itself
 
        # since x changes after each iteration, we need to evaluate the loss again
        # so we input loss func, rather than a pre-computed loss

        if constraints is None:
            constraints = ["None"] * len(losses)
        else:    
            assert len(constraints) == len(losses)
        
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]

        loss_sum = 0
        lambda_sum = 0
        sigma_sum = 0

        for step in range(steps):
            targets, _, lr = self.step(targets, losses, hidden, batch_size, return_lr=True)
    
            # apply contraints on lambda
            lr.requires_grad_()
            lr_filtered = lr.clone().requires_grad_()


            for i, constraint in enumerate(constraints):
                if constraint is "None":
                    # no constraint on loss  -> lambda = 1
                    lr_filtered[:,self.num_losses+i] = 1
                elif constraint is "Zero":
                    # loss = 0  -> no constraint on lambda
                    continue
                elif constraint is "Positive":
                    # loss >= 0 -> lambda <= 0
                    lr_filtered[:,self.num_losses+i] = -F.relu(-lr_filtered[:,self.num_losses+i])
                elif constraint is "Negative":
                    # loss <= 0 -> lambda >= 0
                    lr_filtered[:,self.num_losses+i] = F.relu(lr_filtered[:,self.num_losses+i])

 
            for i, loss_f in enumerate(losses):
                loss = (lr_filtered[:,self.num_losses+i] * loss_f(targets)).mean() / steps

                # evaluate dL / d(sigma), dL / d(lambda)
                # propagate with d(sigma) / d(theta) : descent, d(lambda) / d(theta) : ascent

                d_lr = torch.autograd.grad([loss], [lr], grad_outputs=[torch.ones_like(loss)], create_graph=False, retain_graph=True)[0]

                lr[:,:self.num_losses].backward(d_lr[:,:self.num_losses], retain_graph=True)
                lr[:,self.num_losses:].backward(-d_lr[:,self.num_losses:], retain_graph=True)

                loss_sum += loss.detach()
                sigma_sum += lr[:,:self.num_losses].mean() / steps
                lambda_sum += lr[:,self.num_losses:].mean() / steps
        
        #plt.scatter(targets[0][:,0].detach().cpu().numpy(), targets[0][:,1].detach().cpu().numpy())
        #plt.show()

        return loss_sum, sigma_sum, lambda_sum
 
 
def detach_var(v):
    if v is None:
        return v
        
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var
