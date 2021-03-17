import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.autograd import Variable
import numpy as np

# referenced https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py

def knn(x, k=10, return_dist=False):
    with torch.no_grad():
        xx = torch.sum(x**2, dim=1, keepdim=True)
        d = xx.repeat(1, xx.shape[0]).add_(xx.T)
        d = d.add_(torch.matmul(x, x.T).mul_(-2))

    ind = torch.topk(-d, k=k, dim=1).indices

    if return_dist:
        return ind, d[ind]
    else:
        return ind

def graph_features(x, k=10):
    n = x.shape[0]
    ind = knn(x, k)

    x_ = x.unsqueeze(1).repeat(1,k,1)
    feat = torch.cat([x[ind] - x_, x_], dim=2)

    return feat.view(n*k, -1)
    


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
        n = x.shape[0]                                      # n x c

        #x = graph_features(x, k=k)                          # (n*k) x c
        x = self.layers(x)                                  # (n*k) x c'
        #x = x.view(n, k, -1)                                # n x k x c'
        #x = x.max(dim=1, keepdim=False)[0].contiguous()     # n x c'

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

        self.conv1 = EdgeConv(nn.Sequential(lin(in_channels, 64), lin(64, 64)), k=k)
        self.conv2 = EdgeConv(nn.Sequential(lin(64, 64), lin(64, 64)), k=k)
        self.conv3 = EdgeConv(lin(64, 64), k=k)
        self.conv4 = lin(64*3, 1024)

        self.linear1 = lin(1024 + 64*3, 256)
        self.linear2 = lin(256, 256)
        self.linear3 = lin(256, 128)
        self.linear4 = lin(128, out_channels)

        self.dp1 = nn.Dropout(p=0.8)
        self.dp2 = nn.Dropout(p=0.8)
        self.dp3 = nn.Dropout(p=0.8)

        self.linears = nn.Sequential(self.linear1, self.dp1,
                                     self.linear2, self.dp2,
                                     self.linear3, self.dp3,
                                     self.linear4)
    
    def forward(self, x, hidden):
        n = x.shape[0]              # n x ci

        x1 = self.conv1(x)          # n x 64
        x2 = self.conv2(x1)         # n x 64
        x3 = self.conv3(x2)         # n x 64
        
        x4 = torch.cat([x1, x2, x3], dim=1)                 # n x 192
        
        x5 = self.conv4(x4)                                 # n x 1024
        x5 = x5.max(dim=0, keepdim=True)[0].repeat(n,1)     # n x 1024
        x = torch.cat([x1, x2, x3, x5], dim=1)              # n x (1024 + 64*3)

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
        
        # layers : L*D + H -> D + H

        inc = dim_targets * num_losses + hidden_features
        ouc = dim_targets + hidden_features

        if concat_input:
            inc += dim_targets

        #self.layers = DGFC(inc, ouc, mid_features, k=k)
        self.layers = DGCNN(inc, ouc, k=k)
        #self.layers = LGD_GRU(2, hidden_features)

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
        n = batch_size
        t = len(targets)
 
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
 
        if self.concat_input:
            targets = [target.view(n, -1) for target in targets]
            x = torch.cat([*targets, targets_grad], dim=1)
        
        else:
            x = targets_grad
 
        # output : new grad, new hidden
        # output size : D + H
        dx, hidden = self.layers(x, hidden)
 
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
 
            new_targets.append(target + dx[:, k:k+d].view(*target.shape))
 
            k += d
        
        if return_dx:
            return new_targets, hidden, dx
        else:
            return new_targets, hidden
 
    def trajectory_backward(self, targets, losses, hidden=None, batch_size=1, steps=10):
        # used for training LGD model itself

        # since x changes after each iteration, we need to evaluate the loss again
        # so we input loss func, rather than a pre-computed loss
        
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]
        
        for i in range(steps):
            print(i)

            targets, hidden, dx = self.step(targets, losses, hidden, batch_size, return_dx=True)

            loss = 0
            for loss_f in losses:
                loss += loss_f(targets).mean() 

            loss += 1e-4 * torch.pow(torch.norm(dx, dim=1),2).mean()

            loss /= steps
            loss.backward(retain_graph=True)
 
 
def detach_var(v):
    if v is None:
        return v
        
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var