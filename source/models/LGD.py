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

def graph_features(x, k=10):
    n = x.shape[-2]
    c = x.shape[-1]
    shp = x.shape[:-2]

    x = x.view(-1, n, c)
    feat = torch.zeros(x.shape[0], n, k, 2*c, device=x.device)

    for i, x_ in enumerate(x):
        # for batched inputs (len(x.shape) > 2), knn should be applied inside a batch

        ind = knn_self(x_, k).long()

        x_rep = x_.unsqueeze(-2).repeat(1,k,1)
        feat[i] = torch.cat([x_[ind,:] - x_rep, x_rep], dim=-1)

    return feat.view(*shp, n*k, 2*c)
    


def lin(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels, bias=False),
                        nn.PReLU())
    
    
class EdgeConv(nn.Module):
    def __init__(self, layers, k=10):   
        super(EdgeConv,self).__init__()
        self.layers = layers
        self.k = k

    def forward(self, x):
        shp = x.shape[:-2]

        k = self.k
        n = x.shape[-2]                                      # n x c

        x = graph_features(x, k=k)                           # (n*k) x c
        x = self.layers(x)                                   # (n*k) x c'
        x = x.view(*shp, n, k, -1)                           # n x k x c'
        x = x.max(dim=-2, keepdim=False)[0].contiguous()     # n x c'

        return x


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
    
    def forward(self, x):
        n = x.shape[-2]              # n x ci

        x1 = self.conv1(x)          # n x 64
        x2 = self.conv2(x1)         # n x 64
        x3 = self.conv3(x2)         # n x 64
        
        x4 = torch.cat([x1, x2, x3], dim=-1)                 # n x 192
        
        x5 = self.conv4(x4)                                 # n x 1024
        x5 = x5.max(dim=-2, keepdim=True)[0].repeat(*([1]*(len(x5.shape)-2)), n, 1)     # n x 1024
        x = torch.cat([x1, x2, x3, x5], dim=-1)              # n x (1024 + 64*3)

        x = self.linears(x)         # n x co
        
        return x


class Renderer(nn.Module):
    def __init__(self, input_dim, sdf=None, color=None, include_loss=True, include_grad=True):
        super(Renderer, self).__init__()
        self.input_dim = input_dim

        self.set_sdf(sdf)
        self.set_color(color)

        self.inc = 2*self.input_dim
        if include_loss:
            # include sdf loss
            self.inc += 1
        if include_grad:
            self.inc += self.input_dim

        self.include_loss = include_loss
        self.include_grad = include_grad

        self.layers = DGCNN(self.inc, 4)

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

    def set_sdf(self, sdf):
        # sdf network
        self.sdf = sdf
    
    def set_color(self, color):
        # color network
        self.color = color

    def sdf_loss(self, x, mean=True, keepdim=False):
        assert self.sdf is not None 

        if mean:
            dims = tuple(range(x.dim()))
            return torch.pow(self.sdf(x),2).mean(dim=dims, keepdim=keepdim)
        else:
            return torch.pow(self.sdf(x),2)

    def color_loss(self, x, r, gt_color, mean=True, keepdim=False):
        assert self.color is not None 

        inp = torch.cat([x,r], dim=-1)

        if mean:
            dims = tuple(range(x.dim()))
            return torch.pow(self.color(inp) - gt_color, 2).sum(dim=-1, keepdim=True).mean(dim=dims, keepdim=keepdim)
        else:
            return torch.pow(self.color(inp) - gt_color, 2).sum(dim=-1, keepdim=True)


    def forward(self, rays):
        # d shape : (..., n, 3)
        # x0 shape : (..., 1, 3) or (..., n, 3)
        # r shape : (..., n, 3)

        d, x0, r = rays['d'], rays['p'], rays['n']

        assert x0.shape[-2] == 1 or x0.shape[-2] == d.shape[-2]

        assert d.shape[:-2] == x0.shape[:-2]
        assert d.shape[:-2] == r.shape[:-2]
        assert d.requires_grad == True

        x = x0 + d*r

        sdf_res = self.sdf(x)
        sdf_grad = torch.autograd.grad(sdf_res, 
                                        [x], 
                                        grad_outputs=torch.ones_like(sdf_res), 
                                        create_graph=False,
                                        retain_graph=True)[0].view(x.shape)
        
        layer_input = torch.cat([x, r], dim=-1)

        if self.include_loss:
            layer_input = torch.cat([layer_input, sdf_res], dim=-1)
        if self.include_grad:
            layer_input = torch.cat([layer_input, sdf_grad], dim=-1)

        layer_output = self.layers(layer_input)
        lr1, lr2, lag1, lag2 = layer_output[..., 0:1], layer_output[..., 1:2], layer_output[..., 2:3], layer_output[..., 3:4]

        return lr1, lr2, lag1, lag2, x, sdf_res, sdf_grad


    def step(self, rays):
        d, x0, r = rays['d'], rays['p'], rays['n']
        lr1, lr2, lag1, lag2, x, sdf_res, dx = self(rays)
        
        sdf_loss = sdf_res.mean()
        sdf_grad_d = torch.autograd.grad(sdf_loss, 
                                        [d], 
                                        grad_outputs=torch.ones_like(sdf_loss), 
                                        create_graph=False,
                                        retain_graph=True)[0].view(d.shape)

        """
        color_loss = self.color_loss(x, r, dx, rays['rgb'], mean=True)
        color_grad_d = torch.autograd.grad(color_loss, 
                                        [d], 
                                        grad_outputs=torch.ones_like(color_loss), 
                                        create_graph=False,
                                        retain_graph=True)[0].view(d.shape)
        """
        d2_grad_d = 2*d

        dd = lr1 * sdf_grad_d + lr2 * d2_grad_d
        dd = F.relu(dd)

        new_d = d + dd

        return new_d, [lr1, lr2, lag1, lag2]

    

    def loss_trajectory_backward(self, rays, steps=10):
        d, x0, r = rays['d'], rays['p'], rays['n']
        
        for step in range(steps):
            d, grad_targets = self.step(rays)
            rays['d'] = d

        lr1, lr2, lag1, lag2 = grad_targets

        x = x0+d*r

        sdf_res = self.sdf(x)
        dx = torch.autograd.grad(sdf_res, 
                                [x], 
                                grad_outputs=torch.ones_like(sdf_res), 
                                create_graph=False,
                                retain_graph=True)[0].view(x.shape)

        # backpropagation for renderer (=self.layers)
        sdf_loss = torch.pow(sdf_res, 2)
        color_loss = self.color_loss(x, r, rays['rgb'], mean=False)

        final_loss = torch.pow(d, 2).mean() + (lag1 * sdf_loss).mean() + (lag2 * color_loss).mean()

        grads = torch.autograd.grad(final_loss, 
                                grad_targets, 
                                grad_outputs=[torch.ones_like(final_loss)],
                                create_graph=False,
                                retain_graph=True,
                                allow_unused=True)

        layers_params = self.layers.parameters()
        for i, (target, grad) in enumerate(zip(grad_targets, grads)):
            if i >= 2: # lagrangian
                if grad is not None:
                    target.backward(- grad, retain_graph=True)#, inputs=layers_params)
            else:
                if grad is not None:
                    target.backward(grad, retain_graph=True)#, inputs=layers_params)

        # backpropagation for sdf (=self.sdf)
        sdf_gt = rays['visible'].view(-1)
        sdf_gt_loss = -torch.log(sdf_gt.view(-1,1)[sdf_gt]).sum()
        sdf_gt_loss += -torch.log(1 - sdf_gt.view(-1,1)[~sdf_gt]).sum()
        sdf_gt_loss /= sdf_gt.view(-1,1).shape[0] #torch.pow((sdf_res - sdf_gt), 2).mean()
        sdf_gt_loss.backward(retain_graph=True)#, inputs=self.sdf.parameters())



        rays['rgb'] = self.color(torch.cat([x,r], dim=-1))

        return final_loss.detach().item(), lr1.detach().mean().item(), lr2.detach().mean().item(), lag1.detach().mean().item(), lag2.detach().mean().item()


        

class LGD(nn.Module):
    def __init__(self, dim_targets, num_losses, hidden_features=0, additional_features=0, k=10, concat_input=True):
        super(LGD, self).__init__()
 
        self.dim_targets = dim_targets                  # D: total dimension of targets
        self.num_losses = num_losses                    # L: number of losses
 
        self.hidden_features = hidden_features          # H: hidden state
        self.additional_features = additional_features  # A : additional inputs
        self.k = k                                      # K: k-nearest neighbors
        self.concat_input = concat_input
        
        # layers : (D) + A + H + D*L -> 2*L + H
 
        inc = additional_features + hidden_features + dim_targets * num_losses 
        if concat_input:
            inc += dim_targets

        ouc = 2*num_losses + hidden_features

        self.inc = inc
        self.ouc = ouc

 
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
 
 
    def forward(self, targets, losses, hidden=None, additional=None):
        # targets : list of targets to optimize; flattened to (n, -1) later
        # losses : list of losses
 
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]
        

        h = self.hidden_features
        n = np.prod(list(targets[0].shape[:-1]))
        shp = targets[0].shape[:-1]
        t = len(targets)

        assert sum([target.shape[-1] for target in targets]) == self.dim_targets
        assert len(losses) == self.num_losses
        assert np.prod([np.prod(list(target.shape[:-1])) == n for target in targets]) == 1      # assume batch size is same for every parameter
        
 
        targets_grad = torch.zeros((*shp, 0)).to(targets[0].device)
 
        if hidden is None:
            hidden = torch.zeros((*shp, h)).to(targets[0].device)
        else:
            assert hidden.shape[-1] == self.hidden_features
        
        additional_tensor = torch.zeros((*shp, 0)).to(targets[0].device)
        if additional is not None:
            if type(additional) is not list:
                additional = [additional]
            
            for add in additional:
                if type(add) is torch.Tensor:
                    add = add.view(*shp, -1)
                elif type(add) is type(lambda x:x):
                    add = add(targets).view(*shp, -1)
                
                additional_tensor = torch.cat([additional_tensor, add], dim=-1)
            
            assert additional_tensor.shape[-1] == self.additional_features

        # input : dL1/dx1, dL1/dx2, ..., dL2/dx1, dL2/dx2, ..., hidden
        # input size : L*D + H
        for loss_f in losses:
            loss = loss_f(targets)
            targets_grad_l = torch.autograd.grad(loss, targets, grad_outputs=[torch.ones_like(loss) for _ in range(t)], create_graph=False, allow_unused=True)
            targets_grad_l = [grad.view(*shp, -1) if grad is not None else torch.zeros_like(targets[i]) for i, grad in enumerate(targets_grad_l)]
            targets_grad = torch.cat([targets_grad, *targets_grad_l], dim=-1)

        if self.concat_input:
            targets = [target.view(*shp, -1) for target in targets]
            x = torch.cat([*targets, additional_tensor, hidden, targets_grad], dim=-1)
        
        else:
            x = torch.cat([additional_tensor, hidden, targets_grad], dim=-1)
 
        # input dim : D + H + D*L
        # output : new lr, lambdas
        # output size : 2*L + H
        y = self.layers(x)
        lr = y[...,:self.num_losses*2]
        hidden = y[...,self.num_losses*2:]
 
        return lr, x, hidden
 
    def step(self, targets, losses, hidden=None, additional=None, return_lr=False):
        if type(targets) is not list:
            targets = [targets]
        if type(losses) is not list:
            losses = [losses]

        shp = targets[0].shape[:-1]

        lr, x, hidden = self(targets, losses, hidden=hidden, additional=additional)

        if self.concat_input:
            idx_start = self.dim_targets+self.hidden_features+self.additional_features
        else:
            idx_start = self.hidden_features+self.additional_features

        dx = x[...,idx_start:].view(*shp, self.num_losses, self.dim_targets)

        # lr[..., :self.num_losses] = learning rate (sigma)
        # lr[..., self.num_losses:self.num_losses*2] = evaluation rate (lambda)

        d_target = (lr[...,:self.num_losses].unsqueeze(-1) * dx).sum(dim=-2)

        new_targets = []
        k = 0
 
        for target in targets:
            d = target.shape[1]
 
            new_targets.append(target + d_target[...,k:k+d].view(target.shape))
 
            k += d
        
        if return_lr:
            return new_targets, lr, hidden
        else:
            return new_targets, hidden


    def loss_trajectory_backward(self, targets, losses, hidden=None, additional=None, constraints = None, steps=10):
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

        loss_sum = [0] * len(losses)
        lambda_sum = 0
        sigma_sum = 0

        for step in range(steps):
            targets, lr, hidden = self.step(targets, losses, hidden=hidden, additional=additional, return_lr=True)
    
            # apply contraints on lambda
            lr.requires_grad_()
            lr_filtered = lr.clone().requires_grad_()


            for i, constraint in enumerate(constraints):
                if type(constraint) is float or type(constraint) is int:
                    # fixed constraint
                    lr_filtered[:,self.num_losses+i] = constraint
                elif constraint is "Zero":
                    # loss = 0  -> no constraint on lambda
                    continue
                elif constraint is "Positive":
                    # loss >= 0 -> lambda <= 0
                    lr_filtered[:,self.num_losses+i] = -F.relu(-lr_filtered[:,self.num_losses+i])
                elif constraint is "Negative":
                    # loss <= 0 -> lambda >= 0
                    lr_filtered[:,self.num_losses+i] = F.relu(lr_filtered[:,self.num_losses+i])
                else:
                    # no constraint on loss  -> lambda = 1
                    lr_filtered[:,self.num_losses+i] = 1

            if step == steps-1:
                for i, loss_f in enumerate(losses):
                    loss_true = loss_f(targets)
                    loss = (lr_filtered[:,self.num_losses+i] * loss_true).mean() / steps

                    # evaluate dL / d(sigma), dL / d(lambda)
                    # propagate with d(sigma) / d(theta) : descent, d(lambda) / d(theta) : ascent

                    d_lr = torch.autograd.grad([loss], [lr], grad_outputs=[torch.ones_like(loss)], create_graph=False, retain_graph=True)[0]

                    lr[:,:self.num_losses].backward(d_lr[:,:self.num_losses], retain_graph=True)
                    lr[:,self.num_losses:self.num_losses*2].backward(-d_lr[:,self.num_losses:self.num_losses*2], retain_graph=True)

                    loss_sum[i] += loss_true.mean().detach() / steps
                    sigma_sum += lr[:,:self.num_losses].mean() / steps
                    lambda_sum += lr[:,self.num_losses:self.num_losses*2].mean() / steps

        return loss_sum, sigma_sum, lambda_sum, [target.detach() for target in targets]
 
 
def detach_var(v):
    if v is None:
        return v
        
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var
