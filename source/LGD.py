import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.autograd import Variable

import matplotlib.pyplot as plt


def detach_var(v):
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var

def apply_step(model, step):
    for op, np in zip(model.parameters(), step):
        op.data = np

class LGD(nn.Module):
    def __init__(self, params, layers_generator, n=1):
        super(LGD, self).__init__()
        
        self.layers_generator = layers_generator
        self.define_params(params, n)
        
        
    def define_params(self, params, n):
        k = 0
        for param in params:
            k += param.view(n,-1).shape[1]

        self.n = n
        self.k = k
        self.layers = self.layers_generator(in_features = k, out_features = k)
        self.params = params


    def forward(self, x):
        return self.layers(x)

    def zero_grad(self):
        for param in self.params:
            if param.grad != None:
                param.grad.zero_()

    def detach_params(self):
        new_params = []
        for param in self.params:
            new_params.append(detach_var(param))
        self.params = new_params
        return new_params
        

    def step(self):
        new_params = []
        n = self.n
        grads = torch.zeros((n, 0))

        k = [0]

        for i, param in enumerate(self.params):
            k_ = param.view(n,-1).shape[1]
            if param.grad is not None:
                grad = detach_var(param.grad).view(n, -1)
                grads = torch.cat([grads, grad], dim=1)
            else:
                grads = torch.cat([grads, torch.zeros((n, k_))], dim=1)

            k.append(k_ + k[i])

        grads = self(grads)

        for i, param in enumerate(self.params):
            param_ = param + grads[:,k[i]:k[i+1]].view(param.shape)
            param_.retain_grad()

            new_params.append(param_)

        self.params = new_params
        return new_params


    def apply_step(self, model):
        apply_step(model, self.step())
