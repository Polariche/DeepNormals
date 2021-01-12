import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.sparse
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, Sampler, RandomSampler, BatchSampler

from torch.hub import load_state_dict_from_url

from torch.autograd import Variable, grad
from typing import Type, Any, Callable, Union, List, Optional, Tuple


class DeepSDF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 512, last_activation = None, activation = 'relu', omega_0 = 30.):
        super(DeepSDF, self).__init__()
        
        c = [in_channels] + [mid_channels]*3 + [mid_channels - in_channels] + [mid_channels]*3 + [out_channels]
        
        if last_activation is None:
            last_activation = torch.tanh

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sin':
            self.activation = lambda x: omega_0*torch.sin(x)
            
        self.dropout = nn.Dropout(0.2)
        self.last_activation = last_activation
        
        for i in range(1,9):
            inc = c[i-1] if i!=5 else mid_channels
            ouc = c[i]
            
            fc = nn.Linear(inc, ouc)

            # fc2d has no uniform init, so do it manually
            with torch.no_grad():
                if i == 1:
                    const = 1 / mid_channels 
                else:
                    const = np.sqrt(6/ mid_channels) / omega_0

                fc.weight.data = const * (torch.rand(fc.weight.shape, dtype=fc.weight.dtype, requires_grad = True) * 2-1)
                fc.bias.data = const * (torch.rand(fc.bias.shape, dtype=fc.bias.dtype, requires_grad = True) * 2-1)

            fc = nn.utils.weight_norm(fc)
            bn = nn.LayerNorm(c[i])

            setattr(self, f'fc{i}', fc)
            setattr(self, f'bn{i}', bn)
            
    def forward(self, x):
        identity = x
        
        for i in range(1,9):
            fc = getattr(self, f'fc{i}')
            bn = getattr(self, f'bn{i}')
            
            if i==5:
                x = torch.cat([x, identity], dim=1)
                
            x = fc(x)
            x = bn(x)
            
            if i < 8:
                x = self.activation(x) #x = F.relu(x)
                x = self.dropout(x)
                
        
        x = self.last_activation(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, n):
        assert n % 6 == 0
        super(PositionalEncoding, self).__init__()
        self.pi = torch.acos(torch.zeros(1)).item()
        self.n = n
        
    def forward(self, x):
        pi = self.pi
        n = self.n
        
        x = torch.cat([x/(2**i)*pi for i in range(n//6)], dim=1)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=1)
        
        return x
    
    

# TODO : create optimizer model