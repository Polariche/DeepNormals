
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
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 512, last_activation = None, kernel_size=1, activation = 'relu', omega_0 = 30.):
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
            
            d = pow(2, (i-1)%4)
            k = kernel_size
            p = d*(k//2)
            
            conv = nn.Conv2d(inc, ouc, k, dilation=d, padding=p, padding_mode='replicate')

            # conv2d has no uniform init :(
            """
            with torch.no_grad():
                if i == 1:
                    torch.nn.init.uniform_(conv, -1 / mid_channels, 
                                                1 / mid_channels)
                else:
                    torch.nn.init.uniform_(conv,  -torch.sqrt(6 / mid_channels) / omega_0, 
                                                torch.sqrt(6 / mid_channels) / omega_0)
            """

            conv = nn.utils.weight_norm(conv)
            bn = nn.BatchNorm2d(c[i])
            
            setattr(self, f'conv{i}', conv)
            setattr(self, f'bn{i}', bn)
            
    def forward(self, x):
        identity = x
        
        for i in range(1,9):
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            
            if i==5:
                x = torch.cat([x, identity], dim=1)
                
            x = conv(x)
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