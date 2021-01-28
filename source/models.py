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

# from explore_siren
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features=5, hidden_layers=256, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

class DeepSDF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 512, last_activation = 'tanh', activation = 'relu', omega_0 = 30.):
        super(DeepSDF, self).__init__()
        
        c = [in_channels] + [mid_channels]*3 + [mid_channels - in_channels] + [mid_channels]*3 + [out_channels]
        
        if last_activation == 'tanh':
            self.last_activation = torch.tanh
        elif last_activation is None:
            self.last_activation = lambda x : x

        if activation == 'relu':
            self.activation = lambda x: omega_0*F.relu(x)
        elif activation == 'sin':
            self.activation = lambda x: omega_0*torch.sin(x)
            
        self.dropout = nn.Dropout(0.2)
        
        for i in range(1,9):
            inc = c[i-1] if i!=5 else mid_channels
            ouc = c[i]

            fc = nn.Linear(inc, ouc)

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
                x = self.activation(x)
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