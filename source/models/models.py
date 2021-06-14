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

from torch.nn import Conv2D, Linear

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
    def __init__(self, in_features, out_features, hidden_features=5, hidden_layers=256, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30., bias=True):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0, bias=bias))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0, bias=bias))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features, bias=bias)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0, bias=bias))
        
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

# basecode from https://github.com/facebookresearch/DeepSDF/blob/master/networks/deep_sdf_decoder.py
class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

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
    
    


def Conv2d_relu(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return nn.Sequential(Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, dilation=dilation, 
                            groups=groups, bias=bias, padding_mode=padding_mode), 
                        nn.ReLU())

def Linear_relu(in_channels, out_channels, bias=True):
    return nn.Sequential(Linear(in_channels, out_channels, bias=bias),
                        nn.ReLU())



class PointSetGenerator(nn.Module):
    def __init__(self):
        super(PointSetGenerator, self).__init__()

        # 192 256
        self.block0 = nn.Sequential(Conv2d_relu(4, 16, (3,3), stride=1),
                                    Conv2d_relu(16, 16, (3,3), stride=1),
                                    Conv2d_relu(16, 32, (3,3), stride=2))

        # 96 128
        self.block1 = nn.Sequential(Conv2d_relu(32, 32, (3,3), stride=1),
                                    Conv2d_relu(32, 32, (3,3), stride=1),
                                    Conv2d_relu(32, 64, (3,3), stride=2))

        #48 64
        self.block2 = nn.Sequential(Conv2d_relu(64, 64, (3,3), stride=1),
                                    Conv2d_relu(64, 64, (3,3), stride=1),
                                    Conv2d_relu(64, 128, (3,3), stride=2))

        #24 32
        self.block3 = nn.Sequential(Conv2d_relu(128, 128, (3,3), stride=1),
                                    Conv2d_relu(128, 128, (3,3), stride=1))
        self.block3_cont = Conv2d_relu(128, 256, (3,3), stride=2))

        #12 16
        self.block4 = nn.Sequential(Conv2d_relu(256, 256, (3,3), stride=1),
                                    Conv2d_relu(256, 256, (3,3), stride=1))
        self.block4_cont = Conv2d_relu(256, 512, (3,3), stride=2)

        #6 8
        self.block5 = nn.Sequential(Conv2d_relu(512, 512, (3,3), stride=1),
                                    Conv2d_relu(512, 512, (3,3), stride=1),
                                    Conv2d_relu(512, 512, (3,3), stride=1))

        self.block5_cont = Conv2d_relu(512, 512, (5,5), stride=2, padding=2)

        self.block_additional = nn.Sequential(Linear_relu(512*12, 2048),
                                            Linear_relu(2048, 1024),
                                            Linear(1024, 256*3))



        self.trans5 = ConvTranspose2d(512, 256, (5,5), stride=2, padding=(2, 2))
        self.conv5 = Conv2d(512, 256, (3,3), stride=1)
        # relu(x+x5)

        self.trans4 = nn.Sequential(Conv2d(256, 256, (3,3), stride=1),
                                    ConvTranspose2d(256, 128, (5,5), stride=2, padding=(2, 2)))
        self.conv4 = Conv2d(256, 128, (3,3), stride=1)
        # relu(x+x4)

        self.trans3 = nn.Sequential(Conv2d(128, 128, (3,3), stride=1),
                                    ConvTranspose2d(128, 64, (5,5), stride=2, padding=(2, 2)))
        self.conv3 = Conv2d(128, 64, (3,3), stride=1)
        # relu(x+x3)

        self.final = nn.Sequential(Conv2d(64, 64, (3,3), stride=1),
                                    Conv2d(64, 64, (3,3), stride=1),
                                    Conv2d(64, 3, (3,3), stride=1))


    
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)

        x3 = self.block3(x)
        x = self.block3_cont(x3)

        x4 = self.block4(x)
        x = self.block4_cont(x4)

        x5 = self.block5(x)
        x = self.block5_cont(x5)

        # is it the right way to do this??
        x_additional = self.block_additional(x.reshape(-1, 512*12))
        x_additional = x_additional.reshape(-1, 256, 3)

        x = self.trans5(x)
        x5 = self.conv5(x5)
        x = F.relu(x + x5)

        x = self.trans4(x)
        x4 = self.conv4(x4)
        x = F.relu(x + x4)

        x = self.trans3(x)
        x3 = self.conv3(x3)
        x = F.relu(x + x3)

        x = self.final(x)
        x = x.reshape(-1, 32*24, 3)
        x = torch.cat([x_additional, x], axis=1)
        x = x.reshape(-1, 1024, 3)

        return x

        

    
