import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.autograd import Variable
import numpy as np
import knn_cuda

from models import DeepSDFDecoder

import numpy as np

class CLGD(nn.Module):
    """
    h : (b, 1, 1, ?)
    X : (b, 1, n, 3)
    P : (b, m, 1, 5)
    u : (b, m, n, 2)
    """

    def __init__(self):
        self._sdf = DeepSDFDecoder(latent_size=256, dims=[512]*8, latent_in=(4))

        pass

    def sdf(self, X, H, return_grad=True):
        H_expand = H.expand([-1]*(len(H.shape)-3), -1, X.shape[-2], -1)

        # deepsdf decoder requires (h, x) input
        _input = torch.cat([H_expand, X], dim=-1)
        _output = self._sdf(_input)

        if return_grad:
            assert X.requires_grad is True
            assert H.requires_grad is True

            _output_sum = _output.sum()
            dX, dH = torch.autograd.grad(_output_sum, 
                                        [X, H], 
                                        grad_outputs=torch.ones_like(_output_sum), 
                                        create_graph=False,
                                        retain_graph=True)[0]
            dX = dX.view(X.shape)
            dH = dH.view(H.shape)

            return _output, dX, dH
        else:

            return _output

    def forward(self, X, P, H):
        
        s, dsx, dsh = self.sdf(X, H, return_grad=True)


        # s, dsx, dsh : output from SDF
        # dx, dh,  lam, dp : output from 
        return s, dsx, dsh,  dx, dh,  lam, dp 

    def step(self, X, P, H):
        s, dsx, dsh,  dx, dh,  lam, dp = self(X, P, H)

        X + dx * dsx
        H + dh * dsh