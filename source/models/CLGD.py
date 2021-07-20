import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.autograd import Variable
import numpy as np
import knn_cuda

from models.models import DeepSDFDecoder

import numpy as np
from knn import knn
import utils

class CLGD(nn.Module):
    """
    X : (b, 1, n, 3)
    H : (b, 1, 1, 256)
    P : (b, m, 1, 6)
    G : (b, 1, n, 256)
    """

    def __init__(self):
        super(CLGD, self).__init__()

        self._sdf = DeepSDFDecoder(latent_size=256, dims=[512]*8, latent_in=[4], use_tanh=True)

        self._r_layers = nn.Sequential(nn.Linear(256*2 + 16, 256, bias=False), nn.ReLU())
        self._z_layers = nn.Sequential(nn.Linear(256*2 + 16, 256, bias=False), nn.ReLU())

        self._g_layers = nn.Sequential(nn.Linear(256*2 + 16, 256, bias=False), nn.ReLU())

        self._dx_layers = nn.Sequential(nn.Linear(256*2 + 16, 1, bias=False), nn.ReLU())
        self._dh_layers = nn.Sequential(nn.Linear(256*2 + 16, 1, bias=False), nn.ReLU())
        self._lc_layers = nn.Sequential(nn.Linear(256*2 + 16, 1, bias=False), nn.ReLU())
        self._lp_layers = nn.Sequential(nn.Linear(256*3 + 13, 1, bias=False), nn.ReLU())


    def expand(self, X, dim, target):
        targets = [-1]*3

        for d, t in zip(dim, target):
            targets[d+3] = t
            
        return X.expand(*([-1]*(len(X.shape)-3)), *[targets[i] for i in range(3)])

    def expand_all(self, X, H, P, s, dsx, dsh, G):
        m = P.shape[-3]
        n = X.shape[-2]
        X_expand = self.expand(X, [-3], [m])
        H_expand = self.expand(H, [-3, -2], [m, n])
        P_expand = self.expand(P, [-2], [n])
        
        s_expand = self.expand(s, [-3], [m])
        dsx_expand = self.expand(dsx, [-3], [m])
        dsh_expand = self.expand(dsx, [-3, -2], [m, n])

        G_expand = self.expand(G, [-3], [m])

        return [X_expand, H_expand, P_expand, s_expand, dsx_expand, dsh_expand, G_expand]


    def sdf(self, X, H, return_grad=True):
        H_expand = self.expand(H, [-2], [X.shape[-2]])

        # deepsdf decoder requires [h, x] input
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
                                        retain_graph=True)
            dX = dX.view(X.shape).requires_grad_(True)
            dH = dH.view(H.shape).requires_grad_(True)

            return _output, dX, dH
        else:

            return _output

    def gru(self, X, H, P, s, dsx, dsh, G):
        m = P.shape[-3]
        n = X.shape[-2]

        # expand every input & concat
        expands = self.expand_all(X,H,P,s,dsx,dsh,G)

        _input = torch.cat(expands, dim=-1)

        r = self._r_layers(_input)                          # coefficient of old G into new G network
        z = self._z_layers(_input)                          # interpolation variable (old G : new G)

        G_expand = expands[-1]
        _input = torch.cat([_input[..., :-G.shape[-1]], 
                            r * G_expand], dim=-1)

        new_G = self._g_layers(_input)
        new_G = torch.max(new_G, dim=-3, keepdim=True)[0]      # pooling w.r.t cam dimension

        G = z * new_G + (1-z) * G
        G_expand = self.expand(G, [-3], [m])

        return G

        
    def forward(self, X, H, P, G):
        
        s, dsx, dsh = self.sdf(X, H, return_grad=True)
        G = self.gru(X, H, P, s, dsx, dsh, G)

        expands = self.expand_all(X,H,P,s,dsx,dsh,G)

        _input = torch.cat(expands, dim=-1)

        dx = self._dx_layers(_input)
        dh = self._dh_layers(_input)
        lc = self._lc_layers(_input)
        lp = self._lp_layers(_input)

        dx = torch.max(dx, dim=-3, keepdim=True)[0]
        dh = torch.max(dx, dim=-2, keepdim=True)[0]
        dh = torch.max(dx, dim=-3, keepdim=True)[0]
        lc = torch.max(lc, dim=-2, keepdim=True)[0]

        # s, dsx, dsh : output from SDF
        # dx, dh,  lc, lp : output from 
        return [s, dsx, dsh, G,  dx, dh,  lc, lp] 

    def step(self, X, H, P, G):
        forward_inputs = self(X, H, P, G)
        [s, dsx, dsh,  G,  dx, dh,  lc, lp] = forward_inputs

        X = X + dx * dsx
        H = H + dh * dsh

        return X, H, G, forward_inputs

    def backward(self, X, H, P, G, steps=5):
        for i in range(steps):
            X, H, G, forward_inputs = self.step(X,H,P,G)

        [s, dsx, dsh,  G,  dx, dh,  lc, lp] = forward_inputs
        s, dsx_, dsh_ = self.sdf(X, H, return_grad=True)


        L1, L2, L3, L4 = self.total_loss(X,H, P, s, dsx, lc, lp)
        L = L1 + L2 + L3 + L4

        L_grad_targets = [s, dsx, dsh, dx, dh, lc, lp]

        L_grads = torch.autograd.grad(L, 
                                      L_grad_targets, 
                                      grad_outputs=torch.ones_like(L), 
                                      create_graph=False,
                                      retain_graph=True,
                                      allow_unused=True)

        for i, (t, g) in enumerate(zip(L_grad_targets, L_grads)):
            if g is not None:
                if i >= 5:
                    t.backward(- g, retain_graph=True)
                else:
                    t.backward(g, retain_graph=True)

        return X, dsx_

        


    def total_loss(self, X,H,P,s,dsx, lc, lp):
        L1 = self.projection_loss(X, P, lc)     # weighted reprojection loss

        L2 = self.grad_loss(dsx, lp)            # |dsx| = 1 loss

        L3 = self.sdf_loss(X, P, s)          # BCE(s, repr(X,P))

        L4 = self.H_loss(H)                     # regularization on |H|

        return L1, L2, L3, L4

    
    def projection_loss(self, X, P, lc):
        Y = self.Y

        lc = torch.sqrt(lc)

        x = utils.project(X, P) * lc
        y = utils.project(Y, P) * lc

        return self.find_nearest_correspondences_dist(x,y)

    
    def grad_loss(self, dsx, lp):
        return torch.pow(torch.sqrt(torch.pow(dsx, 2).sum(dim=-1, keepdim=True)) - 1, 2) * lp

    def sdf_loss(self, X, P, s):
        shp = s.shape
        bce = nn.BCELoss(reduce=False)

        Y = self.Y
        x = utils.project(X, P)
        y = utils.project(Y, P)

        with torch.no_grad():
            true_s2 = self.find_nearest_correspondences_dist(x,y) / P.shape[-3]
            true_s2 = torch.clamp(true_s2, 0, 1)

        return bce(torch.pow(s,2), true_s2)

    def H_loss(self, H):
        return torch.pow(H, 2).sum(dim=-1, keepdim=True)



    def find_nearest_correspondences_dist(self, x_hat, x, k=1):
        assert x_hat.shape[-1] == x.shape[-1]
        assert x_hat.shape[-3] == x.shape[-3]
        shp1 = x_hat.shape
        shp2 = x.shape

        # ((batches), m, n, 2)
        x_hat = x_hat.view(-1, shp1[-3], shp1[-2], shp1[-1])
        x = x.view(-1, shp2[-3], shp2[-2], shp2[-1])

        dists = []
        for x_hat_, x_ in zip(x_hat, x):
            # ((batches), n, m*2)
            x_hat_ = x_hat_.transpose(0,1).reshape(shp1[-2], -1)    
            x_ = x_.transpose(0,1).reshape(shp2[-2], -1)

            knn_f = knn.apply
            dist_  = knn_f(x_hat_,x_,k)
            dists.append(dist_.mean(dim=1).unsqueeze(0))

        # ((batches), n)
        dist = torch.cat(dists).view(*shp1[:-3], x_hat.shape[-2]).unsqueeze(-1).unsqueeze(-3)

        return dist
    