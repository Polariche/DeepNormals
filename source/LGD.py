import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings

from collections import defaultdict

class LGD(optim.Optimizer):
    def __init__(self, params, module_class, module_args, hs_features=0):
        defaults = dict(module_class=module_class, module_args=module_args, hs_features=hs_features)
        super(LGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LGD, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                # if module's requires_grad = True, dL/dp (p is module's parameter) is also evaluated,
                # which is accumulated on module's grad over steps.
                # we can run another optimizer to optimize... the optimizer?

                loss = closure()

        for group in self.param_groups:
            for param in group['params']:

                module = group['module'][param]
                k = group['features'][param]

                p = param.view(-1,k)
                if p.grad != None:
                    grad = p.grad
                else:
                    grad = torch.zeros_like(p)

                grad = module(grad)
                param = param.add(grad.view(param.shape))
        
        return loss


    def add_param_group(self, param_group):
        super(LGD, self).add_param_group(param_group)

        # count the total number of variables to optimize
        # all parameters should contain same variable count

        mc = param_group['module_class']

        param_group.setdefault('modules', defaultdict(mc))
        param_group.setdefault('features', defaultdict(int))

        for param in param_group['params']:
            if len(param.shape) < 2:
                k = 1
            elif len(param.shape) == 2:
                k = param.shape[1]
            else:
                k = param.view(param.shape[0], -1).shape[-1]

            # create a network
            module = mc(in_features=k, out_features=k, *param_group['module_args'])

            param_group['modules'][param] = module
            param_group['features'][param] = k


    def parameters(self):
        module_params = []
        for group in self.param_groups:
            for param in group['params']:
                module_params += list(group['modules'][param].parameters())
        return module_params