import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings

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
                # if module's requires_grad = True, dL/dp (p is module's parameter) is be also evaluated,
                # which is accumulated on module's grad over steps.
                # we can run another optimizer to optimize... the optimizer?

                loss = closure()

        for group in self.param_groups:
            module = group['module']
            state = self.state[group]
            hs_features = group['hs_features']
            hs = state['hs']

            params_gcat = torch.zeros((1,0))

            # concat param gradients
            for param in group['params']:
                if param.grad is not None:
                    params_gcat = torch.cat([params_gcat, torch.flatten(param.grad).view(1,-1)], dim=1)
                else:
                    params_gcat = torch.cat([params_gcat, torch.flatten(torch.zeros_like(param)).view(1,-1)], dim=1)

            # add hidden state
            params_gcat = torch.cat([params_gcat, hs], dim=1)

            # pass gradients to network
            params_gcat = module(params_gcat)

            # update params
            i = 0
            for param in group['params']:
                n = torch.flatten(param).shape[0]
                param_g = params_gcat[i:i+n].view(param.shape)
                param.add(param_g, alpha=-1)

                i += n

            # update hidden state
            if hs_features > 0:
                state['hs'] = params_gcat[i:]
        
        return loss


    def add_param_group(self, param_group):
        super(LGD, self).add_param_group(param_group)

        # count the total number of variables to optimize
        n = 0
        for param in param_group['params']:
            n += torch.flatten(param).shape[0]

        # create a network
        mc = param_group['module_class']
        module = mc(in_features=n, out_features=n, *param_group['module_args'])
        param_group.setdefault('module', module)

        # add a hidden state for the group
        hs_features = param_group['hs_features']
        self.state[param_group] = {'hs': torch.zeros((1, hs_features))}