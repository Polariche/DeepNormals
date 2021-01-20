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
            
            k = group['param_features']
            hs_features = group['hs_features']

            params_gcat = torch.zeros((0,k+hs))

            # concat param gradients
            for param in group['params']:
                state = self.state[param]
                hs = state['hs']

                if param.grad is not None:
                    p = param.grad.view(1,k)
                    p = torch.cat([p, hs], dim=1)       # add hidden state
                    params_gcat = torch.cat([params_gcat, p], dim=0)
                else:
                    params_gcat = torch.cat([params_gcat, torch.zeros((1,k+hs_features))], dim=0)

            # pass gradients to network
            params_gcat = module(params_gcat)

            # update params
            for i, param in enumerate(group['params']):
                state = self.state[param]
                param.add(params_gcat[i, :k].view(param.shape), alpha=-1)

                state['hs'] = params_gcat[i, k:]
        
        return loss


    def add_param_group(self, param_group):
        super(LGD, self).add_param_group(param_group)

        # count the total number of variables to optimize
        # all parameters should contain same variable count
        k = torch.flatten(param_group['params'][0]).shape[0]
        for param in param_group['params']:
            k_ = torch.flatten(param).shape[0]
            assert k == k_

        param_group['param_features'] = k

        # create a network
        mc = param_group['module_class']
        module = mc(in_features=k, out_features=k, *param_group['module_args'])
        param_group.setdefault('module', module)

        # add a hidden state for each param
        hs_features = param_group['hs_features']
        for param in param_group['params']:
            self.state[param].setdefault('hs', torch.zeros_like((1,hs_features)))