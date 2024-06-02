import torch
from torch.optim import Optimizer, SGD
import numpy as np

def no_quant(x):
    return x

def find_nearest(array, value):
    array = np.asarray(array.flatten())
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class quantizationSGD(Optimizer):
    
    def __init__(self,optimizer, weight_quant= no_quant, grad_quant= no_quant, momentum_quant= no_quant, acc_quant= no_quant, grad_scaling = 1.0):
        super(quantizationSGD, self).__init__(
            optimizer.param_groups, optimizer.defaults
        )
        self.param_groups = optimizer.param_groups
        self.optimizer = optimizer
        self.weight_quant = weight_quant
        self.grad_quant = grad_quant
        self.acc_quant = acc_quant
        self.momentum_quant = momentum_quant
        self.momentum_keys =  ["momentum_buffer"]
        self.grad_scaling = grad_scaling
        self.weight_acc = {}
        for group in self.param_groups:
            for p in group["params"]:
                self.weight_acc[p] = p.detach().clone()

    def step(self, i, closure=None):
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data = self.grad_quant(p.grad.data * self.grad_scaling)
                
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.weight_acc[p].data

        loss = self.optimizer.step()

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.weight_acc[p].data = self.acc_quant(p.data).data

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.weight_quant(p.data).data

        for group in self.param_groups:
            if group["momentum"] == 0:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.optimizer.state[p]
                for key in self.momentum_keys:
                    param_state[key] = self.momentum_quant(param_state[key])
        
        return loss
    
