import torch
from torch.optim import Optimizer, SGD

def no_quant(x):
    return x


class quantizationSGD(Optimizer):
    
    def __init__(self,optimizer, quant_func= None):
        super(quantizationSGD, self).__init__(
            optimizer.param_groups, optimizer.defaults
        )
        self.param_groups = optimizer.param_groups
        self.optimizer = optimizer
        self.quant_func = quant_func
        self.momentum_keys =  ["momentum_buffer"]

        self.weight_acc = {}
        for group in self.param_groups:
            for p in group["params"]:
                self.weight_acc[p] = p.detach().clone()

    def step(self, closure=None):
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data = self.quant_func(p.grad.data)

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.weight_acc[p].data

        loss = self.optimizer.step()

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.weight_acc[p].data = self.quant_func(p.data).data

        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.quant_func(p.data).data

        for group in self.param_groups:
            if group["momentum"] == 0:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.optimizer.state[p]
                for key in self.momentum_keys:
                    param_state[key] = self.quant_func(param_state[key])
        
        return loss
    
