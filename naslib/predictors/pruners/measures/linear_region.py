import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from . import measure
from pdb import set_trace as bp

class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None
        self.device = torch.cuda.current_device()

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).cuda(device=self.device, non_blocking=True)
        # self.activations[self.ptr:self.ptr+n_batch] = torch.sign(torch.nn.functional.relu(activations))  # after BN
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        # res = torch.einsum('nc,mc->nm', [self.activations, 1-self.activations])
        # res = torch.matmul(self.activations.half(), (1-self.activations).T.half())
        res = torch.matmul(self.activations, (1-self.activations).T)
        res = res + res.T
        res = 1 - torch.sign(res)
        res = res.sum(1)
        res = 1. / res.float()
        self.n_LR = res.sum().item()
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


from operator import mul
from functools import reduce
class Linear_Region_Collector:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs  # BCHW
        self.interFeature = []
        self.hook_handles = [] # store all hooks, remove once finished
        self.device = torch.cuda.current_device()
        self.LRCount = LinearRegionCount(len(inputs))
        if 'micro' in str(type(model)).lower():
            self.relu_range = range(1, 7)
        elif 'macro' in str(type(model)).lower():
            self.relu_range = range(1, 7)
            # self.relu_range = range(self.count_num_relu(self.model))
        elif '101' in str(type(model)).lower():
            self.relu_range = range(1, 12)
        elif '201' in str(type(model)).lower():
            self.relu_range = range(1, 7)
        elif '301' in str(type(model)).lower():
            self.relu_range = range(1, 15)
        else:
            self.relu_range = range(self.count_num_relu(self.model))
        # self.max_channel = int(round(math.log(len(inputs), 2) / len(self.relu_range))) # channels / feature_dim for final activations
        self.max_channel = int(round(math.log(len(inputs), 2))) # channels / feature_dim for final activations
        self.register_hook(self.model)

    def count_num_relu(self, model):
        count_relu = 0
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                count_relu += 1
        return count_relu

    def register_hook(self, model):
        count_relu = 0
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                if self.relu_range is None or count_relu in self.relu_range:
                    handle = m.register_forward_hook(hook=self.hook_in_forward)
                    self.hook_handles.append(handle)
                count_relu += 1

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU
            # self.interFeature.append(output.detach()[:, :self.max_channel])  # for ReLU
            # self.interFeature.append(output.detach().view(len(output), -1)[:, :self.max_channel])  # for ReLU
        else:
            print("hook_in_forward:", input[0].shape)

    def _initialize_weights(self):
        for model in self.models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward_batch_sample(self):
        self.forward(self.model, self.LRCount, self.inputs)
        return self.LRCount.getLinearReginCount()

    def forward(self, model, LRCount, input_data=None):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data.cuda(device=self.device, non_blocking=True))
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            # feature_data = feature_data[:, torch.randperm(feature_data.size()[1])][:, :self.max_channel] # only consider max_channel feature dims
            feature_data = torch.stack([f[torch.randperm(f.size()[0])][:self.max_channel] for f in feature_data], dim=0) # only consider max_channel feature dims
            LRCount.update2D(feature_data)


@measure("linear_region", bn=True)
def compute_lr(net, inputs, targets, split_data=1, loss_fn=None):
    win_s = 4
    if len(inputs.shape) == 5:
        # 12, 9, 3, 64, 64
        B, J, C, H, W = inputs.shape
        inputs = inputs.view(B, J, C, H//win_s, win_s, W//win_s, win_s).permute(0, 3, 5, 1, 2, 4, 6).contiguous().view(-1, J, C, win_s, win_s)
    else:
        B, C, H, W = inputs.shape
        inputs = inputs.view(B, C, H//win_s, win_s, W//win_s, win_s).permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, win_s, win_s)
    lr_collector = Linear_Region_Collector(net, inputs)
    try:
        lr = lr_collector.forward_batch_sample()
    except Exception as e:
        print(e)
        lr = np.nan
    print(lr)
    return lr
