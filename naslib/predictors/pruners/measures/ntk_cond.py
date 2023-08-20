import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from . import measure
from ._ntk import get_ntk
from pdb import set_trace as bp


def _get_ntk(network, inputs, targets, inputs_val=None, targets_val=None, train_mode=True, num_classes=100):
    device = torch.cuda.current_device()
    if train_mode:
        network.train()
    else:
        network.eval()
    grads_x = []
    cellgrads_y = []
    prediction_mse = None

    inputs = inputs.cuda(device=device, non_blocking=True)
    targets = targets.cuda(device=device, non_blocking=True)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets_x_onehot_mean = targets_onehot - targets_onehot.mean(0)
    network.zero_grad()
    logit = network(inputs)
    for _idx in range(len(inputs)):
        logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
        grad = []
        for name, W in network.named_parameters():
            if 'weight' in name and W.grad is not None:
                grad.append(W.grad.view(-1).detach())
        grads_x.append(torch.cat(grad, -1))
        network.zero_grad()
        torch.cuda.empty_cache()
    # NTK cond
    grads_x = torch.stack(grads_x, 0)
    ntk = torch.einsum('nc,mc->nm', [grads_x, grads_x])
    eigenvalues, _ = torch.linalg.eigh(ntk)  # ascending
    cond_x = eigenvalues[-1] / eigenvalues[0]
    if torch.isnan(cond_x):
        # cond_x = -1 # bad gradients
        cond_x = np.nan # bad gradients
    else:
        cond_x = cond_x.item()
    # Val / Test set
    if inputs_val:
        inputs_val = inputs_val.cuda(device=device, non_blocking=True)
        targets_val = targets_val.cuda(device=device, non_blocking=True)
        targets_val_onehot = torch.nn.functional.one_hot(targets_val, num_classes=num_classes).float()
        targets_y_onehot_mean = targets_val_onehot - targets_val_onehot.mean(0)
        network.zero_grad()
        logit = network(inputs_val)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        for _idx in range(len(inputs_val)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            cellgrad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None and "cell" in name:
                    cellgrad.append(W.grad.view(-1).detach())
            cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
            if len(cellgrads_y) == 0:
                cellgrads_y = [cellgrad]
            else:
                cellgrads_y.append(cellgrad)
            network.zero_grad()
            torch.cuda.empty_cache()
        cellgrads_y = torch.stack(cellgrads_y, 0)
        try:
            _ntk_yx = torch.einsum('nc,mc->nm', [cellgrads_y, grads_x])
            PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk), targets_x_onehot_mean)
            prediction_mse = ((PY - targets_y_onehot_mean)**2).sum(1).mean(0).item()
        except RuntimeError:
            # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
            # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
            prediction_mse = np.nan # bad gradients
    ######
    if prediction_mse is None:
        return cond_x
    else:
        return cond_x, prediction_mse


@measure("ntk_cond", bn=True)
def compute_ntk_cond(net, inputs, targets, split_data=1, loss_fn=None):
    try:
        ntk = get_ntk(net, inputs, targets, train_mode=True)
    except Exception as e:
        print("error:", e)
        bp()
        ntk = np.nan
    print(ntk)
    return ntk
