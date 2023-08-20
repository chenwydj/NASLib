import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from pdb import set_trace as bp


def get_samplewise_grad(network, inputs, train_mode=True):
    device = torch.cuda.current_device()
    if train_mode:
        network.train()
    else:
        network.eval()
    grads_x = []

    inputs = inputs.cuda(device=device, non_blocking=True)
    # targets = targets.cuda(device=device, non_blocking=True)
    # targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    # targets_x_onehot_mean = targets_onehot - targets_onehot.mean(0)
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
    grads_x = torch.stack(grads_x, 0)
    return grads_x


def get_ntk(network, inputs, train_mode=True):
    # device = torch.cuda.current_device()
    if train_mode:
        network.train()
    else:
        network.eval()

    # targets = targets.cuda(device=device, non_blocking=True)
    # targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    # targets_x_onehot_mean = targets_onehot - targets_onehot.mean(0)
    # NTK cond
    grads_x = get_samplewise_grad(network, inputs, train_mode=train_mode)
    ntk = torch.einsum('nc,mc->nm', [grads_x, grads_x])
    eigenvalues, _ = torch.linalg.eigh(ntk)  # ascending
    cond_x = eigenvalues[-1] / eigenvalues[0]
    if torch.isnan(cond_x):
        # cond_x = -1 # bad gradients
        cond_x = np.nan # bad gradients
    else:
        cond_x = cond_x.item()
    return cond_x


def get_ntk_regression(network, inputs, targets, inputs_val, targets_val, train_mode=True, num_classes=100):
    device = torch.cuda.current_device()
    if train_mode:
        network.train()
    else:
        network.eval()
    prediction_mse = None

    targets = targets.cuda(device=device, non_blocking=True)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets_x_onehot_mean = targets_onehot - targets_onehot.mean(0)
    grads_x = get_samplewise_grad(network, inputs, train_mode=train_mode)
    ntk = torch.einsum('nc,mc->nm', [grads_x, grads_x])
    # Val / Test set
    grads_y = get_samplewise_grad(network, inputs_val, train_mode=train_mode)
    targets_val = targets_val.cuda(device=device, non_blocking=True)
    targets_val_onehot = torch.nn.functional.one_hot(targets_val, num_classes=num_classes).float()
    targets_y_onehot_mean = targets_val_onehot - targets_val_onehot.mean(0)
    if grads_y.sum() == 0 or grads_x.sum() == 0:
        return np.nan
    try:
        _ntk_yx = torch.einsum('nc,mc->nm', [grads_y, grads_x])
        PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk), targets_x_onehot_mean)
        prediction_mse = ((PY - targets_y_onehot_mean)**2).sum(1).mean(0).item()
    except RuntimeError:
        # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
        # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
        prediction_mse = np.nan # bad gradients
    return prediction_mse
