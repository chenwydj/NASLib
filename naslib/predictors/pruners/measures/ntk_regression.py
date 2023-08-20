import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

from . import measure
from ._ntk import get_ntk_regression
from pdb import set_trace as bp


@measure("ntk_regression", bn=True)
def compute_ntk_regression(net, inputs, targets, split_data=1, loss_fn=None):
    train_val_split = 0.8
    N = len(inputs)
    try:
        ntk = get_ntk_regression(net, inputs[:int(N*train_val_split)], targets[:int(N*train_val_split)], inputs[int(N*train_val_split):], targets[int(N*train_val_split):], train_mode=True, num_classes=net.num_classes)
    except Exception as e:
        print("error:", e)
        ntk = np.nan
    print(ntk)
    return ntk
