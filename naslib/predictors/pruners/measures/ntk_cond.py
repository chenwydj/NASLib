import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from . import measure
from ._ntk import get_ntk
from pdb import set_trace as bp


@measure("ntk_cond", bn=True)
def compute_ntk_cond(net, inputs, targets, split_data=1, loss_fn=None):
    try:
        ntk = get_ntk(net, inputs, train_mode=True)
    except Exception as e:
        print("error:", e)
        ntk = np.nan
    print(ntk)
    return ntk
