import os
import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
from pdb import set_trace as bp

#####################
# name1 = "linear_region/" + "benchmark--transbench101_micro--autoencoder--0-12.json"
# name2 = "ntk_cond/" + "benchmark--transbench101_micro--autoencoder--0-64.json"
# name3 = "ntk_regression/" + "benchmark--transbench101_micro--autoencoder--0-32.json"

# 4 ####################
name1 = "linear_region/" + "benchmark--transbench101_macro--jigsaw--0-128.json"
name2 = "ntk_cond/" + "benchmark--transbench101_macro--jigsaw--0-128.json"
name3 = "ntk_regression/" + "benchmark--transbench101_macro--jigsaw--0-128.json"

# 5 ####################
# name1 = "linear_region/" + "benchmark--nasbench101--cifar10--0-256.json"
# name2 = "ntk_cond/" + "benchmark--nasbench101--cifar10--0-64.json"
# name3 = "ntk_regression/" + "benchmark--nasbench101--cifar10--0.json"

# 6 ####################
# name1 = "linear_region/" + "benchmark--transbench101_macro--autoencoder--0-12.json"
# name2 = "ntk_cond/" + "benchmark--transbench101_macro--autoencoder--0-64.json"
# name3 = "ntk_regression/" + "benchmark--transbench101_macro--autoencoder--0-32.json"

# 7 ####################
# name1 = "linear_region/" + "benchmark--transbench101_macro--class_scene--0-12.json"
# name2 = "ntk_cond/" + "benchmark--transbench101_macro--class_scene--0-128.json"
# name3 = "ntk_regression/" + "benchmark--transbench101_macro--class_scene--0-128.json"

# 8 ####################
#name1 = "linear_region/" + "benchmark--nasbench301--cifar10--0-128.json"
#name2 = "ntk_cond/" + "benchmark--nasbench301--cifar10--0-64.json"
#name3 = "ntk_regression/" + "benchmark--nasbench301--cifar10--0-64.json"

# 12 ####################
# name1 = "linear_region/" + "benchmark--transbench101_micro--jigsaw--0-128.json"
# name2 = "ntk_cond/" + "benchmark--transbench101_micro--jigsaw--0-128.json"
# name3 = "ntk_regression/" + "benchmark--transbench101_micro--jigsaw--0-128.json"

# 14 ####################
# name1 = "linear_region/" + "benchmark--transbench101_micro--class_object--0-12.json"
# name2 = "ntk_cond/" + "benchmark--transbench101_micro--class_object--0-64.json"
# name3 = "ntk_regression/" + "benchmark--transbench101_micro--class_object--0-192.json"

# 16 ####################
# name1 = "linear_region/" + "benchmark--nasbench201--ImageNet16-120--0-128.json"
# name2 = "ntk_cond/" + "benchmark--nasbench201--ImageNet16-120--0-64.json"
# name3 = "ntk_regression/" + "benchmark--nasbench201--ImageNet16-120--0-64.json"

# 17 ####################
# name1 = "linear_region/" + "benchmark--transbench101_micro--class_scene--0-12.json"
# name2 = "ntk_cond/" + "benchmark--transbench101_micro--class_scene--0-128.json"
# name3 = "ntk_regression/" + "benchmark--transbench101_micro--class_scene--0-128.json"

# 18 ####################
# name1 = "linear_region/" + "benchmark--nasbench201--cifar10--0-128.json"
# name2 = "ntk_cond/" + "benchmark--nasbench201--cifar10--0-64.json"
# name3 = "ntk_regression/" + "benchmark--nasbench201--cifar10--0-64.json"

# 19 ####################
# name1 = "linear_region/" + "benchmark--nasbench201--cifar100--0-128.json"
# name2 = "ntk_cond/" + "benchmark--nasbench201--cifar100--0-64.json"
# name3 = "ntk_regression/" + "benchmark--nasbench201--cifar100--0-64.json"

root = "/mnt/vita-nas/chenwy/NASLib/naslib/data/zc_benchmarks/"

values = []
accuracies = []

data_all = []
for _name in [name1, name2, name3]:
    with open(os.path.join(root, _name), 'r') as f:
        data = json.load(f)
        print(_name, len(data))
        data_all.append(data)

for v1, v2, v3 in zip(*data_all):
    _v1 = v1['linear_region']['score']
    _v2 = v2['ntk_cond']['score']
    _v3 = v3['ntk_regression']['score']
    if np.isnan(_v1) or np.isnan(_v2) or np.isnan(_v3): continue
    values.append([_v1, 1./_v2, 1./_v3])
    # values.append([_v1, _v2, _v3])
    accuracies.append(v1['val_accuracy'])

print(len(accuracies), spearmanr(np.array(values)[:, 0], accuracies)[0])
print(len(accuracies), spearmanr(1./np.array(values)[:, 1], accuracies)[0])
print(len(accuracies), spearmanr(1./np.array(values)[:, 2], accuracies)[0])

# print(_path, len(data), np.corrcoef(lrs, acc)[0, 1])
print(len(accuracies), spearmanr(np.prod(values, 1), accuracies)[0])
# print(_path, len(data), kendalltau(lrs, acc)[0])
