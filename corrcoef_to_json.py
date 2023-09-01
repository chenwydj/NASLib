import os
import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
from pdb import set_trace as bp

root = "/mnt/vita-nas/chenwy/NASLib/naslib/data/zc_benchmarks/%s/"%name

result_files = {
    "linear_region": [
    ],
    "ntk_cond": [
    ],
    "ntk_regression": [
    ],
}


SEED = 9000 # TODO

print(root)
for metric, filenames in result_files.items():
    for _filename in filenames:
        with open(os.path.join(root, metric, str(_filename)), 'r') as f:
            data = json.load(f)
        file_components = _filename.split('/')
        predictor = file_components[-2]
        seed = SEED # TODO
        search_space = file_components[-1].split('--')[1]
        dataset = file_components[-1].split('--')[2]
        os.makedirs(os.path.join("run/results/correlation/", search_space, dataset, predictor, str(seed)), exist_ok=True)
        lrs = [item[name]['score'] for item in data if not np.isnan(item[name]['score']) and item[name]['score'] >= 0]
        acc = [item['val_accuracy'] for item in data if not np.isnan(item[name]['score']) and item[name]['score'] >= 0]
        # print(_path, len(data), np.corrcoef(lrs, acc)[0, 1])
        # print(_path, len(data), kendalltau(lrs, acc)[0])
        zc_scores = [{}. {"spearman": spearmanr(lrs, acc)[0]}]
        with open(os.path.join("run/results/correlation/", search_space, dataset, predictor, str(seed), "scores.json"), 'w') as f:
            json.dump(zc_scores, f)
    
