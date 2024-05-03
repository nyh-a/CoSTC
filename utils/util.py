import json
import torch
import pandas as pd
import networkx as nx
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import model.metric as module_metric


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node, 3)
    return [node for node, length in path_lengths.items()
                    if length == n]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def calc_metrics(all_ranks, leaf_ranks, nonleaf_ranks):
    print("Calculate metrics")
    metrics = [
            "micro_mr",
            "macro_mr",
            "mrr_scaled_10",
            "mrr",
            
            "hit_at_1",
            "hit_at_5",
            "hit_at_10",
            "hit_at_50",
            "hit_at_100",

            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "recall_at_50",
            "recall_at_100",

            "precision_at_1",
            "precision_at_5",
            "precision_at_10"
            
        ]

    metric_ftns = [getattr(module_metric, met) for met in metrics]
    
    total_metrics = {metric.__name__ : metric(all_ranks) for metric in metric_ftns}
    leaf_metrics = {f'leaf_{metric.__name__}' : metric(leaf_ranks) for metric in metric_ftns}
    non_leaf_metrics = {f'nonleaf_{metric.__name__}' : metric(nonleaf_ranks) for metric in metric_ftns}


    for key, value in total_metrics.items():
        print('    {:15s}: {}'.format(key, value))
    for key, value in leaf_metrics.items():
        print('    {:15s}: {}'.format(key, value))
    for key, value in non_leaf_metrics.items():
        print('    {:15s}: {}'.format(key, value))


    return total_metrics, leaf_metrics, non_leaf_metrics