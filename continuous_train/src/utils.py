import json
import random
import numpy as np
import torch

def convert_ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(elem) for elem in data]
    else:
        return data
def convert_list_to_ndarray(data):
    if isinstance(data, list):
        try:
            return np.array(data)
        except ValueError:
            # If conversion to numpy array fails, recursively convert elements
            return [convert_list_to_ndarray(elem) for elem in data]
    elif isinstance(data, dict):
        return {k: convert_list_to_ndarray(v) for k, v in data.items()}
    else:
        return data
def seed_func(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
def moving_average(values, window):
    return np.mean(values[-window:]) if len(values) >= window else np.mean(values)
