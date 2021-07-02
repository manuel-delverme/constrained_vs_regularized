import torch
import numpy as np
import pandas as pd

def get_class_dict(batch_y, classes):
    return {cl:torch.where(batch_y == cl)[0] for cl in classes}

def get_ordinal_labels(y, cl_idx):
    y = torch.zeros_like(y)
    for i,idx in enumerate(cl_idx):
        y[idx] = i
    return y

def safe_log10(x):
    return torch.log10(x + 1e-10)

def safe_log(x):
    return torch.log(x + 1e-10)

def tensor2dict(tensor, tensor_name):
    # Converts 1d tensor to dictionary. Used for logging
    return {tensor_name + '_' + str(i+1): val for i,val in enumerate(tensor)}

def reduce_fx(tensor):
    tensor = tensor[tensor>0]
    tensor = tensor.detach().cpu().numpy()
    return np.nanmean(tensor)

def load_logs(path):
    df = pd.read_csv(path)
    df = df.set_index('step')
    return df
