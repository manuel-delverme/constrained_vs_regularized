"""
This script contains some useful functions.
"""

import torch
import numpy as np
import pandas as pd

def get_class_dict(batch_y, classes):
    """
    Produces a dictionary, where the keys are the different classes samples
    can take, and the values correspond to the indices of elements corresponding
    to that specific class. 
    
    Args:
        batch_y (1D array-like): class labels of a batch.
        classes (list): possible values an element of batch_y can take.

    Returns:
        dictionary: contains the indices of batch_y which correspond to
        each one of classes. 
    """
    return {cl:torch.where(batch_y == cl)[0] for cl in classes}

def get_ordinal_labels(y, cl_idx):
    """
    Transforms string labels into ordinal labels.
    
    Args:
        y (1D array-like): labels
        cl_idx (list of indices): indices present in position i of this
        list are converted into ordinal element "i".

    Returns:
        list: [description]
    """
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
    # Produces the mean of a tensor, ignoring zero-valued elements
    tensor = tensor[tensor>0]
    tensor = tensor.detach().cpu().numpy()
    return np.nanmean(tensor)

def load_logs(path):
    # Loads logs saved as CSV
    df = pd.read_csv(path)
    df = df.set_index('step')
    return df
