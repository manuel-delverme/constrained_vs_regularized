"""
This script contains some useful functions.
"""

import torch
from torch import nn
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

def build_mlp(d_in, hid_dims, act_fun, act_args):
    """
    This does not add a final linear layer or final sigmoid
    """
    """
    Builds a general MLP block with batch normalization. 
    
    Args:
        in_feat (scalar int): input dimension to the block.
        out_feat (scalar int): output dimension from the block.
        act_fun (torch.nn activation): class for the activation.
        act_args (list): arguments for the constructor of the activation.

    Returns:
        nn.Module list: layers which comprise the block. 
    """
    layers = [torch.nn.Flatten()]
    for d_out in hid_dims:
        layers = layers + [
            nn.Linear(d_in, d_out),
            nn.BatchNorm1d(d_out),
            act_fun(*act_args)
            ] 
        d_in = d_out        
    return layers

def build_convnet(in_channels, im_dim, conv_channels, kernel_size, stride, pool_k_size,
                act_fun, act_args):
    """
    Builds a general Convolutional block with batch normalization for 2D inputs. 
    
    Args:
        in_channels (scalar int): input channels to the block.
        out_channels (scalar int): output channels from the block.
        kernel_size  (scalar int): kernel size of the convolutional layer.
        stride  (scalar int): stride of the convolutional layer.
        pool_k_size  (scalar int): kernel size of the max pooling layer.
        act_fun (torch.nn activation): class for the activation.
        act_args (list): arguments for the constructor of the activation.

    Returns:
        nn.Module list: layers which comprise the block.
    """   
    current_dim = im_dim
    layers = [] 
    for out_channels in conv_channels:
        layers = layers + [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            act_fun(*act_args),
            nn.MaxPool2d(pool_k_size),
            ] 
        in_channels = out_channels
        # keep track of the output channels
        current_dim = get_conv_dim(current_dim, kernel_size, stride, pool_k_size)
    return layers, out_channels, current_dim

def build_rev_convnet(conv_channels, image_channels, kernel_size, stride, 
                    pool_k_size, act_fun, act_args):
    
    layers = []
    in_channels = conv_channels[0]
    for out_channels in conv_channels[1:]:
        layers = layers + [
            nn.Upsample(scale_factor=pool_k_size),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            #nn.BatchNorm2d(out_channels),
            act_fun(*act_args),
            ] 
        in_channels = out_channels
    
    # Last layer, without activation
    layers.append(nn.Upsample(scale_factor=pool_k_size))
    layers.append(
        nn.ConvTranspose2d(in_channels, image_channels, kernel_size, stride)
        )
    return layers

def get_conv_dim(in_dim, kernel_size, stride, pool_k_size):
    """
    Determines the output dimension of a block consisting of a convolutional 
    layer and a max pooling layer. 
    
    Args:
        in_dim (scalar int): input dimension to the block.
        kernel_size  (scalar int): kernel size of the convolutional layer.
        stride  (scalar int): stride of the convolutional layer.
        pool_k_size  (scalar int): kernel size of the max pooling layer.

    Returns:
        scalar int: output dimension of the convolutional block.
    """
    dim_after_conv = (in_dim - kernel_size)/stride + 1
    dim_after_pool = (int(dim_after_conv) - pool_k_size)/pool_k_size + 1 
    return int(dim_after_pool)