#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:13:15 2021

@author: juan
"""

import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset

##################################
# Toy Data
##################################

def gen_mog_data(n_samples, props, classes, d=2, 
                mus=None, Ls=None, shuffle=True):
    
    n_gaussians = len(props)
    
    # Compute number of samples per class
    n_per_class = [int(n_samples * p) for p in props[:-1]]
    n_per_class.append(n_samples - sum(n_per_class)) # not lose samples in round
    
    samples = {'x': [], 'y': []}
    
    for gaussian_id in range(n_gaussians):
        # Generate input data by mu + x @ sqrt(cov)
        L = (1/d) * torch.eye(d) if (Ls is None) else Ls[gaussian_id]
        mu = torch.randn(d) if (mus is None) else mus[gaussian_id]
        x = mu + torch.randn(n_per_class[gaussian_id], d) @ L
                
        # Generate labels
        y = torch.tensor(n_per_class[gaussian_id]*[classes[gaussian_id]])
        
        samples['x'].append(x)
        samples['y'].append(y)
    
    x, y = torch.cat(samples['x']), torch.cat(samples['y'])
    
    if shuffle:
        idx = torch.randperm(n_samples)
        x, y = x[idx, ...], y[idx, ...]
    
    return {'x': x, 'y': y}

def add_noisy_labels(data, prob=0.01, affected_class = 1, fake_label = 0):
    
    mask = data['y'] == 1
    n_in_class = mask.sum().item()
    
    is_noisy = torch.rand(n_in_class) > 1 - prob   
    data['y'][mask] = torch.LongTensor([fake_label if n else affected_class for n in is_noisy])
    
    return data
    

###################################
# Imbalanced MNIST
###################################

def get_labels_and_class_counts(labels_list):
    '''
    Calculates the counts of all unique classes.
    '''
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts
    
class ImbalancedMNIST(Dataset):
    def __init__(self, digits, props, transform, 
                train = True, download = True, root='./data'):
        
        if any([p < 0.0 for p in props]):
            raise ValueError("Negative proportion of samples")
        
        if np.round(sum(props),2) != 1.0:
            raise Warning("Proportions do not sum to 1")
        
        self.props = props
        self.digits = digits
        self.n_digits = len(digits)
        self.train = train
        
        self.dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transform)
        
        # Resample the dataset
        self.dataset = self.imbal_resample()

    def imbal_resample(self):
        '''
        Resample the indices to create an artificially imbalanced dataset.
        '''
                
        aux_labels = self.dataset.train_labels if self.train else self.dataset.test_labels
        labels, cl_counts = get_labels_and_class_counts(aux_labels)
        cl_counts = cl_counts[self.digits] # only "digits" matter
        
        # All samples from the class with the largest prop are used
        largest_cl_prop = max(self.props)
        largest_cl_idx = [p == largest_cl_prop for p in self.props]
        largest_cl_count = min(cl_counts[largest_cl_idx])
        
        dataset_size = int(largest_cl_count / largest_cl_prop) # rule of three
        self.imbal_cl_counts = [int(p * dataset_size) for p in self.props]
                
        # Subsample within each class
        class_idx = [np.where(labels == d)[0] for d in self.digits] 
        imbal_idx_list = []
        for idx,count in zip(class_idx, self.imbal_cl_counts):
            # sample randomly
            imbal_idx = np.random.choice(idx, size = count, replace = False)
            imbal_idx_list.append(imbal_idx)
        
        imbal_idx_list = np.hstack(imbal_idx_list)
        
        return torch.utils.data.Subset(self.dataset, imbal_idx_list) 
    
    def __getitem__(self, index):
        item, target = self.dataset[index]
        return item, target

    def __len__(self):
        return len(self.dataset)

