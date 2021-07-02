from torch.nn.modules.batchnorm import BatchNorm2d
from lit_constrained import LitConstrained
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torchmetrics import MetricCollection

################################################################
###### Helpers

def mlp_block(in_feat,out_feat,act_fun,act_args):
    # General linear block for MLPs
    layers = [
        nn.Linear(in_feat, out_feat),
        nn.BatchNorm1d(out_feat),
        act_fun(*act_args)
        ] 
    return layers

def conv_block(in_channels, out_channels, kernel_size, stride, 
            pool_k_size, act_fun, act_args):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        act_fun(*act_args),
        nn.MaxPool2d(pool_k_size),
        ] 
    return layers 

def get_conv_dim(w, kernel_size, stride, pool_k_size):
    w_conv = (w - kernel_size)/stride + 1
    w_pool = (int(w_conv) - pool_k_size)/pool_k_size + 1 
    return int(w_pool)

####################################3
# Constrained optimization module for classification

class LitConstrainedClassifier(LitConstrained):
    def __init__(
        self, im_dim, im_channels, hid_dims, act_fun, act_args, 
        const_classes, classes, balanced_ERM=False, fairness=False, 
        conv_channels = None, conv_kwargs = {},
        constrained_kwargs = {}
        ):
        super().__init__(**constrained_kwargs) 
        
        self.classes = classes
        self.const_classes = const_classes
        self.const_classes_idx = [classes.index(d) for d in const_classes]
        self.balanced_ERM = balanced_ERM
        self.fairness = fairness
        
        ## Initialize classifier 
        self.out_dim = len(classes)
        
        modules = []
        
        if conv_channels is not None:
            # Convolutional Layers
            in_channels = im_channels
            w = im_dim
            for out_channels in conv_channels:
                modules = modules + conv_block(in_channels, out_channels,
                    act_fun=act_fun, act_args=act_args, **conv_kwargs)
                
                in_channels = out_channels
                w = get_conv_dim(w, **conv_kwargs) # dimension of the conv output
        
        # Perceptron layers
        modules = modules + [torch.nn.Flatten()]
        
        d_in = out_channels * w**2 # flat dimension
        for h_dim in hid_dims:
            modules = modules + mlp_block(d_in,h_dim,act_fun,act_args)
            # modules.append(nn.Dropout(0.25))
            d_in = h_dim
        modules.append(nn.Linear(d_in, self.out_dim))
        self.model = nn.Sequential(*modules)
        
        # Set metrics - hardcoded. Check micro/macro flag
        self.metrics = MetricCollection([
            tm.Accuracy(num_classes=self.out_dim, average="macro"), # Balanced
            tm.F1(num_classes=self.out_dim, average="micro"),
        ])        
    
    def forward(self, batch):
        # Add support for conv-nets
        out = self.model(batch)
        return out
        
    def eval_losses(self, batch):
        x, y = batch
        
        # Forward
        preds = self(x)
        
        # Indices of batch belonging to each class
        class_dict = utils.get_class_dict(y, self.classes) # inefficient
        labels = utils.get_ordinal_labels(y,class_dict)
        
        loc = []
        loss = 0
        for class_ix in range(self.out_dim):
            cl = self.classes[class_ix]
            cl_idx = class_dict[cl]
            
            # y is mapped from class to range(self.out_dim) 
            class_loss = F.cross_entropy(preds[cl_idx], labels[cl_idx])
            
            if torch.isnan(class_loss): 
                # If no data was available, do not estimate the loss
                class_loss = torch.tensor(0.).to(self.device)
            
            loc.append(class_loss)
            
            # Agreggated loss
            if self.balanced_ERM: loss += class_loss/(len(self.classes))
            else: loss += len(cl_idx)/len(x) * class_loss
        
        # Only return losses on bounded_classes
        loc = torch.stack([loc[idx] for idx in self.const_classes_idx])
        
        if self.fairness: 
            # Gap between bounded loss and aggregated loss
            for i in range(loc.shape[1]):
                loss_other_cl = loss - loc[i] # Weighting?
                loc[i] = torch.abs(loc - loss_other_cl)
        
        return loss, loc, None
        
    def log_metrics(self, batch):
        x, y = batch
        # Forward
        logits = F.softmax(self(x), dim=1)
        # map y to range(output_dim)
        class_dict = utils.get_class_dict(y, self.classes)
        labels = utils.get_ordinal_labels(y,class_dict)
        
        metric_dict = self.metrics(logits, labels)
        self.log_dict(metric_dict, on_epoch=True)