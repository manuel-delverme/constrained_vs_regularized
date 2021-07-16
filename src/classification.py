"""
A classifier implemented on Pytorch with Pytorch Lightning. 
"""

from src.lit_constrained import LitConstrained
from src import nn_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torchmetrics import MetricCollection

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

class LitConstrainedClassifier(LitConstrained):
    """
    Implements an image classifier under pytorch lightning with a 
    torch_constrained optimizer. Instead of doing ERM, the loss for each 
    class is modeled independently.
    """
    def __init__(
        self, im_dim, im_channels, hid_dims, act_fun, act_args, 
        const_classes, classes, kernel_size, stride, pool_k_size,
        conv_channels = None, balanced_ERM=False, fairness=False,
        constrained_kwargs = {}
        ):
        """
        Initializes the model given its architecture. 
        
        Args:
            im_dim (scalar int): dimension of input images.
            im_channels (scalar int): number of image channels.
            hid_dims (int list): hidden dimensions of the MLP classifier. 
            act_fun (torch.nn activation): class for the activation.
            act_args (list): arguments for the constructor of the activation.
            const_classes (list): classes whose loss is to be constrained. 
            classes (list): class labels of the dataset. 
            balanced_ERM (bool, optional): whether the main objective for
                training is the empirical risk, or its balanced alternative 
                where each class is given equal importance. Defaults to False.
            fairness (bool, optional): whether constraints are constructed in 
                a fairness style, where provided losses are bounded to be 
                close to the ERM of the remaining classes. Otherwise, losses 
                are bounded to be below a threshold. Defaults to False, the latter.
            conv_channels (list, optional): channels for the convolutional 
                layers of the network. Defaults to None.
            kernel_size  (scalar int): kernel size of convolutional layers.
            stride  (scalar int): stride of convolutional layers.
            pool_k_size  (scalar int, optional): kernel size of max pooling layers.
                If not provided, no max pooling layer is included.
            constrained_kwargs (dict, optional): arguments for the initialization
                of the LitConstrained class. Defaults to {}, where a single 
                objective optimization problem is considered during training.
        """
        super().__init__(**constrained_kwargs) 
        
        self.classes = classes
        self.const_classes = const_classes
        self.const_classes_idx = [classes.index(d) for d in const_classes]
        self.balanced_ERM = balanced_ERM
        self.fairness = fairness        
        self.out_dim = len(classes)
        
        # -------------------------------------- Conv layers
        if conv_channels is not None:
            conv_layers, out_channels, out_dim, _ = nn_utils.build_convnet(
                im_channels, im_dim, conv_channels, kernel_size, stride,
                pool_k_size, act_fun, act_args
                ) 
        else: 
            conv_layers, out_channels, out_dim = [], im_channels, im_dim
        
        conv_layers.append(nn.Flatten())
        # -------------------------------------- Perceptron layers
        d_in = out_channels * out_dim**2 # flat dimension
        
        if hid_dims is not None:
            mlp_layers = nn_utils.build_mlp(d_in,hid_dims,act_fun,act_args)
            last_dim = hid_dims[-1]
        else:
            mlp_layers = []
            last_dim = d_in
        
        mlp_layers.append(nn.Linear(last_dim, self.out_dim))
        self.model = nn.Sequential(*conv_layers, *mlp_layers)
        
        # Set metrics - hardcoded. Check micro/macro flag
        self.metrics = MetricCollection([
            tm.Accuracy(num_classes=self.out_dim, average="micro"), # not Balanced
            tm.F1(num_classes=self.out_dim, average="micro"),
        ])        
    
    def forward(self, input):
        """
        Produce the scores for every class (before softmax).
        
        Args:
            input (4D tensor): (n_samples, im_size, im_size, channels) input

        Returns:
            2D tensor: (n_samples, n_classes) predictions.
        """
        out = self.model(input)
        return out
        
    def eval_losses(self, batch):
        """
        Evaluate the cross entropy loss of predictions for every class. 

        Args:
            batch (tensor or structure of tensors): a batch.

        Returns:
            tuple of tensors: the ERM/aggregated loss, losses per
            class (for those in "const_classes") and None.
        """
        x, y = batch
        
        # Forward
        preds = self(x)
        
        # Indices of batch belonging to each class
        class_dict = get_class_dict(y, self.classes) # inefficient
        labels = get_ordinal_labels(y,class_dict)
        
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
        
        return loss, None, loc
        
    def log_metrics(self, batch):
        """
        Logs a selection of metrics estimated from the model state for the
        given batch.
        """
        x, y = batch
        # Forward
        logits = F.softmax(self(x), dim=1)
        # map y to range(output_dim)
        class_dict = get_class_dict(y, self.classes)
        labels = get_ordinal_labels(y,class_dict)
        
        metric_dict = self.metrics(logits, labels)
        self.log_dict(metric_dict, on_epoch=True, on_step=False)