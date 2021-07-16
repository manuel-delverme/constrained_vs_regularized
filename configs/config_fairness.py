import experiment_buddy
import torch_constrained

import torch
from torchvision import transforms, datasets
from torch import nn

random_seed = 1111

# -------------------------------------------------------- Data
# Dataset
dataset_class = datasets.MNIST

classes = [0,1,7,9]
props = [0.33,0.01,0.33,0.33]

# Image details and DataLoader
im_dim=28   
im_channels=1
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
batch_size = None 
val_batch_size = 1000
num_workers = 1

# ------------------------------------------------------ Application

# MLP layers
hid_dims = [200,200,200]
act_fun = nn.ReLU
act_args = []

# Conv layers
conv_channels = None
kernel_size = 3
stride = 1
pool_k_size = 2

# Objective function and constraints
balanced_ERM = False
fairness = False
const_classes = [1]

# --------------------------------------------------- torch constrained
# Constrained optimizer
le_levels = None
eq_levels = None
model_lr = 1e-3
dual_lr = 1e-2
log_constraints = True 
optimizer_class = torch.optim.Adam

#----------------------------------------------------- training
stop_delta = 1e-4
stop_patience = 5

max_epochs = 50
auto_scale_batch_size = True
min_epochs = 10
log_every_n_steps = 1 
log_gpu_memory = False
gpus = -1 # set to 0 if cpu use is prefered,

# Experiment Buddy
experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy() # sweep_yaml = ""