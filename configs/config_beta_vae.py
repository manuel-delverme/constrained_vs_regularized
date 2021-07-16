import experiment_buddy

import torch
import torch_constrained
from torchvision import transforms, datasets
from torch import nn

random_seed = 1111

# -------------------------------------------------------- Data
dataset_class = datasets.MNIST

classes = [0,1,2,3,4,5,6,7,8,9]
im_dim = 28   
im_channels = 1
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
batch_size = 256 
val_batch_size = 1000
num_workers = 1

# ------------------------------------------------------ Application

beta = 1 # standard VAE. 
latent_dim = 2

# MLP layers
hid_dims = [200, 200, 200]
act_fun = nn.ReLU
act_args = []

# Conv layers
conv_channels = None
kernel_size = 3
stride = 1
pool_k_size = None

# --------------------------------------------------- torch constrained
# Constrained optimizer
le_levels = [0.01]
eq_levels = None
model_lr = 1e-2
dual_lr = 1e-1
log_constraints = True 
optimizer_class = torch_constrained.ExtraAdam

#----------------------------------------------------- training
stop_delta = 1e-4
stop_patience = 5

max_epochs = 20
auto_scale_batch_size = True
min_epochs = 10
log_every_n_steps = 1 
log_gpu_memory = False
gpus = -1 # set to 0 if cpu use is prefered,


# Experiment Buddy
experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy()