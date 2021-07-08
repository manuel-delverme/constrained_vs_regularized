import dataset_utils
import experiment_buddy

from torchvision import transforms
from torch import nn

random_seed = 1111

# -------------------------------------------------------- Data
# Dataset
digits = [0,1,7,9]
classes = digits
props = [0.33,0.01,0.33,0.33]
data_class = dataset_utils.ImbalancedMNIST

# Image details and DataLoader
im_dim=28   
im_channels=1
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
batch_size = 256 
val_batch_size = 1000
num_workers = 12

# ------------------------------------------------------ Application

# MLP layers
hid_dims = [32]
act_fun = nn.ReLU
act_args = []

# Conv layers
conv_channels = [32]
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
dual_lr = 0.
log_constraints = True 

#----------------------------------------------------- training
stop_delta = 1e-4
stop_patience = 3

max_epochs = 10
auto_scale_batch_size = True
min_epochs = 10
log_every_n_steps = 1 
log_gpu_memory = False
gpus = -1 # set to 0 if cpu use is prefered,


# Experiment Buddy
experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy()